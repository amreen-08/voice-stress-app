"""Shared code for feature extraction, training and the demo app."""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import librosa
import opensmile
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

SR = 16000
TARGET_RMS = 0.05
_smile = None


def _get_smile():
    global _smile
    if _smile is None:
        _smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _smile


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def load_audio(path):
    """Load any common audio format as 16 kHz mono.

    Phone recordings (m4a, mp4, opus) need ffmpeg. If the normal loader fails, this
    converts the file with ffmpeg, using the copy installed by `pip install imageio-ffmpeg`
    when ffmpeg is not on the system PATH.
    """
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        return y
    except Exception as first_error:
        exe = _ffmpeg_exe()
        if exe is None:
            raise RuntimeError(
                f"Could not decode {path}. Run:  pip install imageio-ffmpeg  and try again."
            ) from first_error
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clip.wav")
            r = subprocess.run([exe, "-y", "-i", str(path), "-ac", "1", "-ar", str(SR), out],
                               capture_output=True)
            if r.returncode != 0:
                msg = r.stderr.decode(errors="ignore")[-300:]
                raise RuntimeError(f"ffmpeg could not read {path}: {msg}") from first_error
            y, _ = librosa.load(out, sr=SR, mono=True)
        return y


def snr_db(y):
    """Rough noise check on a raw clip: loud frames against the quietest 10% of frames.

    Not an exact signal-to-noise ratio. Use it only to spot clips recorded in noisy rooms
    (below about 20 dB), and to compare clips with each other.
    """
    frame = int(0.025 * SR)
    n = len(y) // frame
    if n < 20:
        return float("nan")
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1))
    noise, speech = np.percentile(rms, 10), np.percentile(rms, 90)
    if noise <= 0:
        return 60.0
    return float(20 * np.log10(speech / noise))


FREE_WINDOW = 45.0      # seconds of free speech analysed for every student
MIN_PART_SECONDS = 2.0  # takes shorter than this are ignored


def prepare(y, max_seconds=None):
    """Trim leading and trailing silence, optionally keep only the first max_seconds,
    then scale to a fixed RMS.

    Fixing the RMS removes differences caused by phone gain and mic distance,
    so absolute loudness is not used as a feature. Loudness variation inside a
    recording is still captured.
    """
    y, _ = librosa.effects.trim(y, top_db=35)
    if max_seconds is not None:
        y = y[: int(max_seconds * SR)]
    rms = np.sqrt(np.mean(y ** 2)) if len(y) else 0.0
    return y * (TARGET_RMS / rms) if rms > 0 else y


TIMING_KEYS = ["duration", "speech_ratio", "n_pauses", "pauses_per_min",
               "mean_pause", "max_pause", "onset_rate"]


def timing_parts(y, sr=SR):
    """Raw counts for one take. Keeping them raw lets several takes be combined
    without inventing a pause at each join."""
    iv = librosa.effects.split(y, top_db=30, frame_length=1024, hop_length=256)
    gaps = [(iv[i + 1][0] - iv[i][1]) / sr for i in range(len(iv) - 1)]
    return {
        "dur": len(y) / sr,
        "speech": sum(e - s for s, e in iv) / sr,
        "pauses": [g for g in gaps if g >= 0.25],
        "onsets": len(librosa.onset.onset_detect(y=y, sr=sr)),
    }


def timing_features(parts):
    dur = sum(p["dur"] for p in parts)
    if dur == 0:
        return {k: np.nan for k in TIMING_KEYS}
    speech = sum(p["speech"] for p in parts)
    pauses = [g for p in parts for g in p["pauses"]]
    onsets = sum(p["onsets"] for p in parts)
    return {
        "duration": dur,
        "speech_ratio": speech / dur,
        "n_pauses": len(pauses),
        "pauses_per_min": len(pauses) / dur * 60,
        "mean_pause": float(np.mean(pauses)) if pauses else 0.0,
        "max_pause": float(max(pauses)) if pauses else 0.0,
        "onset_rate": onsets / speech if speech > 0 else 0.0,
    }


def clip_features(y, prefix, max_seconds=None):
    """88 eGeMAPS features plus 7 timing features.

    y is one recording, or a list of separate takes that belong together (for example three
    answers recorded one after another). Each take is analysed on its own and the results are
    combined, weighted by length, so joining takes never creates a fake pause or pitch jump.
    max_seconds keeps only the first part of the speech, so every student is judged on the
    same amount of audio.
    """
    takes = list(y) if isinstance(y, (list, tuple)) else [y]
    parts, used = [], 0.0
    for raw in takes:
        left = None if max_seconds is None else max_seconds - used
        if left is not None and left <= 0:
            break
        p = prepare(raw, left)
        if len(p) / SR < MIN_PART_SECONDS:
            continue
        parts.append(p)
        used += len(p) / SR
    if not parts:
        return {}
    w = np.array([len(p) for p in parts], float)[:, None]
    eg = pd.DataFrame([_get_smile().process_signal(p, SR).iloc[0].to_dict() for p in parts])
    vals = eg.to_numpy(float)
    ok = np.isfinite(vals)
    den = (w * ok).sum(axis=0)
    num = np.nansum(vals * w, axis=0)
    egemaps = dict(zip(eg.columns, np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)))
    feats = {**egemaps, **timing_features([timing_parts(p) for p in parts])}
    return {f"{prefix}_{k}": float(v) for k, v in feats.items()}


def clipping_fraction(y_raw):
    return float(np.mean(np.abs(y_raw) > 0.99)) if len(y_raw) else 0.0


GENDER_CODE = {"M": 0, "F": 1}


def gender_to_code(g):
    return GENDER_CODE.get(str(g).strip().upper()[:1], -1)


class GenderStandardizer(BaseEstimator, TransformerMixin):
    """Z-score each feature within gender, using training data only.

    Input X has the gender code in column 0 and features in the rest. Pitch and
    formants differ a lot between male and female voices, so this stops the model
    from simply learning gender. A gender group with fewer than 5 people falls
    back to the overall mean and standard deviation.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        g, F = X[:, 0], X[:, 1:]
        self.mu_all_ = np.nanmean(F, axis=0)
        sd = np.nanstd(F, axis=0)
        sd[sd == 0] = 1.0
        self.sd_all_ = sd
        self.stats_ = {}
        for code in np.unique(g):
            m = g == code
            if code >= 0 and m.sum() >= 5:
                s = np.nanstd(F[m], axis=0)
                s[s == 0] = 1.0
                self.stats_[code] = (np.nanmean(F[m], axis=0), s)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        g, F = X[:, 0], X[:, 1:]
        out = np.empty_like(F)
        for code in np.unique(g):
            m = g == code
            mu, sd = self.stats_.get(code, (self.mu_all_, self.sd_all_))
            out[m] = (F[m] - mu) / sd
        return np.nan_to_num(out, nan=0.0)


class KerasMLP(BaseEstimator, ClassifierMixin):
    """Small TensorFlow network with a scikit-learn interface (binary labels)."""

    def __init__(self, hidden=16, dropout=0.3, l2=1e-2, epochs=150, lr=1e-3, seed=0):
        self.hidden = hidden
        self.dropout = dropout
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed)
        X = np.asarray(X, "float32")
        self.classes_ = np.unique(y)
        y01 = (np.asarray(y) == self.classes_[1]).astype("float32")
        model = tf.keras.Sequential([
            tf.keras.layers.Input((X.shape[1],)),
            tf.keras.layers.Dense(self.hidden, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(self.l2)),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="binary_crossentropy")
        pos = y01.mean()
        weights = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
        model.fit(X, y01, epochs=self.epochs, batch_size=16, verbose=0, class_weight=weights)
        self.model_ = model
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, "float32"), verbose=0).ravel()
        return np.c_[1 - p, p]

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
