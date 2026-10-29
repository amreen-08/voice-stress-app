"""Demo app. Run from the src folder:   streamlit run app.py

Needs ../models/model.joblib, which train.py creates. Nothing runs live except
feature extraction and one prediction, so no training and no internet are needed.
Recordings are processed in memory and a temporary file that is deleted straight away.
"""
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common import FREE_WINDOW, clip_features, load_audio, gender_to_code

BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
PASSAGE = ("On Saturday morning the market near the station was already busy. A man was arranging "
           "oranges in neat rows while his daughter counted the change in a small steel box. The first "
           "train of the day rolled in a little late, and people hurried past with bags and umbrellas. "
           "Someone laughed near the tea stall. The sky was grey but it did not rain, and by noon the "
           "crowd had thinned out. The oranges were nearly gone, the box was heavy with coins, and the "
           "two of them sat down to share what was left of the tea.")
FREE_PROMPT = "Describe a normal day in college for you, from morning to evening (about 40 seconds)."

st.set_page_config(page_title="Voice and stress", layout="wide")
st.title("Can a voice sample show signs of academic stress?")
st.caption("A class project screening tool. It is not a medical test and it does not diagnose anything.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run extract_features.py and train.py first.")
    st.stop()


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


bundle = get_model()
pipe, feats = bundle["pipeline"], bundle["feature_names"]
need = ["read", "free"] if bundle["use"] == "both" else [bundle["use"]]

with st.sidebar:
    st.header("About the model")
    st.write(f"Trained on {bundle['n_students']} students. Cross-validated balanced accuracy on "
             f"students it had not seen: {bundle['cv_balanced_accuracy']:.2f} (0.50 is chance).")
    st.write("One person's result cannot show whether the model is right. The cross-validated "
             "number above is the evidence.")
    gender = st.selectbox("Speaker's gender (used to adjust pitch ranges)", ["F", "M", "Other"])

clips = {}
cols = st.columns(len(need))
for col, kind in zip(cols, need):
    with col:
        st.subheader("1. Read aloud" if kind == "read" else "2. Speak freely")
        st.info(PASSAGE if kind == "read" else FREE_PROMPT)
        rec = st.audio_input("Record", key=f"rec_{kind}")
        up = st.file_uploader("or upload a clip", type=["wav", "m4a", "mp3", "flac", "ogg"], key=f"up_{kind}")
        clips[kind] = rec or up


def to_audio(file):
    suffix = Path(getattr(file, "name", "clip.wav")).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getvalue())
        path = tmp.name
    try:
        return load_audio(path)
    finally:
        os.remove(path)


if all(clips.get(k) is not None for k in need) and st.button("Analyse voice", type="primary"):
    with st.spinner("Extracting voice features..."):
        row = {}
        for kind in need:
            y = to_audio(clips[kind])
            if len(y) / 16000 < 8:
                st.warning(f"The {kind} clip is very short. The result will be unreliable.")
            window = bundle.get("free_window", FREE_WINDOW) if kind == "free" else None
            row.update(clip_features(y, kind, window))
        x = np.array([[gender_to_code(gender)] + [row.get(f, np.nan) for f in feats]], float)

    z = pipe.named_steps["gender"].transform(x)
    pc = pipe.named_steps["pca"].transform(z)[:, :2]
    p_high = float(pipe.predict_proba(x)[0, 1])

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        tp, tl = bundle["train_pca"], bundle["train_labels"]
        for lab, colr, name in [(0, BLUE, "Lower-stress students"), (1, ORANGE, "Higher-stress students")]:
            m = tl == lab
            ax.scatter(tp[m, 0], tp[m, 1], s=30, color=colr, edgecolor=SURFACE, linewidth=1.2, label=name, alpha=0.9)
        ax.scatter(pc[0, 0], pc[0, 1], s=260, marker="*", color=INK, edgecolor=SURFACE, linewidth=1.5, label="This voice", zorder=5)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(color="#e4e3df", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False)
        st.pyplot(fig)
    with right:
        side = "higher-stress" if p_high >= 0.5 else "lower-stress"
        st.subheader(f"Closer to the {side} group")
        st.metric("Model's leaning towards higher stress", f"{p_high * 100:.0f}%")
        st.caption("This describes where the voice sits among our students. It is a screening "
                   "aid only. A model with this accuracy is wrong about a fair share of people.")

    st.subheader("What stands out in this voice")
    tz, tl = bundle["train_z"], bundle["train_labels"]
    names = bundle["top_features"]
    idx = [feats.index(n) for n in names]
    table = pd.DataFrame({
        "feature": [n.replace("_", " ") for n in names],
        "this voice (z-score)": z[0, idx].round(2),
        "lower-stress group avg": tz[tl == 0][:, idx].mean(0).round(2),
        "higher-stress group avg": tz[tl == 1][:, idx].mean(0).round(2),
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("A z-score of 0 is typical for that gender in our sample. Positive means higher than typical.")
else:
    st.write("Record or upload the clip(s) above, then press Analyse voice.")
