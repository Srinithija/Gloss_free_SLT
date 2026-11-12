"""Streamlit front end for the trained iSign video -> English model (real checkpoint, no mock).

Launch:  isign_opt\\run_app.bat
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from isign_opt.infer import DEFAULT_CKPT, VIDEO_EXTS, Translator

st.set_page_config(page_title="iSign Translator", layout="centered")
st.title("Indian Sign Language → English")
st.caption("Gloss-free translation: frozen ResNet18 frame features → temporal conv → 3+3-layer mBART (iSign).")


@st.cache_resource(show_spinner="Loading model and tokenizer…")
def get_translator(ckpt: str):
    return Translator(ckpt)


ckpt = st.sidebar.text_input("Checkpoint", str(DEFAULT_CKPT))
show_frames = st.sidebar.checkbox("Show sampled frames (debug)", value=False)
try:
    tr = get_translator(ckpt)
    st.sidebar.success(f"Loaded epoch {tr.ckpt_epoch} · decoding: {tr.decoding}")
except Exception as e:  # shown to the user, not swallowed
    st.error(f"Could not load the model: {e}")
    st.stop()

up = st.file_uploader("Upload a sign-language video", type=[e.strip(".") for e in VIDEO_EXTS])
if up is not None:
    st.video(up.getvalue())
    if st.button("Translate", type="primary"):
        suffix = Path(up.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(up.getvalue())
            tmp = f.name
        try:
            with st.status("Translating…", expanded=False) as s:
                s.write("Decoding and sampling frames, extracting features, generating text…")
                res = tr.translate(tmp, return_frames=show_frames)
                s.update(label="Done", state="complete")
            st.subheader("Predicted English sentence")
            st.markdown(f"### {res['prediction'] or '(empty output)'}")
            st.caption(f"{res['original_frames']} frames in video → {res['sampled_frames']} sampled · "
                       f"decode {res['decode_seconds']} s · model {res['model_seconds']} s")
            if show_frames:
                fr = res["frames"]
                idx = np.linspace(0, len(fr) - 1, min(12, len(fr)), dtype=int)
                st.image([fr[i] for i in idx], width=110, caption=[f"#{i}" for i in idx])
        except (ValueError, FileNotFoundError) as e:
            st.error(f"Unsupported or invalid video: {e}")
        except Exception as e:
            st.error(f"Translation failed: {type(e).__name__}: {e}")
        finally:
            Path(tmp).unlink(missing_ok=True)
st.info("Model output is a machine translation of open-domain signing and is often inaccurate; see the experiment report for measured quality.")
