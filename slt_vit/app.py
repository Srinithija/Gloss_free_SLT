"""Streamlit demo for the ViT translator. Launch with:  python run_app.py   (http://localhost:8502)"""
import os
import sys
import tempfile
import traceback

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slt_vit import paths  # noqa: E402
from slt_vit.frames import VIDEO_EXTENSIONS  # noqa: E402
from slt_vit.infer import Translator  # noqa: E402

SELECTED = os.path.join(paths.OUTPUTS, 'SELECTED_CHECKPOINT.txt')


def default_checkpoint():
    """The checkpoint chosen on validation, recorded in outputs/SELECTED_CHECKPOINT.txt (first line)."""
    if os.path.isfile(SELECTED):
        with open(SELECTED, encoding='utf-8') as f:
            return os.path.join(paths.PROJECT, f.readline().strip())
    return ''


@st.cache_resource(show_spinner='Loading the trained model...')
def load_translator(path):
    return Translator(path)


def run_translation(translator, filename, data, beams=1):
    """Returns (result, None, None) or (None, message for the user, technical details)."""
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(filename)[1].lower())
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        return translator.translate_video(path, beams), None, None
    except (ValueError, FloatingPointError) as e:
        return None, f'Could not translate this video: {e}', None
    except Exception:  # noqa: BLE001 - shown to the user instead of crashing the page
        return None, 'Translation failed unexpectedly.', traceback.format_exc()
    finally:
        os.remove(path)


def main():
    st.set_page_config(page_title='ISL to English (ViT)', layout='centered')
    st.title('Indian Sign Language to English')
    st.caption('Experimental ViT version: pretrained SigLIP/CLIP frame encoder, temporal convolution, mBART decoder.')
    with st.sidebar:
        ckpt = st.text_input('Checkpoint', os.environ.get('SLT_VIT_CKPT', default_checkpoint()))
        show = st.checkbox('Show sampled frames and timings')
        st.info('Trained on 97 everyday sentences from 5 signers. It can only produce those sentences, and accuracy on '
                'people outside the corpus is low. Outputs are the most likely sentence, not a calibrated confidence.')
    if not ckpt or not os.path.isfile(ckpt):
        st.error(f'No trained checkpoint found ({ckpt or "none selected yet"}). Train and select one first (see README).')
        return
    try:
        tr = load_translator(ckpt)
    except Exception as e:  # noqa: BLE001
        st.error(f'Could not load the model: {e}')
        return
    with st.sidebar:
        v = tr.info.get('val') or {}
        st.caption(f"{tr.info['hf_id']} | epoch {tr.info['epoch_number']} | {tr.device.type} | "
                   f"validation exact {v.get('exact_match', float('nan')):.0%}")
    up = st.file_uploader('Upload a sign language video', type=[e.lstrip('.') for e in VIDEO_EXTENSIONS])
    if up is None:
        return
    data = up.getvalue()
    st.video(data)
    if not st.button('Translate', type='primary'):
        return
    bar = st.progress(0, text='Decoding the video and running the model...')
    result, message, details = run_translation(tr, up.name, data)
    bar.empty()
    if result is None:
        st.error(message)
        if details:
            with st.expander('Error details'):
                st.code(details)
        return
    if result['text']:
        st.success(f"**Predicted sentence:** {result['text']}")
    else:
        st.warning('The model produced an empty sentence for this video.')
    if not result['stopped_at_eos']:
        st.warning('Generation hit the length limit without an end-of-sentence token.')
    if show:
        info = result['video']
        st.write(f"{info['decoded_frames']} frames decoded ({info['width']}x{info['height']}, {info['fps']:.1f} fps); "
                 f"{len(result['sampled_indices'])} sampled uniformly; model {result['model_seconds']:.2f}s, total {result['total_seconds']:.2f}s.")
        idx = result['sampled_indices']
        picks = np.linspace(0, len(idx) - 1, 8).round().astype(int)
        st.image([result['frames'][idx[i]] for i in picks], width=80, caption=[f'frame {idx[i]}' for i in picks])


if __name__ == '__main__':
    main()
