"""Streamlit demo of the improved ISL translator (E3) with a confidence score.

    env\\python.exe experiments\\prediction_improvement_v1\\run_demo.py      ->  http://localhost:8503
"""
import os
import sys
import tempfile
import traceback

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi_exp import common  # noqa: E402
from pi_exp.demo import DemoPredictor  # noqa: E402
from pi_exp.preview import browser_preview  # noqa: E402
from slt_vit import paths  # noqa: E402
from slt_vit.dataset import cache_name  # noqa: E402
from slt_vit.frames import VIDEO_EXTENSIONS, read_video  # noqa: E402

TAG = 'E3'
RESULT = {'val': '74/94 = 78.7%', 'loso': '211/282 = 74.8%', 'test': '59/91 = 64.8%'}


@st.cache_resource(show_spinner='Loading models (first time only)...')
def predictor(tag):
    return DemoPredictor(tag)


@st.cache_data(show_spinner='Preparing the video preview...', max_entries=4)
def preview_for(data, name):
    return browser_preview(data, name)


@st.cache_data(show_spinner=False)
def clip_list(split):
    es = common.entries_for(split, allow_test=split == 'test')
    return {e['text']: e for e in sorted(es, key=lambda e: e['text'])}


def show_result(p, probs, reference=None):
    pred, conf = p.top(probs, 1)[0]
    st.markdown(f"### {pred}")
    st.metric('Confidence', f'{100 * conf:.1f}%')
    if reference is not None:
        if pred == reference:
            st.success(f'Correct. The real sentence is "{reference}".')
        else:
            rank = int(np.sum(probs > probs[p.texts.index(reference)])) + 1
            st.error(f'Wrong. The real sentence is "{reference}" (ranked #{rank}, {100 * probs[p.texts.index(reference)]:.1f}%).')
    if conf >= 0.8:
        st.caption('High confidence: on the test signer, 10 of 11 predictions at 80% or more were correct.')
    elif conf < 0.3:
        st.caption('Low confidence: the model is unsure about this sentence.')


def main():
    st.set_page_config(page_title='ISL to English: improved demo', layout='centered')
    st.title('Indian Sign Language → English')
    with st.sidebar:
        st.markdown('**Model:** SigLIP + CLIP + hand/body landmarks (E3)')
        r = RESULT
        st.markdown(f"**Measured exact-sentence accuracy**  \nvalidation signer 6: {r['val']}  \nsigners 3–5: {r['loso']}  \n"
                    f"test signer 7 (repeat evaluation): {r['test']}  \noriginal model on signer 7: 23/91 = 25.3%")
        st.info('The model chooses one of the 97 sentences it was trained on; it cannot produce new sentences. '
                'Confidence = how strongly it prefers that sentence over the other 96, not a guarantee.')
    try:
        p = predictor(TAG)
    except Exception as e:  # noqa: BLE001
        st.error(f'Could not load the models: {e}')
        return
    upload, dataset = st.tabs(['Upload a video', 'Try a clip from the dataset'])
    with upload:
        up = st.file_uploader('Sign language video', type=[e.lstrip('.') for e in VIDEO_EXTENSIONS])
        if up is not None:
            video, detail = preview_for(up.getvalue(), up.name)
            if video is None:
                st.warning(f'Could not show a preview: {detail}.')
            else:
                st.video(video, format=detail)
            if st.button('Translate video', type='primary'):
                fd, path = tempfile.mkstemp(suffix=os.path.splitext(up.name)[1].lower(), dir=common.out_dir('demo_tmp'))
                with os.fdopen(fd, 'wb') as f:
                    f.write(up.getvalue())
                try:
                    with st.spinner('Loading...'):
                        frames, _ = read_video(path, 224)
                        lm, _, _ = p.landmarks_from_video(path, len(frames)) if p.uses_landmarks else (None, None, 0)
                        probs, _ = p.scores(frames, lm)
                    show_result(p, probs)
                except ValueError as e:
                    st.error(f'Could not read this video: {e}')
                except Exception:  # noqa: BLE001
                    st.error('Translation failed.')
                    st.code(traceback.format_exc())
                finally:
                    os.remove(path)
    with dataset:
        which = st.radio('Signer', ['Signer 6 (validation)', 'Signer 7 (test)'], horizontal=True)
        clips = clip_list('val' if which.startswith('Signer 6') else 'test')
        text = st.selectbox(f'Sentence ({len(clips)} clips)', list(clips))
        if st.button('Translate clip', type='primary'):
            e = clips[text]
            frames = np.load(os.path.join(paths.FRAME_CACHE, cache_name(e)))
            lm, _ = p.landmarks_from_cache(e, len(frames)) if p.uses_landmarks else (None, None)
            probs, _ = p.scores(frames, lm)
            show_result(p, probs, reference=e['text'])


if __name__ == '__main__':
    main()
