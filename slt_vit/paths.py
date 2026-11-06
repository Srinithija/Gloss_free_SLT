"""Every location this project reads or writes. Writes are only allowed inside the project folder."""
import glob
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read-only inputs
DATASET = r'C:\Users\SRINITHIJA\Desktop\Sign_processing\archive\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus'
OLD_MANIFEST = r'C:\Users\SRINITHIJA\Desktop\Sign_processing\sltbaselines\data\ISL-CSLRT\manifest.json'  # cross-check only
CC25_HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub', 'models--facebook--mbart-large-cc25')

# Project-owned outputs
DATA = os.path.join(PROJECT, 'data')
MANIFEST = os.path.join(DATA, 'manifest.json')
CACHE = os.path.join(PROJECT, 'cache')
HF_CACHE = os.path.join(CACHE, 'hf')                  # new downloads (CLIP / SigLIP)
FRAME_CACHE = os.path.join(CACHE, 'frames224')
FEATURE_CACHE = os.path.join(CACHE, 'features')       # one sub-folder per encoder
OUTPUTS = os.path.join(PROJECT, 'outputs')
TEST_LEDGER = os.path.join(OUTPUTS, 'TEST_EVALUATION_LEDGER.csv')


def writable(path):
    """Return ``path`` if it lies inside the project folder; refuse anything else."""
    real = os.path.realpath(path)
    root = os.path.realpath(PROJECT)
    if os.path.commonpath([real, root]) != root:
        raise PermissionError(f'refusing to write outside {PROJECT}: {path}')
    return path


def cc25_file(name):
    """A file from the pre-existing local mBART-cc25 download (read-only)."""
    found = glob.glob(os.path.join(CC25_HUB, 'snapshots', '*', name))
    if not found:
        raise FileNotFoundError(f'{name} of facebook/mbart-large-cc25 not found under {CC25_HUB}')
    return found[0]
