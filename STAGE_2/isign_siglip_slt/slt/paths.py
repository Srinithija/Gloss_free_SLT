"""Fixed locations. Everything outside PROJECT is treated as read-only."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATASET = Path(r"E:\FINAL YEAR PROJECT\GLOSS FREE SLT\dataset\iSign")
VIDEOS = DATASET / "videos" / "iSign-videos_v1.1"
SPLITS = DATASET / "splits" / "v1"
LMDB_ROOT = Path(r"D:\iSign_LMDB")
MBART_TRIMMED = Path(r"E:\FINAL YEAR PROJECT\GLOSS FREE SLT\mbart_models\MBart_trimmed")
OLD_EXP = Path(r"D:\iSign_Experiments\GFSLT\optimized_v2_20261023")

REPORTS = PROJECT / "reports"
MANIFESTS = PROJECT / "manifests"
CACHE = PROJECT / "cache"
LOGS = PROJECT / "logs"
SPLIT_NAMES = ("train", "dev", "test")
