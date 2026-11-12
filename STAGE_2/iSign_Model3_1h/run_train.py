"""Run isign_siglip_slt/scripts/train.py unchanged, but with runs/ and configs/ resolved inside iSign_Model3_1h.
A cache entry may also be a list of stores (see stores.py) to merge the K=32 cache with re-indexed pilot caches."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "isign_siglip_slt"))
sys.path.insert(0, str(HERE))
import scripts.train as T  # noqa: E402
import stores  # noqa: E402
T.PROJECT = HERE  # runs/<name>, configs/model_revisions.json; all data paths in configs are absolute
_orig = T.make_store


def make_store(cfg, key):
    src = cfg.get(key)
    if isinstance(src, list):
        return stores.build(src, cfg["keyframes_json"])
    return _orig(cfg, key)


T.make_store = make_store
if __name__ == "__main__":
    T.main()
