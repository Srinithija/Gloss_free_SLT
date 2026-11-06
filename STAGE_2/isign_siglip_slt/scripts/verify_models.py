"""Download (if needed), pin and verify the pretrained models; one GPU forward pass each.

Writes configs/model_revisions.json and reports/models_verified.json. Stops with an error if any
model fails to load its pretrained weights (never falls back to random init).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.encoders import MBART, MODELS, load_vision_encoder, resolve_revision  # noqa: E402
from slt.paths import PROJECT, REPORTS  # noqa: E402


def main():
    revs, out = {}, {}
    for name, repo in list(MODELS.items()) + [("mbart", MBART)]:
        for attempt in range(5):
            try:
                revs[repo] = resolve_revision(repo)
                break
            except Exception as e:  # noqa: BLE001 - flaky network: retry, then fail loudly
                print(f"revision lookup {repo} failed ({e}); retry {attempt + 1}", flush=True)
                time.sleep(10)
        else:
            raise SystemExit(f"could not resolve revision of {repo}")
    (PROJECT / "configs").mkdir(exist_ok=True)
    (PROJECT / "configs" / "model_revisions.json").write_text(json.dumps(revs, indent=1))

    for name in MODELS:
        model, mean, std, info = load_vision_encoder(name, "cuda", revision=revs[MODELS[name]])
        x = (torch.rand(4, 3, 224, 224, device="cuda") - mean) / std
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            y = model(pixel_values=x).pooler_output
        info |= {"forward_ok": bool(torch.isfinite(y).all()), "pooler_shape": list(y.shape)}
        out[name] = info
        print(name, json.dumps(info), flush=True)
        del model
        torch.cuda.empty_cache()

    from transformers import MBartForConditionalGeneration, MBartTokenizer
    rev = revs[MBART]
    tok = MBartTokenizer.from_pretrained(MBART, revision=rev, src_lang="en_XX", tgt_lang="en_XX")
    m, loading = MBartForConditionalGeneration.from_pretrained(MBART, revision=rev, output_loading_info=True)
    if loading["missing_keys"] or loading["mismatched_keys"]:
        raise SystemExit(f"mBART weights did not load cleanly: {loading}")
    m = m.cuda().eval()
    ids = tok("This is a test sentence.", return_tensors="pt").to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = m(**ids, decoder_input_ids=ids["input_ids"][:, :4])
    out["mbart"] = {"repo": MBART, "revision": rev, "params": sum(p.numel() for p in m.parameters()),
                    "vocab": len(tok), "en_XX_id": tok.convert_tokens_to_ids("en_XX"),
                    "default_generation_num_beams": m.generation_config.num_beams,
                    "forward_ok": bool(torch.isfinite(o.logits).all()), "missing_keys": loading["missing_keys"]}
    print("mbart", json.dumps(out["mbart"]), flush=True)
    out["peak_vram_GiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    (REPORTS / "models_verified.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
