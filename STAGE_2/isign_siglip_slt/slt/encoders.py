"""Pretrained vision towers at pinned revisions. Never falls back to random weights."""
import torch

MODELS = {
    "siglip": "google/siglip-base-patch16-224",
    "clip": "openai/clip-vit-base-patch16",
}
MBART = "facebook/mbart-large-cc25"


def resolve_revision(repo_id):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo_id).sha


def load_vision_encoder(name, device="cuda", revision=None):
    """Return (frozen eval model, mean [1,3,1,1], std [1,3,1,1], info dict)."""
    from transformers import AutoImageProcessor, CLIPVisionModel, SiglipVisionModel

    repo = MODELS[name]
    cls = SiglipVisionModel if name == "siglip" else CLIPVisionModel
    model, loading = cls.from_pretrained(repo, revision=revision, output_loading_info=True)
    bad = {k: v for k, v in loading.items() if v and k in ("missing_keys", "mismatched_keys")}
    if bad:
        raise RuntimeError(f"{repo}: pretrained weights did not load cleanly: {bad}")
    proc = AutoImageProcessor.from_pretrained(repo, revision=revision)
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    info = {"repo": repo, "revision": revision or getattr(model.config, "_commit_hash", None),
            "params": sum(p.numel() for p in model.parameters()), "image_mean": proc.image_mean,
            "image_std": proc.image_std, "pooled_dim": model.config.hidden_size,
            "unused_checkpoint_keys": len(loading.get("unexpected_keys", []))}
    return model, mean, std, info
