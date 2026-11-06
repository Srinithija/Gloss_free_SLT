"""Gloss-free ISL sentence translation with a pretrained ViT (SigLIP or CLIP) vision encoder."""
import os

os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
