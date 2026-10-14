"""
src/embeddings/fasttext_loader.py

FastText embedding matrix builder for Arabic text.

Will contain:
- Load pre-trained Arabic fastText vectors (.vec or .bin)
- Build embedding matrix aligned with vocabulary
- Handle OOV words with random initialization
- Save/load embedding matrix for reuse
"""