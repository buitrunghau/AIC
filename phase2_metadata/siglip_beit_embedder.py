"""Phase 2: Visual embedding generation using SigLIP2 and BEiT-3.

Deliverable: siglip_beit_embedder.py
- Encodes image matrices into dense L2-normalized vectors.
- SigLIP2 uses sigmoid loss (not softmax) for better embedding convergence.
- Output vector dimensions: 768 (BEiT-3) or 1024 (SigLIP2-Large).
"""
