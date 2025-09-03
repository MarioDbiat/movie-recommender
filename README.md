# Movie Recommender (Transformer + TripletLoss + FAISS)

**One-line:** Semantic movie recommendations in ~0.01s using Sentence-Transformers + FAISS.  
**Demo:** (link here) • **Paper (66 pp):** (link) • **Short summary (1–2 pp):** (link)

## Features
- Transformer embeddings (SBERT) + Triplet Loss fine-tuning
- FAISS ANN search (IVF/HNSW/PQ)—fast at scale
- Franchise detection • NSFW safety filter
- (Optional) Hybrid retrieval (BM25 + dense)

## Quickstart
```bash
# Python 3.8
pip install -r requirements.txt
# Minimal demo run (streamlit or CLI)
# streamlit run app/app.py
