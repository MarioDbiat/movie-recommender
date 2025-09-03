import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Why each one:

# pandas (pd) → reading your CSV dataset.

# numpy (np) → saving embeddings to .npy.

# torch → checking CUDA availability and moving the model to GPU.

# sentence_transformers.SentenceTransformer → loading your fine-tuned SBERT model.

# sklearn.preprocessing.normalize → L2-normalizing embeddings for cosine similarity.

# Load the Dataset
df_movies = pd.read_csv("C:\YOURLOCATION\DATASET")
print(f"Movie Count: {len(df_movies)}")

# Load the Fine-Tuned Model
model = SentenceTransformer("C:\YOURLOCATION\DATASET")

# ✅ Move Model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}.")

# Encode all overviews
print(f"🔄 Encoding {len(df_movies)} movie plots...")
movie_embeddings = model.encode(
    df_movies["overview"].tolist(),
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True
)
print("✅ Embeddings complete!")

# Normalize embeddings for cosine similarity
movie_embeddings = normalize(movie_embeddings, norm='l2', axis=1)

# Save the embeddings
np.save("movie_embeddings.npy", movie_embeddings)
print("✅ Embeddings saved to 'movie_embeddings.npy'")