import numpy as np
import faiss

# numpy → required because you’re saving embeddings with np.save.

# faiss → required for building, saving, and loading the FAISS index.

# Load the embeddings
movie_embeddings = np.load("movie_embeddings.npy")
print("✅ Embeddings loaded! Shape:", movie_embeddings.shape)

# Build the index
dimension = movie_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity if vectors are normalized
index.add(movie_embeddings)
print(f"🎯 FAISS index built with {index.ntotal} embeddings.")

# Save the index
faiss.write_index(index, "movie_index.faiss")
np.save("movie_embeddings.npy", movie_embeddings)

# Load the FAISS index
index = faiss.read_index("movie_index.faiss")  # Index file saved earlier