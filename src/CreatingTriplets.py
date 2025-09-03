import random
from tqdm import tqdm

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import faiss

# Explanation of each:

# pandas (pd) → load and clean your CSV dataset.

# numpy (np) → numerical operations (e.g., arrays, embeddings).

# tqdm → progress bar for triplet generation loop.

# random → sampling negatives randomly.

# sentence_transformers.SentenceTransformer → load SBERT model and encode plots.

# sklearn.preprocessing.normalize → normalize embeddings for cosine similarity.

# sklearn.metrics.pairwise.cosine_similarity → compute similarity between embeddings.

# faiss → efficient similarity search and indexing.


# 1️⃣ Load and Clean Dataset
df = pd.read_csv(r"C:\MovieReommenderSystem\TMDB_movie_dataset_v12.csv")
df = df[df["overview"].notna()]
df = df[df["overview"].str.len() > 100]
df = df.drop_duplicates(subset="overview")

# 2️⃣ NSFW Filtering
nsfw_keywords = [
    'xxx', 'porn', 'escort', 'call girl', 'hardcore', 'nude', 'sex', 'slut', 'babe',
    'cock', 'milf', 'pounded', 'oral', 'fetish', 'suck', 'jerk', 'cum', 'nsfw',
    'panties', 'blowjob', 'fucking', 'fuck', 'felatio', 'nasty', 'nipple', 'vibrator',
    'thagson', 'dildo', 'erotic', 'lust', 'orgasm', 'squirt', 'rape', 'raped', 'raping'
]
def is_nsfw(text):
    return any(word in text.lower() for word in nsfw_keywords)
df = df[~df["overview"].apply(is_nsfw)].reset_index(drop=True)
print(f"✅ Dataset cleaned: {len(df)} entries remain after NSFW filtering.")

# 3️⃣ Sample ~100k plots (helps us generate more than 35k triplets)
df_sample = df.sample(n=min(100000, len(df)), random_state=42).reset_index(drop=True)
plots = df_sample["overview"].tolist()

# 4️⃣ SBERT Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
print("🔄 Encoding plots with SBERT...")
embeddings = model.encode(plots, batch_size=64, convert_to_numpy=True, show_progress_bar=True)

# 5️⃣ Normalize for cosine similarity
normalized_embeddings = normalize(embeddings, axis=1)

# 6️⃣ FAISS Index for fast similarity search
dimension = normalized_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(normalized_embeddings)

# 7️⃣ Generate Triplets
triplets = []
skipped = 0

print("🚀 Generating triplets...")
for i in tqdm(range(len(df_sample)), desc="🔍 Filtering Triplets"):
    anchor_idx = i
    anchor_vec = normalized_embeddings[anchor_idx].reshape(1, -1)

    # Top 20 similar indices (exclude self)
    _, I = index.search(anchor_vec, 21)
    similar_indices = I[0][1:]

    # Find valid positive
    selected_positive = None
    for pos_idx in similar_indices:
        sim_score = cosine_similarity(anchor_vec, normalized_embeddings[pos_idx].reshape(1, -1))[0][0]
        if sim_score > 0.60:
            selected_positive = pos_idx
            break
    if selected_positive is None:
        skipped += 1
        continue

    # Find valid negative
    selected_negative = None
    for _ in range(10):
        neg_idx = random.randint(0, len(df_sample) - 1)
        if neg_idx in [anchor_idx, selected_positive]:
            continue
        sim_score = cosine_similarity(anchor_vec, normalized_embeddings[neg_idx].reshape(1, -1))[0][0]
        if sim_score < 0.45:
            selected_negative = neg_idx
            break
    if selected_negative is None:
        skipped += 1
        continue

    # Append triplet
    triplets.append([
        plots[anchor_idx],
        plots[selected_positive],
        plots[selected_negative]
    ])

    # 💡 Stop at 50,000 triplets max
    if len(triplets) >= 50000:
        break

# 8️⃣ Save Triplets
triplet_df = pd.DataFrame(triplets, columns=["anchor", "positive", "negative"])
triplet_df.to_csv("expanded_triplet_dataset_v2.csv", index=False)

print(f"\n✅ Generated {len(triplets)} high-quality triplets")
print(f"❌ Skipped {skipped} anchors with no valid match")
print("💾 Saved as 'expanded_triplet_dataset.csv_v2'")
