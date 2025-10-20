# app/ApplicationFix.py
# =========================================
# Movie Recommender — Streamlit app (popularity toggle + safe fixes)
# =========================================

# --- Streamlit must be configured first ---
import streamlit as st
st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- Std/3p imports ---
import os
import re
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# Silence only the specific FutureWarning from transformers -> torch.load
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
    module="transformers.modeling_utils",
)

# Avoid noisy tokenizer threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Optional FAISS (app still runs without it)
try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    faiss = None
    _HAVE_FAISS = False

from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
# Adjust these defaults if your files are elsewhere.
MODEL_NAME   = os.getenv("MODEL_NAME", "SBERT_Movie_Recommender_v1")
PARQUET_FILE = os.getenv("PARQUET_FILE", "data/movies.parquet")
INDEX_FILE   = os.getenv("INDEX_FILE",   "movie_index.faiss")      # root
EMB_FILE     = os.getenv("EMB_FILE",     "movie_embeddings.npy")   # root

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# Use the same candidate pool style as your notebook for parity
def choose_pool(top_k: int, use_genres: bool) -> int:
    # notebook-like behavior: fanout = top_k * 20
    return max(top_k * 20, top_k)

# ---------------- Caching layers ----------------
@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    # Loads your local fine-tuned SBERT (folder) or HF id
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(show_spinner=False)
def load_metadata(parquet_path: str) -> pd.DataFrame:
    # Reset index -> ensures row ↔ vector positions align
    df = pd.read_parquet(parquet_path).reset_index(drop=True)

    # Normalize columns we rely on
    if "title" in df.columns:
        df["title"] = df["title"].astype(str)
    if "overview" in df.columns:
        df["overview"] = df["overview"].fillna("").astype(str)

    # Genres -> list[str]
    if "genres" in df.columns:
        def _fix_genres(x):
            if isinstance(x, list):
                return [g for g in x if str(g).strip().lower() != "unknown"]
            s = str(x).strip()
            if not s or s.lower() == "unknown":
                return []
            for sep in ("|", ",", ";", "/"):
                if sep in s:
                    return [g.strip() for g in s.split(sep)
                            if g.strip() and g.strip().lower() != "unknown"]
            return [s] if s.lower() != "unknown" else []
        df["genres"] = df["genres"].apply(_fix_genres)

    if "franchise" in df.columns:
        df["franchise"] = df["franchise"].fillna("Unknown").astype(str)

    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)

    return df

@st.cache_resource(show_spinner=False)
def load_faiss(index_path: str):
    if not _HAVE_FAISS or not os.path.exists(index_path):
        return None
    try:
        return faiss.read_index(index_path)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_embeddings(npy_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(npy_path):
        return None
    try:
        arr = np.load(npy_path, mmap_mode="r")
        return arr.astype("float32") if arr.dtype != np.float32 else arr
    except Exception:
        return None

# ---------------- Utilities ----------------
def _split_genres(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s or s.lower() == "unknown":
        return []
    for sep in ["|", ",", ";", "/"]:
        if sep in s:
            return [g.strip() for g in s.split(sep)
                    if g.strip() and g.lower() != "unknown"]
    return [s] if s.lower() != "unknown" else []

def _find_query_movie_genres(df: pd.DataFrame, query_text: str) -> List[str]:
    if "title" in df.columns and "genres" in df.columns:
        q = query_text.strip().lower()
        hit = df[df["title"].str.lower() == q]
        if not hit.empty:
            g = hit.iloc[0]["genres"]
            return g if isinstance(g, list) else _split_genres(str(g))
    return []

def _find_query_movie_franchise(df: pd.DataFrame, query_text: str) -> str:
    if "title" not in df.columns or "franchise" not in df.columns:
        return ""
    q = query_text.strip().lower()
    if not q:
        return ""
    exact = df[df["title"].str.lower() == q]
    if not exact.empty:
        return str(exact.iloc[0]["franchise"]).strip()
    # fallback: look in popular titles containing the query
    sample = df.nlargest(50000, "popularity") if "popularity" in df.columns and len(df) > 50000 else df
    titles = sample["title"].astype(str).tolist()
    idxs = [i for i, t in enumerate(titles) if q in t.lower()]
    if idxs and "franchise" in sample.columns:
        return str(sample.iloc[idxs[0]]["franchise"]).strip()
    return ""

FRANCHISE_KEYWORDS = {
    "Marvel": [
        "avengers", "iron man", "captain america", "thor", "hulk", "black widow",
        "ant-man", "black panther", "spider-man", "doctor strange", "dr. strange",
        "shang-chi", "guardians of the galaxy", "eternals", "wanda", "ms. marvel",
        "falcon", "winter soldier", "multiverse", "kang", "loki"
    ],
    "DC": [
        "batman", "superman", "wonder woman", "aquaman", "flash", "justice league",
        "suicide squad", "joker", "shazam", "black adam", "dc", "zatanna",
        "cyborg", "green lantern", "penguin"
    ],
    "Harry Potter": [
        "harry potter", "hogwarts", "voldemort", "dumbledore", "hermione",
        "ron weasley", "fantastic beasts", "grindelwald", "quidditch", "slytherin",
        "gryffindor", "hufflepuff", "ravenclaw"
    ],
    "Lord of the Rings": [
        "lord of the rings", "frodo", "gandalf", "aragorn", "middle earth", "sauron",
        "legolas", "hobbit", "bilbo", "tolkien", "elrond", "mordor"
    ],
    "Star Wars": [
        "star wars", "skywalker", "darth vader", "yoda", "jedi", "sith",
        "death star", "grogu", "mandalorian", "obi-wan", "kenobi", "dooku",
        "anakin", "rey", "bb-8", "galactic empire"
    ],
    "Fast & Furious": [
        "fast and furious", "fast & furious", "dom toretto", "vin diesel", "furious",
        "fast x", "tokyo drift", "hobbs", "shaw"
    ],
    "Transformers": [
        "transformers", "bumblebee", "optimus prime", "megatron", "autobot",
        "decepticon", "rise of the beasts"
    ],
    "Twilight": [
        "twilight", "edward cullen", "bella swan", "jacob black", "vampire",
        "werewolf", "breaking dawn"
    ],
    "The Hunger Games": [
        "hunger games", "katniss", "peeta", "panem", "district", "catching fire",
        "mockingjay", "president snow"
    ],
    "James Bond": [
        "james bond", "007", "spectre", "quantum of solace", "skyfall",
        "casino royale", "no time to die", "moneypenny", "mi6"
    ],
    "Pirates of the Caribbean": [
        "pirates of the caribbean", "jack sparrow", "black pearl", "davy jones",
        "will turner", "elizabeth swann", "barbossa"
    ],
    "Mission: Impossible": [
        "mission impossible", "ethan hunt", "imf", "ghost protocol",
        "rogue nation", "fallout", "dead reckoning"
    ],
    "John Wick": [
        "john wick", "continental", "baba yaga", "high table", "assassin"
    ],
    "The Matrix": [
        "matrix", "neo", "trinity", "morpheus", "agent smith", "zion",
        "red pill", "blue pill"
    ],
    "Despicable Me": [
        "despicable me", "minions", "gru", "agnes", "vector"
    ],
    "Shrek": [
        "shrek", "donkey", "fiona", "far far away", "puss in boots", "farquaad"
    ],
    "Frozen": [
        "frozen", "elsa", "anna", "olaf", "arendelle", "let it go"
    ],
    "Cars": [
        "cars", "lightning mcqueen", "mater", "radiator springs", "doc hudson"
    ],
    "Jurassic Park": [
        "jurassic park", "jurassic world", "raptor", "velociraptor", "t-rex",
        "indominus", "ian malcolm", "claire dearing"
    ]
}


def detect_franchise(title: str, overview: str) -> str:
    text = f"{title} {overview}".lower()
    matches = {}
    for franchise, keywords in FRANCHISE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits:
            matches[franchise] = hits
    if not matches:
        return "Unknown"
    return max(matches, key=matches.get)



def cosine_topk(query_vec: np.ndarray, emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    En = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = En @ qn
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

# ---------------- Core search ----------------

def run_search(
    parquet_name: str,
    index_name: str,
    emb_name: Optional[str],
    query_text: str,
    franchise_only: bool,
    safe_mode: bool,
    use_genres: bool,
    use_popularity: bool,   # checkbox
    top_k: int,
) -> pd.DataFrame:
    df = load_metadata(parquet_name)
    model = load_model()
    index_obj = load_faiss(index_name)
    embeddings = load_embeddings(emb_name or "")

    if embeddings is None:
        st.error("Embeddings file is required for scoring.")
        return pd.DataFrame()

    if len(df) != len(embeddings):
        st.error("Embeddings and metadata shapes do not align.")
        return pd.DataFrame()

    query_clean = query_text.strip()
    if not query_clean:
        return pd.DataFrame()

    query_lower = query_clean.lower()
    movie_row = df[df["title"].str.lower() == query_lower] if "title" in df.columns else pd.DataFrame()

    if not movie_row.empty:
        row = movie_row.iloc[0]
        overview = str(row.get("overview", "") or "")
        query_vec = model.encode(overview, convert_to_tensor=True)
        detected_franchise = detect_franchise(str(row.get("title", "")), overview)
        genres_val = row.get("genres", [])
        if isinstance(genres_val, list):
            input_genres = {g for g in genres_val if g != "Unknown"}
        else:
            input_genres = set(_split_genres(str(genres_val)))
        use_name_anchor = True
    else:
        overview = query_clean
        query_vec = model.encode(overview, convert_to_tensor=True)
        detected_franchise = detect_franchise("", overview)
        input_genres = set()
        use_name_anchor = False

    query_np = query_vec.detach().cpu().numpy().reshape(1, -1).astype("float32")
    fanout = max(top_k * 20, top_k)

    candidate_idx = None
    if index_obj is not None:
        try:
            _, idxs = index_obj.search(query_np, fanout)
            candidate_idx = idxs[0]
        except Exception as exc:
            st.warning(f"FAISS search failed ({exc}); falling back to cosine search.")
            candidate_idx = None

    if candidate_idx is None or len(candidate_idx) == 0:
        query_vec_np = query_np[0]
        emb_norms = np.linalg.norm(embeddings, axis=1) + 1e-12
        sims_all = (embeddings @ query_vec_np) / (emb_norms * (np.linalg.norm(query_vec_np) + 1e-12))
        topn = min(fanout, sims_all.shape[0])
        candidate_idx = np.argsort(-sims_all)[:topn]
    else:
        candidate_idx = np.asarray(candidate_idx, dtype=int)

    candidate_idx = candidate_idx[(candidate_idx >= 0) & (candidate_idx < len(df))]
    if candidate_idx.size == 0:
        return pd.DataFrame()

    results = df.iloc[candidate_idx].copy()
    cand_embeddings = embeddings[candidate_idx]

    query_norm = np.linalg.norm(query_np[0]) + 1e-12
    cand_norms = np.linalg.norm(cand_embeddings, axis=1) + 1e-12
    sims = (cand_embeddings @ query_np[0]) / (cand_norms * query_norm)

    if use_popularity and "popularity" in results.columns:
        pop_vals = pd.to_numeric(results["popularity"], errors="coerce").fillna(0).to_numpy("float32")
        scores = (sims * np.log1p(pop_vals)).astype("float32")
    else:
        scores = sims.astype("float32")

    results = results.assign(score=scores, similarity=sims.astype("float32"))

    if safe_mode and not results.empty:
        if "nsfw" in results.columns:
            results = results[~results["nsfw"].astype(bool)]
        else:
            bad_terms = ["nsfw", "adult", "porn", "xxx", "sex", "erotic", "babe"]
            pattern = "(?i)" + "|".join(bad_terms)
            mask = pd.Series(False, index=results.index)
            for col in ["genres", "title", "overview"]:
                if col in results.columns:
                    mask |= results[col].astype(str).str.contains(pattern, na=False, regex=True)
            results = results.loc[~mask]

    if results.empty:
        return results

    if franchise_only and detected_franchise != "Unknown" and not results.empty:
        if "franchise" not in results.columns:
            results["franchise"] = "Unknown"
        missing_mask = results["franchise"].fillna("Unknown").isin(["", "Unknown"])
        if missing_mask.any():
            results.loc[missing_mask, "franchise"] = results.loc[missing_mask].apply(
                lambda row: detect_franchise(str(row.get("title", "")), str(row.get("overview", ""))),
                axis=1,
            )
        results = results[results["franchise"] == detected_franchise]

    if results.empty:
        return results

    if use_genres and input_genres and not results.empty:
        def genre_overlap(val):
            if isinstance(val, list):
                working = {g for g in val if g != "Unknown"}
            else:
                working = set(_split_genres(str(val)))
            return len(input_genres & working)

        results = results.assign(genre_overlap=results["genres"].apply(genre_overlap))
        min_required = 2 if len(input_genres) >= 2 else 1
        filtered = results[results["genre_overlap"] >= min_required]

        if len(filtered) < top_k and results["genre_overlap"].max() >= 1:
            fallback_threshold = 1 if min_required > 1 else 0
            filtered = results[results["genre_overlap"] >= fallback_threshold]

        results = filtered
        if not results.empty:
            results = results.assign(score=results["score"] * (1.0 + 0.05 * results["genre_overlap"]))

    if results.empty:
        return results

    if use_name_anchor and "title" in results.columns:
        results = results[results["title"].str.lower() != query_lower]

    if results.empty:
        return results

    results = results.sort_values(by="score", ascending=False)
    top_results = results.head(top_k).copy()

    if "franchise" not in top_results.columns:
        top_results["franchise"] = "Unknown"
    else:
        top_results["franchise"] = top_results["franchise"].fillna("Unknown").replace("", "Unknown")

    if "poster_path" in top_results.columns:
        top_results["poster_url"] = top_results["poster_path"].astype(str).str.strip()
        mask = top_results["poster_url"].str.len() > 0
        top_results.loc[mask, "poster_url"] = TMDB_IMG_BASE + top_results.loc[mask, "poster_url"]
    elif "poster_url" not in top_results.columns:
        top_results["poster_url"] = ""

    keep = ["title", "genres", "franchise", "popularity", "poster_url", "overview", "score", "similarity"]
    if "id" in top_results.columns:
        keep.append("id")
    existing = [c for c in keep if c in top_results.columns]
    return top_results[existing].reset_index(drop=True)


# ---------------- UI ----------------
st.title("🎬 Mario's Netflix")

with st.expander("Technical info"):
    st.write(f"Model: `{MODEL_NAME}`")
    st.write(f"Parquet: `{PARQUET_FILE}`  (exists: {os.path.exists(PARQUET_FILE)})")
    st.write(f"FAISS index: `{INDEX_FILE}`  (exists: {os.path.exists(INDEX_FILE)})  FAISS available: `{_HAVE_FAISS}`")
    st.write(f"Embeddings: `{EMB_FILE}`  (exists: {os.path.exists(EMB_FILE)})")

st.subheader("🔍 Search")
query = st.text_input("Title or description", placeholder="e.g., Interstellar — gritty space survival")

st.divider()
st.subheader("⚙️ Filters")

col1, col2, col3, col4 = st.columns(4)
with col1:
    franchise_only = st.checkbox(
        "Filter by franchise",
        value=False,
        help="Only show results from the same franchise as the query movie."
    )
with col2:
    use_genres = st.checkbox(
        "Use genres of query movie",
        value=False,
        help="HARD filter: keep only results sharing at least 2 genres with the query (exact title works best)."
    )
with col3:
    safe_mode = st.checkbox(
        "Safe mode (hide NSFW)",
        value=True,
        help="Hide adult/NSFW content from results."
    )
with col4:
    use_popularity = st.checkbox(
        "Use popularity boost",
        value=True,
        help="When on, blend similarity with log(popularity). When off, rank by similarity only."
    )

top_k = st.slider(
    "How many recommendations?",
    min_value=5, max_value=50, value=10, step=5
)

st.divider()
st.subheader("🎯 Recommendations")

if st.button("Recommend", type="primary"):
    if not query.strip():
        st.warning("Please enter a title or short description.")
    else:
        with st.spinner("🔎 Finding great matches…"):
            out = run_search(
                parquet_name=PARQUET_FILE,
                index_name=INDEX_FILE,
                emb_name=EMB_FILE,
                query_text=query.strip(),
                franchise_only=franchise_only,
                safe_mode=safe_mode,
                use_genres=use_genres,
                use_popularity=use_popularity,  # wired to checkbox
                top_k=top_k,
            )

        if out is None or len(out) == 0:
            st.info("No results. Try a different title/description or relax filters.")
        else:
            for _, row in out.iterrows():
                poster_url = row.get("poster_url") or "https://via.placeholder.com/120x180?text=No+Image"
                c1, c2 = st.columns([1, 5])
                with c1:
                    st.image(poster_url, width=110)
                with c2:
                    st.markdown(f"### {row.get('title','Untitled')}")
                    if row.get("genres"):
                        st.caption(f"Genres: {', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']}")
                    if row.get("franchise"):
                        st.caption(f"🎬 Franchise: {row['franchise']}")
                    if row.get("popularity") is not None:
                        try:
                            st.caption(f"⭐ Popularity: {float(row['popularity']):.2f}")
                        except Exception:
                            pass
                    if row.get("overview"):
                        st.write(row["overview"])
                st.divider()

            with st.expander("See raw table"):
                st.dataframe(out)
