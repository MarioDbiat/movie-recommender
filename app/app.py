# app/app.py
# -----------------------------------------
# Movie Recommender (Cloud-optimized)
# -----------------------------------------

# Make sure Cloud doesn't try to use CUDA or spawn too many tokenizer threads
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import difflib
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# ---------------- Warnings ----------------
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ---------------- Config (env-overridable) ---------------
# Model & dataset locations on Hugging Face
MODEL_ID      = os.getenv("MODEL_ID", "Mariodb/movie-recommender-model")

HF_REPO_ID    = os.getenv("HF_REPO_ID", "Mariodb/movie-recommender-dataset")
HF_REPO_TYPE  = os.getenv("HF_REPO_TYPE", "dataset")

# Default to compact artifacts suitable for small containers
HF_DF_FILE    = os.getenv("HF_DF_FILE",  "movies_compact.parquet")     # or movies_lite.parquet
HF_IDX_FILE   = os.getenv("HF_IDX_FILE", "movie_index_ivfpq.faiss")    # or movie_index_lite.faiss
HF_EMB_FILE   = os.getenv("HF_EMB_FILE", "movie_embeddings_lite.npy")  # not used by default

# Local fallbacks (used only if HF fetch fails AND files exist)
DATA_CSV      = os.getenv("DATA_CSV",   "data/movies.csv")
FAISS_PATH    = os.getenv("FAISS_PATH", "indexes/movie_index_ivfpq.faiss")
EMB_PATH      = os.getenv("EMB_PATH",   "artifacts/movie_embeddings.npy")

# Pool sizes (keep conservative for Cloud)
POOL_BASE = int(os.getenv("POOL_BASE", "50"))
POOL_WIDE = int(os.getenv("POOL_WIDE", "300"))
POOL_MAX  = int(os.getenv("POOL_MAX",  "500"))

# Toggle for loading embeddings as a last resort (DISABLED by default on Cloud)
LOAD_EMB = os.getenv("LOAD_EMB", "0") == "1"

# Try FAISS; keep CPU usage low
FAISS_OK = False
try:
    import faiss
    FAISS_OK = True
    try:
        faiss.omp_set_num_threads(1)  # avoid CPU spikes on small machines
    except Exception:
        pass
except Exception:
    pass

# ---------------- Small helpers ----------------
def _split_genres(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s or s.lower() == "unknown":
        return []
    for sep in ["|", ",", ";", "/"]:
        if sep in s:
            return [g.strip() for g in s.split(sep) if g.strip() and g.lower() != "unknown"]
    return [s] if s.lower() != "unknown" else []

def _normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# ---------------- Franchise detection ----------------
_FRANCHISE_KEYWORDS = {
    "marvel": [
        "avengers","iron man","captain america","thor","hulk","black widow","ant-man",
        "black panther","spider-man","dr. strange","doctor strange","shang-chi",
        "guardians of the galaxy","eternals","wanda","marvel","ms. marvel","falcon",
        "winter soldier","multiverse","kang","loki"
    ],
    "dc": [
        "batman","superman","wonder woman","aquaman","flash","justice league",
        "suicide squad","joker","shazam","black adam","dc","zatanna","cyborg",
        "green lantern","penguin","man of steel","superman"
    ],
    "harry potter": [
        "harry potter","hogwarts","voldemort","dumbledore","hermione","ron weasley",
        "fantastic beasts","grindelwald","quidditch","slytherin","gryffindor",
        "hufflepuff","ravenclaw"
    ],
    "lord of the rings": [
        "lord of the rings","frodo","gandalf","aragorn","middle earth","sauron",
        "legolas","hobbit","bilbo","tolkien","elrond","mordor"
    ],
    "star wars": [
        "star wars","skywalker","darth vader","yoda","jedi","sith","death star",
        "grogu","mandalorian","obi-wan","kenobi","dooku","padmé","anakin","rey",
        "bb-8","galactic empire"
    ],
    "fast & furious": [
        "fast and furious","fast & furious","dom toretto","vin diesel","furious","f9",
        "fast x","tokyo drift","ludacris","hobs and shaw","hobbs and shaw"
    ],
    "transformers": [
        "transformers","bumblebee","optimus prime","megatron","autobot","decepticon",
        "rise of the beasts"
    ],
    "twilight": [
        "twilight","edward cullen","bella swan","jacob black","vampire","werewolf",
        "breaking dawn"
    ],
    "the hunger games": [
        "hunger games","katniss everdeen","peeta","panem","district","catching fire",
        "mockingjay","snow"
    ],
    "james bond": [
        "james bond","007","spectre","quantum of solace","skyfall","casino royale",
        "no time to die","moneypenny","q "," mi6","mi6"
    ],
    "pirates of the caribbean": [
        "pirates of the caribbean","jack sparrow","black pearl","davy jones",
        "will turner","elizabeth swann","barbossa"
    ],
    "mission: impossible": [
        "mission impossible","mission: impossible","ethan hunt","imf","ghost protocol",
        "rogue nation","fallout","dead reckoning"
    ],
    "john wick": ["john wick","continental","baba yaga","high table","dog","assassin"],
    "the matrix": ["matrix","neo","trinity","morpheus","agent smith","zion","red pill","blue pill"],
    "despicable me": ["despicable me","minions","gru","agnès","vector","gru jr","agnes"],
    "shrek": ["shrek","donkey","fiona","far far away","puss in boots","lord farquaad"],
    "frozen": ["frozen","elsa","anna","olaf","arendelle","let it go"],
    "cars": ["cars","lightning mcqueen","mater","radiator springs","doc hudson"],
    "jurassic park": [
        "jurassic park","jurassic world","raptor","velociraptor","t-rex","indominus",
        "dr. grant","ian malcolm","claire dearing"
    ],
}

def detect_franchise(title: str, overview: str) -> str:
    text = f"{str(title)} {str(overview)}".lower()
    matches = {}
    for franchise, kws in _FRANCHISE_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                matches[franchise] = matches.get(franchise, 0) + 1
    return max(matches, key=matches.get) if matches else ""

def _ensure_franchise(df: pd.DataFrame) -> pd.DataFrame:
    if "franchise" not in df.columns:
        df["franchise"] = ""
    mask = df["franchise"].fillna("").str.strip()
    mask = (mask == "") | (mask.str.lower() == "unknown")
    if mask.any():
        df.loc[mask, "franchise"] = [
            detect_franchise(t, o) for t, o in zip(
                df.loc[mask, "title"].astype(str),
                df.loc[mask, "overview"].astype(str) if "overview" in df.columns else ["" for _ in range(mask.sum())]
            )
        ]
    return df

def _find_query_movie_genres(df: pd.DataFrame, query_text: str) -> List[str]:
    if "title" not in df.columns or "genres" not in df.columns:
        return []
    q = query_text.strip().lower()
    if not q:
        return []
    exact = df[df["title"].str.lower() == q]
    if not exact.empty:
        return _split_genres(str(exact.iloc[0]["genres"]))
    sample = df
    if "popularity" in df.columns and len(df) > 50000:
        sample = df.nlargest(50000, "popularity")
    titles = sample["title"].astype(str).tolist()
    candidates = difflib.get_close_matches(query_text, titles, n=1, cutoff=0.92)
    if candidates:
        row = sample[sample["title"] == candidates[0]]
        if not row.empty:
            return _split_genres(str(row.iloc[0]["genres"]))
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
    sample = df
    if "popularity" in df.columns and len(df) > 50000:
        sample = df.nlargest(50000, "popularity")
    titles = sample["title"].astype(str).tolist()
    cand = difflib.get_close_matches(query_text, titles, n=1, cutoff=0.92)
    if cand:
        row = sample[sample["title"] == cand[0]]
        if not row.empty:
            return str(row.iloc[0]["franchise"]).strip()
    return ""

# ---------------- HF download helpers ----------------
@st.cache_data
def _hf_path(filename: str) -> str:
    """Download a single file from the HF dataset repo and return local cache path."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type=HF_REPO_TYPE,
        local_dir="data",
        local_dir_use_symlinks=False
    )

@st.cache_data
def _download_df_from_hf() -> pd.DataFrame:
    local_path = _hf_path(HF_DF_FILE)
    if HF_DF_FILE.endswith(".parquet"):
        use_cols = ["title","overview","genres","franchise","popularity"]
        return pd.read_parquet(local_path, columns=use_cols)
    return pd.read_csv(local_path, usecols=["title","overview","genres","franchise","popularity"])

# ---------------- Cached loaders ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_ID)

@st.cache_resource
def load_metadata() -> pd.DataFrame:
    try:
        df = _download_df_from_hf()
    except Exception as e:
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV)
        else:
            st.error(
                "Couldn't fetch dataset from Hugging Face and no local CSV found.\n\n"
                f"Tried: {HF_REPO_ID}/{HF_DF_FILE} and {DATA_CSV}\n\nError: {e}"
            )
            st.stop()

    for col in ["title", "overview", "genres", "franchise"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(lambda s: "" if str(s).strip().lower() == "unknown" else s)

    df = _ensure_franchise(df)
    return df

@st.cache_resource
def load_faiss():
    """Read FAISS index via memory-mapping to reduce RAM usage."""
    if not FAISS_OK:
        st.warning("FAISS is not available in this environment. Search will be disabled.")
        return None
    try:
        path = FAISS_PATH if os.path.exists(FAISS_PATH) else _hf_path(HF_IDX_FILE)
        index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
        # If IVF family, set a reasonable nprobe
        try:
            index.nprobe = int(os.getenv("FAISS_NPROBE", "16"))
        except Exception:
            pass
        return index
    except Exception as e:
        st.error(f"FAISS load failed: {e}")
        return None

@st.cache_resource
def load_embeddings() -> Optional[np.ndarray]:
    """
    Disabled by default on Cloud to avoid OOM.
    Enable by setting LOAD_EMB=1 and ensuring EMB_PATH/HF_EMB_FILE exists.
    """
    if not LOAD_EMB:
        return None
    try:
        path = EMB_PATH if os.path.exists(EMB_PATH) else _hf_path(HF_EMB_FILE)
        if not os.path.exists(path):
            return None
        return np.load(path, mmap_mode="r")
    except Exception as e:
        st.warning(f"Embeddings fallback not available: {e}")
        return None

# ---------------- Search helpers ----------------
def cosine_topk(query_vec: np.ndarray, emb: np.ndarray, k: int):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = e @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def run_search(query_text: str, franchise_only: bool, safe_mode: bool, use_genres: bool, top_k: int) -> pd.DataFrame:
    meta = load_metadata()
    model = load_model()
    q_vec = model.encode([query_text])[0].astype("float32")

    base_pool = POOL_WIDE if use_genres else POOL_BASE
    pool = int(min(POOL_MAX, max(base_pool, top_k * 10)))

    idx = None

    # Try FAISS first
    faiss_index = load_faiss()
    if faiss_index is not None:
        try:
            q = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            D, I = faiss_index.search(q[None, :], pool)
            idx = I[0]
            # Filter out possible -1 entries (IVF edge cases)
            if idx is not None:
                idx = idx[idx >= 0]
        except Exception as e:
            st.warning(f"FAISS search failed (fallback only if embeddings are enabled): {e}")
            idx = None  # fall back below

    # Cosine fallback (disabled by default on Cloud)
    if idx is None or len(idx) == 0:
        emb = load_embeddings()
        if emb is None:
            st.error("Search index unavailable. Please enable FAISS or local embeddings.")
            return pd.DataFrame()
        idx, _ = cosine_topk(q_vec, emb, pool)

    # Guard: if still empty
    if idx is None or len(idx) == 0:
        return pd.DataFrame()

    results = meta.iloc[idx].copy()

    q_lower = query_text.strip().lower()
    if "title" in results.columns and q_lower:
        results = results[results["title"].str.lower() != q_lower]

    if franchise_only:
        q_fr = _find_query_movie_franchise(meta, query_text)
        if q_fr and q_fr.lower() != "unknown" and "franchise" in results.columns:
            results = results[results["franchise"].str.strip().str.lower() == q_fr.lower()]

    if safe_mode:
        bad_words = ["nsfw", "adult", "porn", "xxx", "sex", "erotic", "babe"]
        pattern = "(?i)" + "|".join(bad_words)
        mask = pd.Series(False, index=results.index)
        if "genres" in results.columns:
            mask |= results["genres"].str.contains(pattern, na=False, regex=True)
        if "title" in results.columns:
            mask |= results["title"].str.contains(pattern, na=False, regex=True)
        if "overview" in results.columns:
            mask |= results["overview"].str.contains(pattern, na=False, regex=True)
        results = results.loc[~mask]

    if results.empty:
        return results

    if "popularity" in results.columns:
        base = _normalize(results["popularity"].to_numpy(dtype="float32"))
    else:
        base = np.zeros(len(results), dtype="float32")

    genre_bonus = np.zeros(len(results), dtype="float32")
    if use_genres:
        query_genres = _find_query_movie_genres(meta, query_text)
        if query_genres and "genres" in results.columns:
            qset = set(query_genres)
            overlaps = []
            for g in results["genres"].tolist():
                rset = set(_split_genres(g))
                overlaps.append(len(qset & rset))
            overlaps = np.array(overlaps, dtype="float32")
            genre_bonus = np.minimum(overlaps, 3.0) * 0.1

    score = base + genre_bonus
    results = results.assign(score=score).sort_values("score", ascending=False)

    if franchise_only:
        keep = [c for c in ["title", "genres", "franchise", "popularity"] if c in results.columns]
    else:
        keep = [c for c in ["title", "genres", "popularity"] if c in results.columns]

    return results[keep].head(top_k)

# ---------------- UI ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender (Cloud-optimized)")

st.write(
    "Type a **movie name** or a short **description**. "
    "If you enable *Use genres of query movie*, the app will look up the movie's genres "
    "and softly boost results that share them. If the movie has no genres, search runs normally. "
    "Use the *franchise filter* to only show items from the same franchise as the query movie."
)

query = st.text_input("Search", placeholder="e.g., Interstellar  •  gritty space survival")

col_left, col_right = st.columns([2, 1])
with col_left:
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        franchise_only = st.checkbox("Filter by franchise", value=False)
    with subcol2:
        use_genres = st.checkbox("Use genres of query movie", value=False)
with col_right:
    safe_mode = st.checkbox("Safe mode (hide NSFW)", value=True)

top_k = st.slider(
    "How many recommendations?",
    min_value=5, max_value=50, value=5, step=5,
    help="Choose how many results to show. Larger values may be slower.",
)

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching…"):
            out = run_search(
                query_text=query.strip(),
                franchise_only=franchise_only,
                safe_mode=safe_mode,
                use_genres=use_genres,
                top_k=top_k,
            )
        if out.empty:
            st.info("No results. Try fewer filters or ensure the FAISS index is available.")
        else:
            st.subheader(f"Top {top_k}")
            st.dataframe(out, use_container_width=True)
