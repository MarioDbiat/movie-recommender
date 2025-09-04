import os
import difflib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# ---------------- Warnings ----------------
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load`")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ---------------- Config ---------------
# Hugging Face model & dataset
MODEL_ID     = "Mariodb/movie-recommender-model"     # your fine-tuned SBERT
HF_REPO_ID   = "Mariodb/movie-recommender-dataset"   # dataset repo
HF_REPO_TYPE = "dataset"
HF_DF_FILE   = "movies.parquet"                      # metadata
HF_IDX_FILE  = "movie_index.faiss"                   # FAISS (optional)
HF_EMB_FILE  = "movie_embeddings.npy"                # embeddings (fallback)

# Local fallbacks (used only if HF fetch fails)
DATA_CSV   = "data/movies.csv"
FAISS_PATH = "indexes/movie_index.faiss"
EMB_PATH   = "artifacts/movie_embeddings.npy"

POOL_BASE = 50
POOL_WIDE = 300
POOL_MAX  = 1000

# Try FAISS; mark available but don't crash if import fails
try:
    import faiss  # type: ignore
    FAISS_IMPORT_OK = True
except Exception:
    faiss = None  # type: ignore
    FAISS_IMPORT_OK = False

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
        "fast x","tokyo drift","ludacris","hobbs and shaw","hobs and shaw"
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
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type=HF_REPO_TYPE,
        local_dir="data",
        local_dir_use_symlinks=False,
    )

@st.cache_data
def _download_df_from_hf() -> pd.DataFrame:
    local_path = _hf_path(HF_DF_FILE)
    if HF_DF_FILE.endswith(".parquet"):
        return pd.read_parquet(local_path)
    return pd.read_csv(local_path)

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
def load_faiss() -> Tuple[Optional[object], str]:
    """
    Try to load FAISS index. Returns (index_or_None, status_str).
    Never raises—returns (None, reason) on any failure.
    """
    if not FAISS_IMPORT_OK:
        return None, "faiss not importable"

    # Resolve path (prefer local if present)
    path = FAISS_PATH if os.path.exists(FAISS_PATH) else None
    if path is None:
        try:
            path = _hf_path(HF_IDX_FILE)
        except Exception as e:
            return None, f"download failed: {e}"

    # Read index safely
    try:
        idx = faiss.read_index(path)  # type: ignore
        return idx, "ok"
    except MemoryError:
        return None, "memory error while reading index"
    except Exception as e:
        return None, f"read_index failed: {e}"

@st.cache_resource
def load_embeddings() -> Tuple[Optional[np.ndarray], str]:
    """
    Load embeddings via memory-map. Returns (array_or_None, status_str).
    """
    path = EMB_PATH if os.path.exists(EMB_PATH) else None
    if path is None:
        try:
            path = _hf_path(HF_EMB_FILE)
        except Exception as e:
            return None, f"download failed: {e}"

    try:
        arr = np.load(path, mmap_mode="r")  # stream from disk; low RAM
        return arr, "ok"
    except MemoryError:
        return None, "memory error while loading embeddings"
    except Exception as e:
        return None, f"np.load failed: {e}"

# ---------------- Search helpers ----------------
def cosine_topk(query_vec: np.ndarray, emb: np.ndarray, k: int):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = e @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def run_search(query_text: str, franchise_only: bool, safe_mode: bool, use_genres: bool, top_k: int,
               force_no_faiss: bool) -> Tuple[pd.DataFrame, str]:
    meta = load_metadata()
    model = load_model()
    q_vec = model.encode([query_text])[0].astype("float32")

    base_pool = POOL_WIDE if use_genres else POOL_BASE
    pool = int(min(POOL_MAX, max(base_pool, top_k * 10)))

    backend = ""
    idx = None

    if not force_no_faiss:
        faiss_index, faiss_status = load_faiss()
        if faiss_index is not None:
            q = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            D, I = faiss_index.search(q[None, :], pool)
            idx = I[0]
            backend = "FAISS"
        else:
            backend = f"cosine (no FAISS: {faiss_status})"

    if idx is None:
        emb, emb_status = load_embeddings()
        if emb is None:
            st.error("No FAISS index and no embeddings available.\n"
                     f"FAISS status: {faiss_status if not force_no_faiss else 'disabled by user'}\n"
                     f"Embeddings status: {emb_status}")
            return pd.DataFrame(), backend or "unavailable"
        idx, _ = cosine_topk(q_vec, emb, pool)
        if not backend:
            backend = f"cosine (embeddings {emb_status})"

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
        return results, backend

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

    return results[keep].head(top_k), backend

# ---------------- UI ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

with st.sidebar:
    st.markdown("### Status")
    st.caption(f"Model: `{MODEL_ID}`")
    st.caption(f"Dataset: `{HF_REPO_ID}`")
    force_no_faiss = st.toggle("Force disable FAISS", value=False, help="Use cosine+mmap even if FAISS is available")
    st.divider()

st.write(
    "Type a **movie name** or a short **description**. "
    "Optional: *Use genres of query movie* will look up that movie and softly boost results sharing its genres. "
    "*Filter by franchise* restricts results to the detected franchise."
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
    help="Larger values may be slower, especially in cosine fallback mode.",
)

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching…"):
            out, backend = run_search(
                query_text=query.strip(),
                franchise_only=franchise_only,
                safe_mode=safe_mode,
                use_genres=use_genres,
                top_k=top_k,
                force_no_faiss=force_no_faiss,
            )
        st.sidebar.success(f"Backend: {backend}")
        if out.empty:
            st.info("No results. Try fewer filters.")
        else:
            st.subheader(f"Top {top_k}  •  backend: {backend}")
            st.dataframe(out, use_container_width=True)
