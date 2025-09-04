# app/app.py
# =========================================
# Movie Recommender — dual-mode (Cloud + Local), Py3.8-safe
# =========================================

import os

# ---- Streamlit must be first command ----
import streamlit as st
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Environment safety & tokenizer threads
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import difflib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ---------------- Mode / config (env-overridable) ----------------
# MODE = auto | lite | full
MODE = os.getenv("MODE", "auto").lower()
LOAD_EMB = os.getenv("LOAD_EMB", "0") == "1"  # rarely needed online

# HF dataset repo (same for both modes)
HF_REPO_ID   = os.getenv("HF_REPO_ID", "Mariodb/movie-recommender-dataset")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")

# Filenames in the HF repo
LITE_PARQUET = os.getenv("LITE_PARQUET", "movies_compact.parquet")
LITE_INDEX   = os.getenv("LITE_INDEX",   "movie_index_ivfpq.faiss")

FULL_PARQUET = os.getenv("FULL_PARQUET", "movies.parquet")
FULL_INDEX   = os.getenv("FULL_INDEX",   "movie_index.faiss")
FULL_EMB     = os.getenv("FULL_EMB",     "movie_embeddings.npy")  # optional

# Local fallbacks (if HF fetch fails AND files exist locally)
DATA_CSV   = os.getenv("DATA_CSV",   "data/movies.csv")
FAISS_PATH = os.getenv("FAISS_PATH", "indexes/movie_index_ivfpq.faiss")
EMB_PATH   = os.getenv("EMB_PATH",   "artifacts/movie_embeddings.npy")

# Model
MODEL_ID = os.getenv("MODEL_ID", "Mariodb/movie-recommender-model")

# Pools (Cloud-friendly defaults)
POOL_BASE = int(os.getenv("POOL_BASE", "50"))
POOL_WIDE = int(os.getenv("POOL_WIDE", "300"))
POOL_MAX  = int(os.getenv("POOL_MAX",  "500"))

# FAISS availability (import after set_page_config)
FAISS_OK = False
try:
    import faiss
    FAISS_OK = True
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass
except Exception:
    pass

# ---------------- Utilities ----------------
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
    "twilight": ["twilight","edward cullen","bella swan","jacob black","vampire","werewolf","breaking dawn"],
    "the hunger games": ["hunger games","katniss everdeen","peeta","panem","district","catching fire","mockingjay","snow"],
    "james bond": ["james bond","007","spectre","quantum of solace","skyfall","casino royale","no time to die","moneypenny","q "," mi6","mi6"],
    "pirates of the caribbean": ["pirates of the caribbean","jack sparrow","black pearl","davy jones","will turner","elizabeth swann","barbossa"],
    "mission: impossible": ["mission impossible","mission: impossible","ethan hunt","imf","ghost protocol","rogue nation","fallout","dead reckoning"],
    "john wick": ["john wick","continental","baba yaga","high table","dog","assassin"],
    "the matrix": ["matrix","neo","trinity","morpheus","agent smith","zion","red pill","blue pill"],
    "despicable me": ["despicable me","minions","gru","agnès","vector","gru jr","agnes"],
    "shrek": ["shrek","donkey","fiona","far far away","puss in boots","lord farquaad"],
    "frozen": ["frozen","elsa","anna","olaf","arendelle","let it go"],
    "cars": ["cars","lightning mcqueen","mater","radiator springs","doc hudson"],
    "jurassic park": ["jurassic park","jurassic world","raptor","velociraptor","t-rex","indominus","dr. grant","ian malcolm","claire dearing"],
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
    sample = df.nlargest(50000, "popularity") if "popularity" in df.columns and len(df) > 50000 else df
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
    sample = df.nlargest(50000, "popularity") if "popularity" in df.columns and len(df) > 50000 else df
    titles = sample["title"].astype(str).tolist()
    cand = difflib.get_close_matches(query_text, titles, n=1, cutoff=0.92)
    if cand:
        row = sample[sample["title"] == cand[0]]
        if not row.empty:
            return str(row.iloc[0]["franchise"]).strip()
    return ""

# ---------------- HF helpers ----------------
@st.cache_data
def _hf_path(filename: str) -> str:
    """Download a single file from the HF dataset repo and return local cache path."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type=HF_REPO_TYPE,
        local_dir="data",
        local_dir_use_symlinks=False,
    )

def _choose_files() -> Tuple[str, str, Optional[str]]:
    """
    Decide which artifacts to use (lite vs full).
    We do this at runtime (after set_page_config) to satisfy Streamlit's ordering rule.
    """
    def exists_on_hf(name: str) -> bool:
        try:
            _ = _hf_path(name)   # will download or hit cache
            return True
        except Exception:
            return False

    if MODE == "lite":
        return (LITE_PARQUET, LITE_INDEX, None)
    if MODE == "full":
        return (FULL_PARQUET, FULL_INDEX, FULL_EMB if LOAD_EMB else None)

    # MODE == auto: prefer lite if both files exist, else fall back to full
    if exists_on_hf(LITE_PARQUET) and exists_on_hf(LITE_INDEX):
        return (LITE_PARQUET, LITE_INDEX, None)
    return (FULL_PARQUET, FULL_INDEX, FULL_EMB if LOAD_EMB else None)

@st.cache_data
def _download_df_from_hf(parquet_name: str) -> pd.DataFrame:
    path = _hf_path(parquet_name)
    if parquet_name.endswith(".parquet"):
        use_cols = ["title","overview","genres","franchise","popularity"]
        return pd.read_parquet(path, columns=[c for c in use_cols if c])
    return pd.read_csv(path, usecols=["title","overview","genres","franchise","popularity"])

# ---------------- Cached loaders ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_ID)

@st.cache_resource
def load_metadata(parquet_name: str) -> pd.DataFrame:
    try:
        df = _download_df_from_hf(parquet_name)
    except Exception as e:
        # local fallback
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV)
        else:
            st.error(
                "Couldn't fetch dataset from Hugging Face and no local CSV found.\n\n"
                "Tried: {}/{} and {}\n\nError: {}".format(HF_REPO_ID, parquet_name, DATA_CSV, e)
            )
            st.stop()

    for col in ["title","overview","genres","franchise"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(lambda s: "" if str(s).strip().lower() == "unknown" else s)
    return _ensure_franchise(df)

@st.cache_resource
def load_faiss(index_name: str):
    if not FAISS_OK:
        st.warning("FAISS is not available; will use embeddings only if enabled.")
        return None
    try:
        # HF first, then local fallback
        path = FAISS_PATH if os.path.exists(FAISS_PATH) else _hf_path(index_name)
        index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
        try:
            index.nprobe = int(os.getenv("FAISS_NPROBE", "16"))
        except Exception:
            pass
        return index
    except Exception as e:
        st.error("FAISS load failed: {}".format(e))
        return None

@st.cache_resource
def load_embeddings(emb_name: Optional[str]) -> Optional[np.ndarray]:
    if not emb_name:
        return None
    try:
        path = EMB_PATH if os.path.exists(EMB_PATH) else _hf_path(emb_name)
        if not os.path.exists(path):
            return None
        return np.load(path, mmap_mode="r")
    except Exception as e:
        st.warning("Embeddings fallback not available: {}".format(e))
        return None

# ---------------- Search helpers ----------------
def cosine_topk(query_vec: np.ndarray, emb: np.ndarray, k: int):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = e @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def run_search(parquet_name: str, index_name: str, emb_name: Optional[str],
               query_text: str, franchise_only: bool, safe_mode: bool, use_genres: bool, top_k: int) -> pd.DataFrame:
    meta = load_metadata(parquet_name)
    model = load_model()
    q_vec = model.encode([query_text])[0].astype("float32")

    base_pool = POOL_WIDE if use_genres else POOL_BASE
    pool = int(min(POOL_MAX, max(base_pool, top_k * 10)))

    idx = None

    faiss_index = load_faiss(index_name)
    if faiss_index is not None:
        try:
            q = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            D, I = faiss_index.search(q[None, :], pool)
            idx = I[0]
            if idx is not None:
                idx = idx[idx >= 0]
        except Exception as e:
            st.warning("FAISS search failed; will try embeddings if enabled. Error: {}".format(e))
            idx = None

    if idx is None or len(idx) == 0:
        emb = load_embeddings(emb_name)
        if emb is None:
            st.error("Search index unavailable (no FAISS / embeddings). "
                     "Upload lite/full index to HF or enable embeddings locally.")
            return pd.DataFrame()
        idx, _ = cosine_topk(q_vec, emb, pool)

    if idx is None or len(idx) == 0:
        return pd.DataFrame()

    results = meta.iloc[idx].copy()

    q_lower = query_text.strip().lower()
    if "title" in results.columns and q_lower:
        results = results[results["title"].str.lower() != q_lower]

    if franchise_only and "franchise" in results.columns:
        q_fr = _find_query_movie_franchise(meta, query_text)
        if q_fr and q_fr.lower() != "unknown":
            results = results[results["franchise"].str.strip().str.lower() == q_fr.lower()]

    if safe_mode:
        bad = ["nsfw","adult","porn","xxx","sex","erotic","babe"]
        pattern = "(?i)" + "|".join(bad)
        mask = pd.Series(False, index=results.index)
        for col in ["genres","title","overview"]:
            if col in results.columns:
                mask |= results[col].str.contains(pattern, na=False, regex=True)
        results = results.loc[~mask]

    if results.empty:
        return results

    base = _normalize(results["popularity"].to_numpy(dtype="float32")) if "popularity" in results.columns else np.zeros(len(results), dtype="float32")

    genre_bonus = np.zeros(len(results), dtype="float32")
    if use_genres and "genres" in results.columns:
        qg = _find_query_movie_genres(meta, query_text)
        if qg:
            qset = set(qg)
            overlaps = []
            for g in results["genres"].tolist():
                rset = set(_split_genres(g))
                overlaps.append(len(qset & rset))
            overlaps = np.asarray(overlaps, dtype="float32")
            genre_bonus = np.minimum(overlaps, 3.0) * 0.1

    score = base + genre_bonus
    results = results.assign(score=score).sort_values("score", ascending=False)
    keep = [c for c in (["title","genres","franchise","popularity"] if franchise_only else ["title","genres","popularity"]) if c in results.columns]
    return results[keep].head(top_k)

# ---------------- UI ----------------
mode_badge = {"lite": "(Lite index)", "full": "(Full index)", "auto": "(Auto mode)"}.get(MODE, "")
st.title("🎬 Movie Recommender {}".format(mode_badge))
st.caption(
    "Search by movie name or short description. "
    "In *auto* mode the app uses lite artifacts if available (best for Cloud), "
    "otherwise uses full artifacts (best for local)."
)

# Resolve which files to use (AFTER set_page_config)
PARQUET_FILE, INDEX_FILE, EMB_FILE = _choose_files()

query = st.text_input("Search", placeholder="e.g., Interstellar — gritty space survival")
col_left, col_right = st.columns([2, 1])
with col_left:
    a, b = st.columns(2)
    with a:
        franchise_only = st.checkbox("Filter by franchise", value=False)
    with b:
        use_genres = st.checkbox("Use genres of query movie", value=False)
with col_right:
    safe_mode = st.checkbox("Safe mode (hide NSFW)", value=True)

top_k = st.slider("How many recommendations?", 5, 50, 5, 5)

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching…"):
            out = run_search(
                parquet_name=PARQUET_FILE,
                index_name=INDEX_FILE,
                emb_name=EMB_FILE,
                query_text=query.strip(),
                franchise_only=franchise_only,
                safe_mode=safe_mode,
                use_genres=use_genres,
                top_k=top_k,
            )
        if out.empty:
            st.info("No results. Try fewer filters or ensure your FAISS index files exist in the HF dataset.")
        else:
            st.subheader("Top {}".format(top_k))
            st.dataframe(out, use_container_width=True)
