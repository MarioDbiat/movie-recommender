# app/app.py
# =========================================
# Movie Recommender — dual-mode (Cloud + Local), Py3.8-safe
# Uses your HF files:
#   movies.parquet, movie_index.faiss, movie_embeddings.npy
# =========================================

import os

# ---- Streamlit MUST be the first command used ----
import streamlit as st
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Keep Cloud safe (no CUDA) and avoid noisy tokenizer threads
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
MODE = os.getenv("MODE", "auto").lower().strip()
LOAD_EMB = os.getenv("LOAD_EMB", "0") == "1"  # embeddings fallback (off by default)

# Hugging Face dataset repo
HF_REPO_ID   = os.getenv("HF_REPO_ID", "Mariodb/movie-recommender-dataset").strip()
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset").strip()
HF_REVISION  = os.getenv("HF_REVISION", "main").strip()

# Filenames in your HF repo (you showed these names)
LITE_PARQUET = os.getenv("LITE_PARQUET", "movies.parquet").strip()
LITE_INDEX   = os.getenv("LITE_INDEX",   "movie_index.faiss").strip()

FULL_PARQUET = os.getenv("FULL_PARQUET", "movies.parquet").strip()
FULL_INDEX   = os.getenv("FULL_INDEX",   "movie_index.faiss").strip()
FULL_EMB     = os.getenv("FULL_EMB",     "movie_embeddings.npy").strip()

# Local fallbacks (used only if HF fetch fails AND files exist locally)
DATA_CSV   = os.getenv("DATA_CSV",   "data/movies.csv")
FAISS_PATH = os.getenv("FAISS_PATH", "indexes/movie_index.faiss")
EMB_PATH   = os.getenv("EMB_PATH",   "artifacts/movie_embeddings.npy")

# Model
MODEL_ID = os.getenv("MODEL_ID", "Mariodb/movie-recommender-model").strip()

# Pools (Cloud-friendly defaults)
POOL_BASE = int(os.getenv("POOL_BASE", "50"))
POOL_WIDE = int(os.getenv("POOL_WIDE", "300"))
POOL_MAX  = int(os.getenv("POOL_MAX",  "500"))

# FAISS availability
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
    """Download a file from the HF dataset repo (trims env vars; supports revision)."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type=HF_REPO_TYPE,
        revision=HF_REVISION,
        local_dir="data",
        local_dir_use_symlinks=False,
        # token=os.getenv("HF_TOKEN")  # uncomment if your repo is private
    )

def _choose_files() -> Tuple[str, str, Optional[str]]:
    """
    Decide which artifacts to use (lite vs full) WITHOUT pre-downloading.
    We try lite first during load; on 404 we fall back to full automatically.
    """
    if MODE == "lite":
        return (LITE_PARQUET, LITE_INDEX, None)
    if MODE == "full":
        return (FULL_PARQUET, FULL_INDEX, FULL_EMB if LOAD_EMB else None)
    # auto
    return (LITE_PARQUET, LITE_INDEX, None)

# ---------------- Cached loaders ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_ID)

@st.cache_resource
def load_metadata(parquet_name: str) -> pd.DataFrame:
    def _load_one(name: str) -> pd.DataFrame:
        path = _hf_path(name)
        return pd.read_parquet(path)

    try:
        df = _load_one(parquet_name)
    except Exception as e:
        msg = str(e).lower()
        if ("404" in msg or "entry not found" in msg) and parquet_name != FULL_PARQUET:
            try:
                df = _load_one(FULL_PARQUET)  # fallback to full name
            except Exception as e2:
                if os.path.exists(DATA_CSV):
                    df = pd.read_csv(DATA_CSV)
                else:
                    st.error(
                        f"Couldn't fetch dataset from Hugging Face and no local CSV found.\n\n"
                        f"Tried: {HF_REPO_ID}/{parquet_name} then {HF_REPO_ID}/{FULL_PARQUET} and {DATA_CSV}\n\n"
                        f"Error: {e2}"
                    )
                    st.stop()
        else:
            if os.path.exists(DATA_CSV):
                df = pd.read_csv(DATA_CSV)
            else:
                st.error(
                    f"Couldn't fetch dataset from Hugging Face and no local CSV found.\n\n"
                    f"Tried: {HF_REPO_ID}/{parquet_name} and {DATA_CSV}\n\nError: {e}"
                )
                st.stop()

    # Normalize columns we use
    for col in ["title", "overview", "genres", "franchise"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(lambda s: "" if str(s).strip().lower() == "unknown" else s)

    # Ensure franchise column
    df = _ensure_franchise(df)

    # If some columns are missing, create them so UI doesn't crash
    for col in ["title", "genres", "franchise", "popularity", "overview"]:
        if col not in df.columns:
            df[col] = "" if col != "popularity" else 0.0

    return df

@st.cache_resource
def load_faiss(index_name: str):
    if not FAISS_OK:
        st.warning("FAISS is not available; will use embeddings only if enabled.")
        return None

    def _load_one(name: str):
        # Use local path if present, otherwise fetch from HF
        path = FAISS_PATH if os.path.exists(FAISS_PATH) else _hf_path(name)
        idx = faiss.read_index(path, faiss.IO_FLAG_MMAP)  # mmap to avoid huge RAM
        try:
            idx.nprobe = int(os.getenv("FAISS_NPROBE", "16"))  # good default for IVF
        except Exception:
            pass
        return idx

    try:
        return _load_one(index_name)
    except Exception as e:
        msg = str(e).lower()
        if ("404" in msg or "entry not found" in msg) and index_name != FULL_INDEX:
            try:
                return _load_one(FULL_INDEX)  # fallback to full name
            except Exception as e2:
                st.error(f"FAISS load failed (tried {index_name} then {FULL_INDEX}): {e2}")
                return None
        else:
            st.error(f"FAISS load failed: {e}")
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
        st.warning(f"Embeddings fallback not available: {e}")
        return None

# ---------------- Search helpers ----------------
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

def cosine_topk(query_vec: np.ndarray, emb: np.ndarray, k: int):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = e @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def run_search(parquet_name: str, index_name: str, emb_name: Optional[str],
               query_text: str, franchise_only: bool, safe_mode: bool,
               use_genres: bool, top_k: int) -> pd.DataFrame:
    """Return a top-k DataFrame with columns ready for the UI (incl. poster_url)."""
    meta = load_metadata(parquet_name)
    model = load_model()

    # Prefer encoding the movie OVERVIEW if the user typed an exact title (richer signal)
    _q = query_text.strip().lower()
    if "title" in meta.columns:
        q_row = meta[meta["title"].str.lower() == _q]
    else:
        q_row = pd.DataFrame()

    if (len(q_row) == 1
        and "overview" in q_row.columns
        and isinstance(q_row.iloc[0]["overview"], str)
        and len(q_row.iloc[0]["overview"]) > 20):
        q_vec = model.encode([q_row.iloc[0]["overview"]])[0].astype("float32")
    else:
        q_vec = model.encode([query_text])[0].astype("float32")

    # Candidate pool size
    base_pool = POOL_WIDE if use_genres else POOL_BASE
    pool = int(min(POOL_MAX, max(base_pool, top_k * 10)))

    # ---- Retrieve candidate ids (FAISS -> cosine fallback) ----
    idx = None
    faiss_index = load_faiss(index_name)
    if faiss_index is not None:
        try:
            q = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            D, I = faiss_index.search(q[None, :], pool)
            idx = I[0]
            if idx is not None:
                idx = idx[idx >= 0]  # guard -1 entries
        except Exception as e:
            st.warning(f"FAISS search failed; will try embeddings if enabled. Error: {e}")
            idx = None

    if idx is None or len(idx) == 0:
        emb = load_embeddings(emb_name)
        if emb is None:
            st.error("Search index unavailable (no FAISS / embeddings). "
                     "Upload FAISS to HF or enable embeddings locally with LOAD_EMB=1.")
            return pd.DataFrame()
        idx, _ = cosine_topk(q_vec, emb, pool)

    if idx is None or len(idx) == 0:
        return pd.DataFrame()

    results = meta.iloc[idx].copy()

    # ---- Post-filtering & safety ----
    q_lower = query_text.strip().lower()
    if "title" in results.columns and q_lower:
        results = results[results["title"].str.lower() != q_lower]  # drop exact self-match

    # Franchise = HARD FILTER (same as notebook)
    if franchise_only and "franchise" in results.columns:
        q_fr = _find_query_movie_franchise(meta, query_text)
        if q_fr and q_fr.lower() != "unknown":
            results = results[results["franchise"].str.strip().str.lower() == q_fr.lower()]

    if safe_mode and not results.empty:
        bad = ["nsfw", "adult", "porn", "xxx", "sex", "erotic", "babe"]
        pattern = "(?i)" + "|".join(bad)
        mask = pd.Series(False, index=results.index)
        for col in ["genres", "title", "overview"]:
            if col in results.columns:
                mask |= results[col].astype(str).str.contains(pattern, na=False, regex=True)
        results = results.loc[~mask]

    if results.empty:
        return results

    # --- GENRE = HARD THRESHOLD (>=2 overlaps), do it BEFORE scoring ---
    if use_genres and "genres" in results.columns:
        qg = _find_query_movie_genres(meta, query_text)
        if qg:
            qset = set(qg)
            def _ov(g):
                if isinstance(g, list):
                    rset = set(g)
                else:
                    rset = set(_split_genres(str(g)))
                return len(qset & rset)
            results["genre_overlap"] = results["genres"].apply(_ov)
            results = results[results["genre_overlap"] >= 2]
            if results.empty:
                return results

    # ---- Re-rank similarity for candidates ----
    sims = None
    try:
        emb = load_embeddings(emb_name)
        if emb is not None:
            cand_vecs = emb[results.index.to_numpy()]
            q = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            E = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
            sims = (E @ q).astype("float32")
    except Exception:
        sims = None

    if sims is None and "overview" in results.columns:
        try:
            cand_vecs = model.encode(results["overview"].fillna("").tolist()).astype("float32")
            q = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            E = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
            sims = (E @ q).astype("float32")
        except Exception:
            sims = None

    if sims is None:
        sims = np.zeros(len(results), dtype="float32")

    # Normalize sims from [-1,1] -> [0,1] to mix with other [0,1] features
    sims01 = (sims + 1.0) * 0.5
    results["sim"] = sims01

    # ---- Feature terms (all 0..1) ----
    if "popularity" in results.columns:
        pop_log = np.log1p(pd.to_numeric(results["popularity"], errors="coerce").fillna(0).to_numpy("float32"))
        pop_log = pop_log / (pop_log.max() + 1e-6)
    else:
        pop_log = np.zeros(len(results), dtype="float32")

    # For notebook parity: no soft genre bonus
    genre_bonus = np.zeros(len(results), dtype="float32")

    # Only compute franchise boost when the checkbox is ON (otherwise no soft effect)
    fr_boost = np.zeros(len(results), dtype="float32")
    if franchise_only and "franchise" in results.columns:
        q_fr = _find_query_movie_franchise(meta, query_text)
        if q_fr and q_fr.strip().lower() != "unknown":
            fr_boost = (
                results["franchise"].astype(str).str.strip().str.lower()
                == q_fr.strip().lower()
            ).to_numpy("bool").astype("float32")

    # ---- Weights and final score ----
    W_SIM, W_POP, W_GEN, W_FR = 0.70, 0.15, 0.00, 0.15
    score = (
        W_SIM * results["sim"].to_numpy("float32") +
        W_POP * pop_log +
        W_GEN * genre_bonus +
        W_FR  * fr_boost
    )
    results = results.assign(score=score).sort_values("score", ascending=False)

    # ---- Build poster_url for UI cards ----
    if "poster_path" in results.columns:
        results["poster_url"] = results["poster_path"].astype(str).str.strip()
        mask = results["poster_url"].str.len() > 0
        results.loc[mask, "poster_url"] = TMDB_IMG_BASE + results.loc[mask, "poster_url"]

    # ---- Return top-k with the columns the UI needs ----
    base_cols = ["title", "genres", "popularity"]
    if "franchise" in results.columns:
        base_cols.append("franchise")
    extras = [c for c in ["overview", "poster_path", "poster_url", "sim", "score"] if c in results.columns]
    keep = [c for c in base_cols if c in results.columns] + extras

    return results[keep].head(top_k).reset_index(drop=True)

# ---------------- UI helpers ----------------
def render_results_as_cards(df: pd.DataFrame, show_franchise: bool = False):
    if df.empty:
        st.info("No results found.")
        return

    for _, row in df.iterrows():
        poster_url = row.get("poster_url") or "https://via.placeholder.com/120x180?text=No+Image"
        c1, c2 = st.columns([1, 5], vertical_alignment="center")
        with c1:
            st.image(poster_url, width=110)
        with c2:
            st.markdown(f"### {row.get('title','Untitled')}")
            if row.get("genres"):
                st.caption(f"Genres: {row['genres']}")
            # show franchise ONLY if the toggle is on
            if show_franchise and row.get("franchise"):
                st.caption(f"🎬 Franchise: {row['franchise']}")
            if row.get("popularity") is not None:
                st.caption(f"⭐ Popularity: {row['popularity']:.2f}")
        st.divider()

# Pick which files to use (lite/full/auto) BEFORE the UI
PARQUET_FILE, INDEX_FILE, EMB_FILE = _choose_files()

# ---------------- UI ----------------
st.title("🎬 Mario's Netflix")

# Friendly intro for end-users
st.caption(
    "Looking for your next favorite movie? Type a title or describe what you want to watch, "
    "and we’ll suggest the best matches — from classics to hidden gems. "
    "_Tip: Genre filters only work when you search by an exact movie title._"
)

# Keep developer info out of the way but accessible
with st.expander("Technical info"):
    st.write(f"Mode: {MODE}")
    st.write(f"Repo: {HF_REPO_ID}")
    st.write(f"Parquet: {PARQUET_FILE}")
    st.write(f"Index: {INDEX_FILE}")
    st.write(f"Embeddings: {EMB_FILE if LOAD_EMB else 'Disabled'}")

# ---- Sections: Search / Filters / Results ----
st.subheader("🔍 Search")
query = st.text_input("Title or description", placeholder="e.g., Interstellar — gritty space survival")

st.divider()

st.subheader("⚙️ Filters")
col1, col2, col3 = st.columns(3)
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

top_k = st.slider(
    "How many recommendations?",
    min_value=5, max_value=50, value=10, step=5,
    help="More items may be a bit slower."
)

st.divider()

st.subheader("🎯 Recommendations")

if st.button("Recommend"):
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
                top_k=top_k,
            )
        if out.empty:
            st.info("No results. Try fewer filters or a different query.")
        else:
            # Posters + nice layout
            render_results_as_cards(out, show_franchise=franchise_only)

# Small footer
st.markdown("<br><hr><center>Made with ❤️ by Mario</center>", unsafe_allow_html=True)
