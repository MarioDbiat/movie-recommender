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

NSFW_TERMS = ["nsfw", "adult", "porn", "xxx", "sex", "erotic", "babe"]

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

def run_search(
    parquet_name: str,
    index_name: str,
    emb_name: Optional[str],
    query_text: str,
    franchise_only: bool,
    safe_mode: bool,
    use_genres: bool,
    use_popularity: bool,
    top_k: int,
) -> pd.DataFrame:
    """Return a top-k DataFrame with upgraded scoring and safety filters."""
    meta = load_metadata(parquet_name)
    if meta.empty:
        return pd.DataFrame()

    meta = _ensure_franchise(meta.copy())
    model = load_model()

    embeddings = None
    if emb_name:
        embeddings = load_embeddings(emb_name)
    elif LOAD_EMB and FULL_EMB:
        embeddings = load_embeddings(FULL_EMB)

    if embeddings is not None and len(embeddings) != len(meta):
        st.warning("Embeddings and metadata have different lengths; ignoring embeddings for now.")
        embeddings = None

    query_clean = query_text.strip()
    if not query_clean:
        return pd.DataFrame()

    query_lower = query_clean.lower()
    exact = meta[meta["title"].str.lower() == query_lower] if "title" in meta.columns else pd.DataFrame()

    detected_franchise = "Unknown"
    input_genres = set()
    use_name_anchor = False

    if not exact.empty:
        row = exact.iloc[0]
        overview = str(row.get("overview", "") or query_clean)
        q_vec = model.encode([overview], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        detected_franchise = (
            str(row.get("franchise", "")).strip()
            or detect_franchise(row.get("title", ""), overview)
            or "Unknown"
        )
        genres_val = row.get("genres", [])
        if isinstance(genres_val, list):
            input_genres = {
                str(g).strip() for g in genres_val if str(g).strip() and str(g).lower() != "unknown"
            }
        else:
            input_genres = {
                g for g in _split_genres(str(genres_val)) if g and g.lower() != "unknown"
            }
        use_name_anchor = True
    else:
        q_vec = model.encode([query_clean], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        detected_franchise = detect_franchise("", query_clean) or "Unknown"

    if use_genres and not input_genres:
        fallback_genres = _find_query_movie_genres(meta, query_text)
        input_genres = {
            g for g in fallback_genres if g and g.lower() != "unknown"
        }

    base_pool = POOL_WIDE if use_genres else POOL_BASE
    pool = int(min(POOL_MAX, max(base_pool, top_k * 10)))

    faiss_indices = None
    faiss_index = load_faiss(index_name)
    if faiss_index is not None:
        try:
            _, faiss_results = faiss_index.search(q_vec[None, :], pool)
            faiss_indices = faiss_results[0]
            if faiss_indices is not None:
                faiss_indices = faiss_indices[faiss_indices >= 0]
        except Exception as exc:
            st.warning(f"FAISS search failed; falling back to embeddings. Error: {exc}")
            faiss_indices = None

    if (faiss_indices is None or len(faiss_indices) == 0) and embeddings is not None:
        faiss_indices, _ = cosine_topk(q_vec, embeddings, pool)

    if faiss_indices is None or len(faiss_indices) == 0:
        st.error(
            "Search index unavailable (no FAISS results and embeddings not provided). "
            "Upload FAISS to Hugging Face or include embeddings locally."
        )
        return pd.DataFrame()

    faiss_indices = np.asarray(faiss_indices, dtype=int)
    faiss_indices = faiss_indices[(faiss_indices >= 0) & (faiss_indices < len(meta))]
    if faiss_indices.size == 0:
        return pd.DataFrame()

    results = meta.iloc[faiss_indices].copy()

    sims: Optional[np.ndarray] = None
    if embeddings is not None:
        try:
            cand_vecs = embeddings[faiss_indices].astype("float32")
            norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12
            normalized = cand_vecs / norms
            sims = (normalized @ q_vec).astype("float32")
        except Exception:
            sims = None

    if sims is None:
        try:
            texts = (
                results["overview"].fillna("").astype(str).tolist()
                if "overview" in results.columns
                else results["title"].astype(str).tolist()
            )
            cand_vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            sims = (cand_vecs @ q_vec).astype("float32")
        except Exception:
            sims = np.zeros(len(results), dtype="float32")

    sims = np.squeeze(sims).astype("float32")
    scores = sims.copy()

    if use_popularity and "popularity" in results.columns:
        pop_vals = pd.to_numeric(results["popularity"], errors="coerce").fillna(0).to_numpy("float32")
        pop_scale = np.log1p(np.clip(pop_vals, 0, None))
        max_pop = float(pop_scale.max()) if pop_scale.size else 0.0
        if max_pop > 0:
            pop_scale = pop_scale / max_pop
        scores = scores * (1.0 + pop_scale)

    results = results.assign(score=scores.astype("float32"), similarity=sims)

    if safe_mode and not results.empty:
        if "nsfw" in results.columns:
            results = results[~results["nsfw"].astype(bool)]
        else:
            pattern = "(?i)" + "|".join(NSFW_TERMS)
            mask = pd.Series(False, index=results.index)
            for col in ["genres", "title", "overview"]:
                if col in results.columns:
                    mask |= results[col].astype(str).str.contains(pattern, na=False, regex=True)
            results = results.loc[~mask]
        if results.empty:
            return results

    target_franchise = detected_franchise.strip()
    if not target_franchise or target_franchise.lower() == "unknown":
        target_franchise = _find_query_movie_franchise(meta, query_text).strip()

    if franchise_only and target_franchise and target_franchise.lower() != "unknown":
        if "franchise" not in results.columns:
            results["franchise"] = "Unknown"
        missing_mask = results["franchise"].fillna("").str.strip().isin(["", "Unknown"])
        if missing_mask.any():
            results.loc[missing_mask, "franchise"] = results.loc[missing_mask].apply(
                lambda row: detect_franchise(str(row.get("title", "")), str(row.get("overview", ""))) or "Unknown",
                axis=1,
            )
        results = results[
            results["franchise"].astype(str).str.strip().str.lower() == target_franchise.lower()
        ]
        if results.empty:
            return results

    if use_genres and input_genres and "genres" in results.columns:
        def genre_overlap(val) -> int:
            if isinstance(val, list):
                working = {str(g).strip() for g in val if str(g).strip() and str(g).lower() != "unknown"}
            else:
                working = set(_split_genres(str(val)))
            return len(input_genres & working)

        overlaps = results["genres"].apply(genre_overlap)
        results = results.assign(genre_overlap=overlaps)
        min_required = 2 if len(input_genres) >= 2 else 1
        max_overlap = int(results["genre_overlap"].max()) if not results.empty else 0
        filtered = results[results["genre_overlap"] >= min_required]
        if len(filtered) < top_k and max_overlap >= 1:
            filtered = results[results["genre_overlap"] >= 1]
        results = filtered
        if results.empty:
            return results
        results = results.assign(score=results["score"] * (1.0 + 0.05 * results["genre_overlap"]))

    if use_name_anchor and "title" in results.columns:
        results = results[results["title"].astype(str).str.lower() != query_lower]
        if results.empty:
            return results

    results = results.sort_values(by="score", ascending=False)

    if "poster_path" in results.columns:
        results["poster_url"] = results["poster_path"].astype(str).str.strip()
        mask = results["poster_url"].str.len() > 0
        results.loc[mask, "poster_url"] = TMDB_IMG_BASE + results.loc[mask, "poster_url"]
    elif "poster_url" not in results.columns:
        results["poster_url"] = ""

    keep = [
        col for col in [
            "title",
            "genres",
            "franchise",
            "popularity",
            "overview",
            "poster_path",
            "poster_url",
            "similarity",
            "score",
            "genre_overlap",
            "id",
        ]
        if col in results.columns
    ]
    return results[keep].head(top_k).reset_index(drop=True)

# ---------------- UI helpers ----------------
def render_results_as_cards(df: pd.DataFrame, show_franchise: bool = False) -> None:
    if df.empty:
        st.info("No results found.")
        return

    for _, row in df.iterrows():
        poster_url = row.get("poster_url") or "https://via.placeholder.com/120x180?text=No+Image"
        left, right = st.columns([1, 5], vertical_alignment="center")
        with left:
            st.image(poster_url, width=110)
        with right:
            st.markdown(f"### {row.get('title', 'Untitled')}")
            genres_val = row.get("genres")
            if genres_val:
                if isinstance(genres_val, list):
                    genres_text = ", ".join(str(g) for g in genres_val if g)
                else:
                    genres_text = str(genres_val)
                st.caption(f"Genres: {genres_text}")
            if show_franchise and row.get("franchise"):
                st.caption(f"Franchise: {row['franchise']}")
            if row.get("popularity") is not None:
                st.caption(f"Popularity: {float(row['popularity']):.2f}")
            if row.get("score") is not None:
                st.caption(f"Score: {float(row['score']):.3f}")
            if row.get("overview"):
                st.write(row["overview"])
        st.divider()

# Pick which files to use (lite/full/auto) BEFORE the UI
PARQUET_FILE, INDEX_FILE, EMB_FILE = _choose_files()

# ---------------- UI ----------------
st.title("Mario's Netflix")

# Friendly intro for end-users
st.caption(
    "Looking for your next favorite movie? Type a title or describe what you want to watch, "
    "and we'll suggest the best matches from classics to hidden gems. "
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
st.subheader("Search")
query = st.text_input("Title or description", placeholder="e.g., Interstellar - gritty space survival")

st.divider()

st.subheader("Filters")
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
        help="Hard filter: keep only results sharing at least 2 genres with the query (exact title works best)."
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
        help="Blend similarity with popularity so blockbusters rank a little higher."
    )

top_k = st.slider(
    "How many recommendations?",
    min_value=5, max_value=50, value=10, step=5,
    help="More items may be a bit slower."
)

st.divider()

st.subheader("Recommendations")

if st.button("Recommend", type="primary"):
    if not query.strip():
        st.warning("Please enter a title or short description.")
    else:
        with st.spinner("Finding great matches..."):
            out = run_search(
                parquet_name=PARQUET_FILE,
                index_name=INDEX_FILE,
                emb_name=EMB_FILE,
                query_text=query.strip(),
                franchise_only=franchise_only,
                safe_mode=safe_mode,
                use_genres=use_genres,
                use_popularity=use_popularity,
                top_k=top_k,
            )
        if out.empty:
            st.info("No results. Try fewer filters or a different query.")
        else:
            render_results_as_cards(out, show_franchise=franchise_only)

# Small footer
st.markdown("<br><hr><center>Made with love by Mario</center>", unsafe_allow_html=True)
