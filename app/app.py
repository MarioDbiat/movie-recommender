# app/app.py
# =========================================
# Movie Recommender — Streamlit app (FAISS-mmap fix)
# - Hugging Face–aware loaders (parquet / faiss / npy)
# - FAISS memory-mapped to avoid OOM on small machines
# - Falls back to embeddings.npy if FAISS missing/unusable
# =========================================

# --- Streamlit must be configured first ---
import streamlit as st
st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- Std/3p imports ---
import os
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# Keep Cloud safe (no CUDA) and avoid noisy tokenizer threads
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Silence only the specific FutureWarning from transformers -> torch.load
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
    module="transformers",
)

# Optional FAISS (app still runs without it)
try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
    try:
        faiss.omp_set_num_threads(1)  # be nice to CPU on shared machines
    except Exception:
        pass
except Exception:
    faiss = None
    _HAVE_FAISS = False

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# ---------------- Config ----------------
MODEL_NAME   = os.getenv("MODEL_NAME", "Mariodb/movie-recommender-model").strip()

# Base filenames that exist locally or on HF
PARQUET_FILE = os.getenv("PARQUET_FILE", "movies.parquet").strip()
INDEX_FILE   = os.getenv("INDEX_FILE",   "movie_index.faiss").strip()
EMB_FILE     = os.getenv("EMB_FILE",     "movie_embeddings.npy").strip()

# Hugging Face dataset repo info
HF_REPO_ID   = os.getenv("HF_REPO_ID", "Mariodb/movie-recommender-dataset").strip()
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset").strip()
HF_REVISION  = os.getenv("HF_REVISION", "main").strip()
HF_TOKEN     = os.getenv("HF_TOKEN")  # optional; only needed if repo is private

# Allow falling back to embeddings when FAISS fails (recommended=True)
FORCE_FAISS_ONLY = os.getenv("FORCE_FAISS_ONLY", "0").strip() == "1"

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# Candidate pool sizing (kept generous for recall; adjust if needed)
def choose_pool(top_k: int, use_genres: bool) -> int:
    return max(top_k * 20, top_k)

# ---------------- Path resolver (local / URL / HF) ----------------
def _resolve_path(path_or_name: str, *, must_be_local: bool) -> Optional[str]:
    """
    Return a local path to use. Order:
      1) If a local file exists at `path_or_name`, use it.
      2) If it's http(s) and not must_be_local -> return the URL (for pandas parquet).
      3) Attempt to download a file named basename(path_or_name) from the HF repo.
    For FAISS and npy we set must_be_local=True (they need a local file).
    """
    if not path_or_name:
        return None

    # Already a local file?
    if os.path.exists(path_or_name):
        return path_or_name

    # Remote URL allowed?
    if path_or_name.startswith(("http://", "https://")):
        return None if must_be_local else path_or_name

    # Try Hugging Face by filename
    filename = os.path.basename(path_or_name)
    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
            local_dir="data",
            local_dir_use_symlinks=False,
            token=HF_TOKEN,  # harmless if None for public repos
        )
        return local
    except Exception:
        return None

# ---------------- Caching layers ----------------
@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(show_spinner=False)
def load_metadata(parquet_path: str) -> pd.DataFrame:
    """
    Loads parquet from:
      - local path if present
      - http(s) URL (pandas + pyarrow)
      - Hugging Face (downloaded to ./data/)
    """
    resolved = _resolve_path(parquet_path, must_be_local=False)
    try:
        if resolved is None:
            if parquet_path.startswith(("http://", "https://")):
                df = pd.read_parquet(parquet_path)  # pyarrow handles URLs
            else:
                raise FileNotFoundError(f"Could not resolve '{parquet_path}' locally or from HF")
        else:
            df = pd.read_parquet(resolved)
    except Exception as e:
        st.error(f"Failed to load parquet: {parquet_path}\nError: {e}")
        return pd.DataFrame()

    # Reset index -> ensures row ↔ vector positions align
    df = df.reset_index(drop=True)

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
    """
    FAISS requires a local file. We resolve via:
      local path -> HF download to ./data -> else None
    Memory-map the index to avoid loading ~1.23 GB fully into RAM.
    """
    if not _HAVE_FAISS:
        return None
    local = _resolve_path(index_path, must_be_local=True)
    if not local or not os.path.exists(local):
        return None
    try:
        idx = faiss.read_index(local, faiss.IO_FLAG_MMAP)  # <-- critical fix
        # Optional tuning if index is IVF: modest nprobe
        try:
            if hasattr(idx, "nprobe"):
                idx.nprobe = int(os.getenv("FAISS_NPROBE", "16"))
        except Exception:
            pass
        return idx
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_embeddings(npy_path: str) -> Optional[np.ndarray]:
    """
    Embeddings need a local .npy. We resolve via:
      local path -> HF download to ./data -> else None
    """
    local = _resolve_path(npy_path, must_be_local=True)
    if not local or not os.path.exists(local):
        return None
    try:
        arr = np.load(local, mmap_mode="r")
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
    for i, t in enumerate(titles):
        if q in t.lower():
            return str(sample.iloc[i].get("franchise", "")).strip()
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
    ],
}

def detect_franchise(title: str, overview: str) -> str:
    text = f"{title} {overview}".lower()
    scores = {}
    for name, kws in FRANCHISE_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in text)
        if hits:
            scores[name] = hits
    return max(scores, key=scores.get) if scores else "Unknown"

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
    if df.empty:
        return pd.DataFrame()

    model = load_model()
    index_obj = load_faiss(index_name)

    # Prefer FAISS. Load embeddings only if FAISS is missing/unavailable (unless forced off).
    embeddings = None
    if (index_obj is None) and (not FORCE_FAISS_ONLY):
        embeddings = load_embeddings(emb_name or "")

    if index_obj is None and embeddings is None:
        st.error("Need either a FAISS index or an embeddings .npy to score.")
        return pd.DataFrame()

    if embeddings is not None and len(df) != len(embeddings):
        st.error("Embeddings and metadata shapes do not align.")
        return pd.DataFrame()

    query_clean = query_text.strip()
    if not query_clean:
        return pd.DataFrame()

    query_lower = query_clean.lower()
    movie_row = df[df["title"].str.lower() == query_lower] if "title" in df.columns else pd.DataFrame()

    # Prefer encoding the movie OVERVIEW if the user typed an exact title (richer signal)
    if not movie_row.empty:
        row = movie_row.iloc[0]
        overview = str(row.get("overview", "") or "")
        query_vec = model.encode(overview, convert_to_tensor=True)
        detected_franchise = detect_franchise(str(row.get("title", "")), overview)
        genres_val = row.get("genres", [])
        if isinstance(genres_val, list):
            input_genres = {g for g in genres_val if g and g.lower() != "unknown"}
        else:
            input_genres = set(_split_genres(str(genres_val)))
        use_name_anchor = True
    else:
        overview = query_clean
        query_vec = model.encode(overview, convert_to_tensor=True)
        detected_franchise = detect_franchise("", overview)
        input_genres = set()
        use_name_anchor = False

    query_np = query_vec.detach().cpu().numpy().reshape(-1).astype("float32")
    fanout = choose_pool(top_k, use_genres)

    candidate_idx = None
    sims = None

    # ---- Retrieve with FAISS (memory-mapped) ----
    if index_obj is not None:
        try:
            # Normalize query for cosine/IP style
            qn = query_np / (np.linalg.norm(query_np) + 1e-12)
            D, I = index_obj.search(qn[None, :], fanout)
            candidate_idx = I[0]
            sims = D[0].astype("float32")
        except Exception as exc:
            st.warning(f"FAISS search failed ({exc}); falling back to cosine.")
            candidate_idx = None
            sims = None

    # ---- Fallback: cosine over embeddings.npy (mmap) ----
    if (candidate_idx is None or len(candidate_idx) == 0) and embeddings is not None:
        qn = query_np / (np.linalg.norm(query_np) + 1e-12)
        En = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        sims_all = En @ qn
        topn = min(fanout, sims_all.shape[0])
        candidate_idx = np.argsort(-sims_all)[:topn]
        sims = sims_all[candidate_idx].astype("float32")

    if candidate_idx is None:
        return pd.DataFrame()

    candidate_idx = np.asarray(candidate_idx, dtype=int)
    candidate_idx = candidate_idx[(candidate_idx >= 0) & (candidate_idx < len(df))]
    if candidate_idx.size == 0 or sims is None:
        return pd.DataFrame()

    results = df.iloc[candidate_idx].copy()

    # Safety filter
    if safe_mode and not results.empty:
        bad_terms = ["nsfw", "adult", "porn", "xxx", "sex", "erotic", "babe"]
        pattern = "(?i)" + "|".join(bad_terms)
        mask = pd.Series(False, index=results.index)
        for col in ["genres", "title", "overview"]:
            if col in results.columns:
                mask |= results[col].astype(str).str.contains(pattern, na=False, regex=True)
        results = results.loc[~mask]

    if results.empty:
        return results

    # Franchise hard filter (only when toggle is ON)
    if franchise_only:
        q_fr = _find_query_movie_franchise(df, query_clean)
        if q_fr and q_fr.lower() != "unknown":
            if "franchise" not in results.columns:
                results["franchise"] = "Unknown"
            results = results[results["franchise"].astype(str).str.strip().str.lower() == q_fr.lower()]
        else:
            # If franchise not known, keep results as-is
            pass

    if results.empty:
        return results

    # Genre overlap filter + soft boost
    if use_genres and input_genres:
        def genre_overlap(val):
            if isinstance(val, list):
                working = {g for g in val if g and g.lower() != "unknown"}
            else:
                working = set(_split_genres(str(val)))
            return len(input_genres & working)

        results = results.assign(genre_overlap=results["genres"].apply(genre_overlap))
        min_required = 2 if len(input_genres) >= 2 else 1
        filtered = results[results["genre_overlap"] >= min_required]

        # If too strict, relax to at least 1 overlap
        if len(filtered) < top_k and results["genre_overlap"].max() >= 1:
            filtered = results[results["genre_overlap"] >= 1]

        results = filtered
        if results.empty:
            return results

    # Exclude the exact query title when anchoring on title
    if use_name_anchor and "title" in results.columns:
        results = results[results["title"].str.lower() != query_lower]
        if results.empty:
            return results

    # Build a score
    sims = sims[: len(results)].astype("float32")  # align length
    if use_popularity and "popularity" in results.columns:
        pop_vals = pd.to_numeric(results["popularity"], errors="coerce").fillna(0).to_numpy("float32")
        score = (sims * np.log1p(pop_vals)).astype("float32")
    else:
        score = sims

    # Soft bonus for genre overlap (if computed)
    if "genre_overlap" in results.columns:
        score = score * (1.0 + 0.05 * results["genre_overlap"].to_numpy("float32"))

    results = results.assign(score=score, similarity=sims)

    # Build poster_url for UI cards
    if "poster_path" in results.columns:
        results["poster_url"] = results["poster_path"].astype(str).str.strip()
        mask = results["poster_url"].str.len() > 0
        results.loc[mask, "poster_url"] = TMDB_IMG_BASE + results.loc[mask, "poster_url"]
    elif "poster_url" not in results.columns:
        results["poster_url"] = ""

    keep = ["title", "genres", "franchise", "popularity", "poster_url", "overview", "score", "similarity"]
    if "id" in results.columns:
        keep.append("id")
    existing = [c for c in keep if c in results.columns]

    results = results.sort_values(by="score", ascending=False)
    return results[existing].head(top_k).reset_index(drop=True)

# ---------------- UI ----------------
st.title("🎬 Mario's Netflix")

with st.expander("Technical info"):
    st.write(f"Model: `{MODEL_NAME}`")
    st.write(f"Parquet: `{PARQUET_FILE}`  (exists: {os.path.exists(PARQUET_FILE)})")
    st.write(f"FAISS index: `{INDEX_FILE}`  (exists: {os.path.exists(INDEX_FILE)})  FAISS available: `{_HAVE_FAISS}`  | FORCE_FAISS_ONLY={FORCE_FAISS_ONLY}")
    st.write(f"Embeddings: `{EMB_FILE}`  (exists: {os.path.exists(EMB_FILE)})  (loaded only if no FAISS)")
    st.write(f"HF repo: `{HF_REPO_ID}`  type=`{HF_REPO_TYPE}`  rev=`{HF_REVISION}`")

st.caption(
    "Type a title or describe what you want to watch. "
    "Tip: the Genre filter works best when you enter an exact movie title."
)

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
        help="HARD-ish filter: prefer results sharing ≥2 genres with the query (works best with exact title)."
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
                use_popularity=use_popularity,
                top_k=top_k,
            )

        if out is None or len(out) == 0:
            st.info("No results. Try a different title/description or relax filters.")
        else:
            for _, row in out.iterrows():
                poster_url = row.get("poster_url") or "https://via.placeholder.com/120x180?text=No+Image"
                c1, c2 = st.columns([1, 5], vertical_alignment="center")
                with c1:
                    st.image(poster_url, width=110)
                with c2:
                    st.markdown(f"### {row.get('title','Untitled')}")
                    if row.get("genres"):
                        if isinstance(row["genres"], list):
                            st.caption(f"Genres: {', '.join(row['genres'])}")
                        else:
                            st.caption(f"Genres: {row['genres']}")
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

# Footer
st.markdown("<br><hr><center>Made with ❤️ by Mario</center>", unsafe_allow_html=True)
