# app.py — FAISS-only, Cloud-safe
# =========================================
# Uses FAISS for candidate retrieval; encodes candidates on-the-fly for scoring.
# CPU by default; FAISS mmap; capped fanout; small fallback pool; IVF nprobe set.
# =========================================

import streamlit as st
st.set_page_config(page_title="Movie Recommender", layout="wide")

import os
import warnings
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
    module="transformers.modeling_utils",
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU by default
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---- FAISS (optional) -------------------------------------------------------
try:
    import faiss
    _HAVE_FAISS = True
    try:
        faiss.omp_set_num_threads(1)  # avoid CPU spikes
    except Exception:
        pass
except Exception:
    faiss = None
    _HAVE_FAISS = False

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import torch
torch.set_num_threads(1)

# ---------------- Config (reads from Secrets or falls back to defaults) ------
MODEL_NAME   = os.getenv("MODEL_NAME", "Mariodb/movie-recommender-model").strip()
PARQUET_FILE = os.getenv("PARQUET_FILE", "movies.parquet").strip()
INDEX_FILE   = os.getenv("INDEX_FILE",   "movie_index.faiss").strip()
EMB_FILE     = os.getenv("EMB_FILE", "").strip()   # ⬅️ keep empty in cloud (FAISS-only)

HF_REPO_ID   = os.getenv("HF_REPO_ID", "Mariodb/movie-recommender-dataset").strip()
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset").strip()
HF_REVISION  = os.getenv("HF_REVISION", "main").strip()
HF_TOKEN     = os.getenv("HF_TOKEN")  # optional, needed for private repos

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# ---------------- Cloud caps --------------------------------------------------
CLOUD_MODE = True
CLOUD_FANOUT_MAX = 64        # tighter cap to stay well under RAM
CLOUD_FALLBACK_POOL = 2500   # encode-at-most this many if FAISS missing

def choose_pool(top_k: int, use_genres: bool) -> int:
    base = max(top_k * 20, top_k)  # same spirit as your old app
    return min(base, CLOUD_FANOUT_MAX) if CLOUD_MODE else base

# ---------------- Hugging Face path resolver ---------------------------------
def _resolve_path(path_or_name: str, *, must_be_local: bool) -> Optional[str]:
    """Prefer local; else URL (if allowed); else download from HF to ./data."""
    if not path_or_name:
        return None
    if os.path.exists(path_or_name):
        return path_or_name
    if path_or_name.startswith(("http://", "https://")):
        return None if must_be_local else path_or_name

    filename = os.path.basename(path_or_name)
    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
            local_dir="data",
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
        return local
    except Exception:
        return None

# ---------------- Caches ------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    device = "cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES")) else "cpu"
    return SentenceTransformer(MODEL_NAME, device=device)

@st.cache_data(show_spinner=False)
def load_metadata(parquet_path: str) -> pd.DataFrame:
    resolved = _resolve_path(parquet_path, must_be_local=False)
    try:
        if resolved is None:
            if parquet_path.startswith(("http://", "https://")):
                df = pd.read_parquet(parquet_path)
            else:
                raise FileNotFoundError(f"Could not resolve '{parquet_path}' locally or from HF")
        else:
            df = pd.read_parquet(resolved)
    except Exception as e:
        st.error(f"Failed to load parquet: {parquet_path}\nError: {e}")
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    if "title" in df.columns: df["title"] = df["title"].astype(str)
    if "overview" in df.columns: df["overview"] = df["overview"].fillna("").astype(str)
    if "genres" in df.columns:
        def _fix_genres(x):
            if isinstance(x, list):
                return [g for g in x if str(g).strip().lower() != "unknown"]
            s = str(x).strip()
            if not s or s.lower() == "unknown": return []
            for sep in ("|", ",", ";", "/"):
                if sep in s:
                    return [g.strip() for g in s.split(sep)
                            if g.strip() and g.strip().lower() != "unknown"]
            return [s] if s.lower() != "unknown" else []
        df["genres"] = df["genres"].apply(_fix_genres)
    if "franchise" in df.columns: df["franchise"] = df["franchise"].fillna("Unknown").astype(str)
    if "popularity" in df.columns: df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    return df

@st.cache_resource(show_spinner=False)
def load_faiss(index_path: str):
    if not _HAVE_FAISS:
        return None
    local = _resolve_path(index_path, must_be_local=True)
    if not local or not os.path.exists(local):
        return None
    try:
        idx = faiss.read_index(local, faiss.IO_FLAG_MMAP)
    except Exception:
        try:
            idx = faiss.read_index(local)
        except Exception:
            return None
    # ✅ IVF/PQ-friendly probe (ignored by Flat indexes)
    try:
        idx.nprobe = 16
    except Exception:
        pass
    return idx

@st.cache_resource(show_spinner=False)
def load_embeddings(npy_path: str) -> Optional[np.ndarray]:
    # Cloud profile: EMB_FILE="" → always skip
    npy_path = (npy_path or "").strip()
    if npy_path == "":
        return None
    local = _resolve_path(npy_path, must_be_local=True)
    if not local or not os.path.exists(local):
        return None
    try:
        arr = np.load(local, mmap_mode="r")
        return arr.astype("float32") if arr.dtype != np.float32 else arr
    except Exception:
        return None

# ---------------- Utilities ---------------------------------------------------
def _split_genres(s: str) -> List[str]:
    if not isinstance(s, str): return []
    s = s.strip()
    if not s or s.lower() == "unknown": return []
    for sep in ["|", ",", ";", "/"]:
        if sep in s:
            return [g.strip() for g in s.split(sep) if g.strip() and g.lower() != "unknown"]
    return [s] if s.lower() != "unknown" else []

FRANCHISE_KEYWORDS = {
    "Marvel": ["avengers","iron man","captain america","thor","hulk","black widow","spider-man","doctor strange"],
    "DC": ["batman","superman","wonder woman","aquaman","flash","justice league","joker","shazam"],
    "Harry Potter": ["harry potter","hogwarts","voldemort","dumbledore","hermione","ron weasley","fantastic beasts"],
    "Lord of the Rings": ["lord of the rings","frodo","gandalf","aragorn","middle earth","sauron","legolas","hobbit"],
    "Star Wars": ["star wars","skywalker","darth vader","yoda","jedi","sith","death star","mandalorian","obi-wan"],
}
def detect_franchise(title: str, overview: str) -> str:
    text = f"{title} {overview}".lower()
    matches = {}
    for name, kws in FRANCHISE_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in text)
        if hits: matches[name] = hits
    return max(matches, key=matches.get) if matches else "Unknown"

# ---------------- Core search -------------------------------------------------
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

    df = load_metadata(parquet_name)
    model = load_model()
    index_obj = load_faiss(index_name)
    embeddings = load_embeddings(emb_name or "")

    if len(df) == 0: return pd.DataFrame()
    q_text = query_text.strip()
    if not q_text: return pd.DataFrame()

    # Build query vector (exact title → use its overview; else free text)
    exact = df[df["title"].str.lower() == q_text.lower()] if "title" in df.columns else pd.DataFrame()
    if not exact.empty:
        row = exact.iloc[0]
        overview = str(row.get("overview", "") or "")
        q_vec = model.encode([overview], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_franchise = row.get("franchise") or detect_franchise(str(row.get("title","")), overview)
        q_genres = row.get("genres", [])
        if not isinstance(q_genres, list): q_genres = _split_genres(str(q_genres))
    else:
        overview = q_text
        q_vec = model.encode([overview], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_franchise = detect_franchise("", overview)
        q_genres = []

    q_np = q_vec.reshape(1, -1).astype("float32")
    fanout = choose_pool(top_k, use_genres)

    # --- Candidate retrieval
    cand_idx = None

    # A) FAISS
    if index_obj is not None:
        try:
            _, I = index_obj.search(q_np, fanout)
            cand_idx = I[0]
        except Exception as exc:
            st.warning(f"FAISS search failed ({exc}); falling back to popularity pool.")
            cand_idx = None

    # B) Popularity pool if FAISS missing/failed
    if cand_idx is None or len(cand_idx) == 0:
        pool_n = CLOUD_FALLBACK_POOL if CLOUD_MODE else max(5000, fanout)
        if "popularity" in df.columns:
            pool = df.nlargest(min(pool_n, len(df)), "popularity").copy()
        else:
            pool = df.head(min(pool_n, len(df))).copy()

        texts = (pool["overview"].fillna("").astype(str).tolist()
                 if "overview" in pool.columns else
                 pool["title"].astype(str).tolist())
        cand_vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        q = (q_np[0] / (np.linalg.norm(q_np[0]) + 1e-12)).astype("float32")
        E = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
        sims = (E @ q).astype("float32")
        part = np.argpartition(-sims, min(len(sims)-1, fanout-1))[:fanout]
        order = np.argsort(-sims[part])
        top_local = part[order]
        cand_idx = pool.index.to_numpy()[top_local]

    cand_idx = np.asarray(cand_idx, dtype=int)
    cand_idx = cand_idx[(cand_idx >= 0) & (cand_idx < len(df))]
    if cand_idx.size == 0: return pd.DataFrame()

    # --- Scoring (FAISS-only profile: always encode candidates)
    results = df.iloc[cand_idx].copy()
    texts = (results["overview"].fillna("").astype(str).tolist()
             if "overview" in results.columns else
             results["title"].astype(str).tolist())
    cand_vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    q = (q_np[0] / (np.linalg.norm(q_np[0]) + 1e-12)).astype("float32")
    E = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
    sims = (E @ q).astype("float32")

    # Popularity blend
    if use_popularity and "popularity" in results.columns:
        pop_vals = pd.to_numeric(results["popularity"], errors="coerce").fillna(0).to_numpy("float32")
        scores = (sims * np.log1p(pop_vals)).astype("float32")
    else:
        scores = sims.astype("float32")

    results = results.assign(score=scores, similarity=sims.astype("float32"))

    # Safe mode
    if safe_mode and not results.empty:
        if "nsfw" in results.columns:
            results = results[~results["nsfw"].astype(bool)]
        else:
            bad = ["nsfw","adult","porn","xxx","sex","erotic","babe"]
            pattern = "(?i)" + "|".join(bad)
            mask = pd.Series(False, index=results.index)
            for col in ["genres","title","overview"]:
                if col in results.columns:
                    mask |= results[col].astype(str).str.contains(pattern, na=False, regex=True)
            results = results.loc[~mask]
    if results.empty: return results

    # Franchise filter
    if franchise_only and not results.empty:
        if "franchise" not in results.columns: results["franchise"] = "Unknown"
        detected = (exact.iloc[0]["franchise"].strip()
                    if (not exact.empty and str(exact.iloc[0].get("franchise","")).strip())
                    else q_franchise)
        if not detected or detected == "Unknown":
            detected = detect_franchise(q_text, overview)
        missing_mask = results["franchise"].fillna("Unknown").isin(["", "Unknown"])
        if missing_mask.any():
            results.loc[missing_mask, "franchise"] = results.loc[missing_mask].apply(
                lambda row: detect_franchise(str(row.get("title","")), str(row.get("overview",""))), axis=1
            )
        results = results[results["franchise"] == detected]
    if results.empty: return results

    # Genre overlap (soft)
    if use_genres and not results.empty:
        input_genres = q_genres if isinstance(q_genres, list) else _split_genres(str(q_genres))
        input_genres = set([g for g in input_genres if g and g != "Unknown"])
        if input_genres:
            def genre_overlap(val):
                if isinstance(val, list): working = {g for g in val if g != "Unknown"}
                else: working = set(_split_genres(str(val)))
                return len(input_genres & working)
            results = results.assign(
                genre_overlap=results["genres"].apply(genre_overlap) if "genres" in results.columns else 0
            )
            min_req = 2 if len(input_genres) >= 2 else 1
            filtered = results[results["genre_overlap"] >= min_req]
            if len(filtered) < top_k and results["genre_overlap"].max() >= 1:
                filtered = results[results["genre_overlap"] >= 1]
            results = filtered
            if not results.empty:
                results = results.assign(score=results["score"] * (1.0 + 0.05 * results.get("genre_overlap", 0)))
    if results.empty: return results

    # Exclude exact self
    if "title" in results.columns:
        results = results[results["title"].str.lower() != q_text.lower()]
    if results.empty: return results

    results = results.sort_values(by="score", ascending=False)
    keep = ["title","genres","franchise","popularity","overview","score","similarity","poster_url","poster_path","id"]
    exist = [c for c in keep if c in results.columns]
    out = results.head(top_k).copy()

    # Poster
    if "poster_url" not in out.columns:
        if "poster_path" in out.columns:
            out["poster_url"] = out["poster_path"].astype(str).str.strip()
            m = out["poster_url"].str.len() > 0
            out.loc[m, "poster_url"] = TMDB_IMG_BASE + out.loc[m, "poster_url"]
        else:
            out["poster_url"] = ""
    return out[exist + (["poster_url"] if "poster_url" in out.columns and "poster_url" not in exist else [])].reset_index(drop=True)

# ---------------- UI ----------------------------------------------------------
st.title("🎬 Mario's Netflix (FAISS-only, Cloud Mode)")

with st.expander("Technical info"):
    st.write(f"Model: `{MODEL_NAME}`")
    st.write(f"Parquet: `{PARQUET_FILE}`")
    st.write(f"FAISS index: `{INDEX_FILE}`  (FAISS available: `{_HAVE_FAISS}`)")
    st.write(f"Embeddings: `{EMB_FILE or '(disabled)'}`")
    st.write(f"HF repo: `{HF_REPO_ID}`  type=`{HF_REPO_TYPE}`  rev=`{HF_REVISION}`")
    st.write({"CLOUD_MODE": CLOUD_MODE, "CLOUD_FANOUT_MAX": CLOUD_FANOUT_MAX, "CLOUD_FALLBACK_POOL": CLOUD_FALLBACK_POOL})

st.subheader("🔍 Search")
query = st.text_input("Title or description", placeholder="e.g., Interstellar — gritty space survival")

st.divider()
st.subheader("⚙️ Filters")
col1, col2, col3, col4 = st.columns(4)
with col1: franchise_only = st.checkbox("Filter by franchise", value=False)
with col2: use_genres = st.checkbox("Use genres of query movie", value=False)
with col3: safe_mode = st.checkbox("Safe mode (hide NSFW)", value=True)
with col4: use_popularity = st.checkbox("Use popularity boost", value=True)

top_k = st.slider("How many recommendations?", min_value=5, max_value=50, value=10, step=5)

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
                emb_name=EMB_FILE,     # empty => embeddings disabled (FAISS-only)
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
                c1, c2 = st.columns([1, 5])
                with c1:
                    st.image(poster_url, width=110)
                with c2:
                    st.markdown(f"### {row.get('title','Untitled')}")
                    if row.get("genres"):
                        st.caption(
                            f"Genres: {', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']}"
                        )
                    if row.get("franchise"): st.caption(f"🎬 Franchise: {row['franchise']}")
                    if row.get("popularity") is not None:
                        try: st.caption(f"⭐ Popularity: {float(row['popularity']):.2f}")
                        except Exception: pass
                    if row.get("overview"): st.write(row["overview"])
                st.divider()

            with st.expander("See raw table"):
                st.dataframe(out, use_container_width=True)
