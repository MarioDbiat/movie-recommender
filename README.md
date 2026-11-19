
# AI Movie Recommender

### Semantic Movie Discovery with Sentence-BERT, Triplet Loss, and FAISS

A content-based movie recommendation system designed to help users discover films beyond mainstream recommendations, including lesser-known and independent movies.

Instead of relying solely on genre labels, ratings, or popularity, the system uses transformer-based text embeddings to match movies based on their descriptions, themes, and semantic similarity.

The project combines a fine-tuned Sentence-BERT model, FAISS vector search, and configurable recommendation filters. It was developed as an independent machine-learning research and software engineering project.

## Key Highlights

- **798,029 movies** in the processed dataset
- **384-dimensional embeddings** generated using Sentence-BERT
- **Triplet Loss fine-tuning** to improve semantic ranking
- **FAISS IndexFlatIP** for similarity-based retrieval
- **Popularity-aware reranking** with optional genre and franchise filtering
- **Streamlit interface** for interactive movie discovery
- **FastAPI backend** maintained in a separate repository
- Research documentation covering model development and evaluation

## How It Works

The recommendation pipeline consists of five stages.

### 1. Data Preparation

Movie records are processed into a structured dataset containing information such as titles, overviews, genres, and popularity.

The processed dataset contains **798,029 movies** and is stored in Parquet format for efficient loading.

### 2. Sentence-BERT Embeddings

The system uses `all-MiniLM-L6-v2`, a Sentence-Transformers model that represents text as 384-dimensional vectors.

Movie descriptions are embedded into a shared semantic space, allowing the recommender to retrieve films based on meaning rather than exact keyword matches.

### 3. Model Fine-Tuning

The embedding model was fine-tuned using a triplet-based training approach.

Each triplet contains:

- **Anchor:** A reference movie description
- **Positive:** A semantically related description
- **Negative:** A less-related description

Triplet Loss encourages the model to position related descriptions closer together while separating less-related examples.

### 4. Vector Search with FAISS

Movie embeddings are stored in a FAISS `IndexFlatIP` index.

The retrieval pipeline uses inner-product similarity. For normalized vectors, inner product is equivalent to cosine similarity.

Unlike approximate nearest-neighbor indexes such as IVF or HNSW, `IndexFlatIP` performs **exact vector search**.

### 5. Reranking and Filtering

The system retrieves candidate movies and computes semantic similarity scores. It then optionally applies popularity-based reranking:

`score = similarity × log(1 + popularity)`

Additional options in the Streamlit application include:

- Genre-overlap filtering
- Heuristic franchise detection and filtering
- Keyword-based adult-content filtering
- Adjustable recommendation count
- Search using a movie title or a natural-language description

The safety filter is heuristic and should not be treated as a comprehensive content-classification system.

## Example Queries

Users can search for a specific movie or describe the kind of story they want to watch.

**Movie-title search**

`Interstellar`

For an exact title match, the Streamlit application uses the matched movie's overview as the semantic query and excludes that title from the returned recommendations.

**Natural-language search**

`A slow, emotional science-fiction story about isolation and discovery`

The system converts the query into an embedding and retrieves movies with semantically related descriptions.

This approach can surface relevant films even when they do not share an exact title, keyword, or genre label.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embedding model | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Deep learning | PyTorch |
| Model training | Sentence-Transformers, Triplet Loss |
| Vector retrieval | FAISS |
| Data processing | Pandas, NumPy, PyArrow |
| Interactive interface | Streamlit |
| Separate backend | FastAPI |
| Model and dataset hosting | Hugging Face |
| Research environment | Jupyter Notebook |

## Project Structure

```text
movie-recommender/
├── app/
│   └── app.py
├── notebooks/
│   └── RecommendationSystem.ipynb
├── src/
│   ├── CreatingTriplets.py
│   ├── Embeddings.py
│   └── FIASS.py
├── docs/
│   ├── English_Reaserch_Paper.pdf
│   └── MovieRecommendationResearchPaper.pdf
├── requirements.txt
├── runtime.txt
└── README.md
```

The notebook documents the experimental model-development workflow, including embedding generation, triplet construction, training, and evaluation.

The `app/` directory contains the interactive Streamlit recommender.

The `src/` directory includes supporting research scripts. Some scripts retain environment-specific paths and are not yet packaged into a fully reproducible training pipeline.

Research PDFs are tracked using Git LFS.

## Running the Streamlit Application

### Prerequisites

- Git
- Python 3.10
- An environment compatible with the pinned dependencies
- Internet access for initial model and dataset downloads

The current dependency configuration includes a CPU-specific PyTorch wheel and has not been validated across all operating systems, including Apple Silicon macOS.

### Installation

Clone the repository:

```bash
git clone https://github.com/MarioDbiat/movie-recommender.git
cd movie-recommender
```

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app/app.py
```

The application is designed to download the required model and data artifacts from Hugging Face on initial use.

**Note:** The installation commands document the intended setup, not a currently verified clean installation. Dependency compatibility and reproducibility are ongoing improvement areas.

### Hugging Face Resources

**Fine-tuned model**

https://huggingface.co/Mariodb/movie-recommender-model

**Dataset and retrieval artifacts**

https://huggingface.co/datasets/Mariodb/movie-recommender-dataset

Large dataset, embedding, and index artifacts are stored externally rather than committed directly to Git.

## Evaluation

The research notebook includes experiments measuring the model's ability to distinguish semantically related movie descriptions and evaluating retrieval and reranking behavior.

Recorded experimental results include:

| Experiment | Recorded result |
|---|---:|
| Triplet-ranking accuracy | 99.94% |
| Pure retrieval P@10 | 0.9828 |
| Popularity-weighted retrieval P@10 | 0.9666 |

**Important evaluation limitation:** These results use synthetically constructed examples and relevance criteria derived from the underlying retrieval setup. They should not be interpreted as independently verified recommendation accuracy, real-user satisfaction, or performance on a human-labeled relevance benchmark.

A stronger evaluation would include human relevance judgments, representative query sets, popularity and diversity analysis, and comparisons against simpler retrieval baselines.

No production latency benchmark is claimed here.

## API and Web Frontend

A separate repository contains an experimental FastAPI service and browser frontend:

**[Movie Recommender API](https://github.com/MarioDbiat/movie-recommender-api)**

The backend exposes recommendation and health-check endpoints.

It also contains optional query-rewriting functionality for descriptive searches, including a rule-based fallback and an LLM-assisted implementation that is disabled by default.

The API and frontend are being improved independently of the Streamlit application. A publicly accessible, operational deployment is not currently guaranteed.

## Research and Development

This project began as an independent investigation into whether semantic embeddings could improve movie discovery beyond conventional genre- and popularity-based approaches.

The development process included:

1. Preparing and cleaning a large movie dataset
2. Generating sentence embeddings for movie descriptions
3. Constructing synthetic triplets for contrastive-style training
4. Fine-tuning the Sentence-BERT encoder
5. Building a FAISS retrieval index
6. Implementing reranking and filtering strategies
7. Developing interactive recommendation interfaces
8. Evaluating model behavior and documenting limitations

The repository includes research reports in the `docs/` directory and an experimental Jupyter notebook.

## Limitations and Future Work

The current implementation provides a research-oriented recommendation pipeline rather than a fully validated production service.

Areas for improvement include:

- Reproducible data preparation and training scripts
- Cross-platform dependency management
- Automated unit and integration tests
- Independent human-labeled retrieval evaluation
- Diversity and novelty metrics for hidden-gem discovery
- Improved franchise detection and content moderation
- API/frontend feature consistency
- Deployment reliability and performance benchmarking

## Author

**Mario Dbiat**

Computer Science — University of California, San Diego

Independent project focused on natural language processing, semantic search, recommendation systems, and applied machine learning.

[GitHub Profile](https://github.com/MarioDbiat)
