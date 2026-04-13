# DocMind Backend (NEXUS API)

An AI-powered document research backend built with FastAPI. Upload PDFs, ask questions, get cited answers powered by hybrid retrieval and LLMs.

## Live Demo
- **Backend API**: https://docmind-backend-s3tu.onrender.com
- **Frontend**: https://docmind-frontend-iota.vercel.app
- **API Docs**: https://docmind-backend-s3tu.onrender.com/docs

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Framework | FastAPI + Uvicorn |
| Vector Store | Pinecone |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free API) |
| Retrieval | Hybrid BM25 + Semantic + RRF Fusion |
| Reranking | CrossEncoder `ms-marco-MiniLM-L-6-v2` (lazy loaded) |
| LLM | Groq (LLaMA 3.3 70B) / Gemini 2.0 Flash |
| PDF Parsing | PyMuPDF |
| Hosting | Render (free tier) |

## Architecture

```
PDF Upload
    ↓
PyMuPDF Text Extraction
    ↓
Semantic Chunking (180 words, 2-sentence overlap)
    ↓
HuggingFace Embeddings → Pinecone Vector Store
    ↓
Query Time:
  ├── Semantic Search (Pinecone + MMR)
  ├── BM25 Keyword Search
  └── RRF Fusion → LLM Answer (Groq/Gemini)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status and doc count |
| POST | `/upload` | Upload a PDF document |
| GET | `/documents` | List all indexed documents |
| DELETE | `/documents/{id}` | Delete a document |
| POST | `/query` | Query documents (streaming SSE) |
| POST | `/benchmark` | Run 10-question evaluation |

## Query Modes
- `fast` — Semantic + BM25 + RRF (default, recommended for free tier)
- `balanced` — Semantic + BM25 + RRF with more candidates
- `deep` — Full pipeline with cross-encoder reranking (requires more RAM)

## Environment Variables
Set these in your Render dashboard:

```
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=docmind-vectors
HUGGINGFACE_API_KEY=your_hf_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

## Local Development

```bash
# Clone the repo
git clone https://github.com/AdityaSingh-ist/docmind-backend.git
cd docmind-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your .env file with the keys above

# Run the server
uvicorn main:app --reload --port 10000
```

## Deployment
Deployed on Render with auto-deploy from GitHub main branch.

Start command:
```
uvicorn main:app --host 0.0.0.0 --port 10000
```
