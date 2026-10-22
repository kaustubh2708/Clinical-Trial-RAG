# clinical-trial-rag — Onboarding Guide

Generated with [Understand-Anything](https://github.com/Egonex-AI/Understand-Anything) from `.ua/knowledge-graph.json` (commit `1821c73`). The graph has 242 nodes, 760 edges, 9 architectural layers and a 14-step tour. To explore it interactively, run `/understand-dashboard`. Rerun `/understand` after changes; later runs are incremental.

## Project overview

| | |
| --- | --- |
| What it is | A FastAPI microservice that answers eligibility questions about clinical trial protocols and pre-screens a patient summary against every inclusion/exclusion criterion. Answers are grounded in verbatim, verified protocol text |
| Languages | Python, Markdown, YAML, TOML, Dockerfile |
| Frameworks | FastAPI, Uvicorn, Pydantic v2, LangChain, ChromaDB, Pytest, Docker, GitHub Actions |
| LLM | OpenAI, or a local model through Ollama (e.g. `qwen2.5:7b`) |
| Size | 49 files: 36 code, 8 docs, 3 infra, 2 config. 159 functions and 32 classes |

**The one idea to hold on to:** the LLM only reads text and fills a typed form. Code makes every other decision:
- which trial's chunks are searched;
- whether to answer at all;
- whether each quote is real;
- the final eligible / not-eligible result.

## Architecture layers

Dependencies point downwards: API → RAG → Ingestion → Configuration & Schemas. Tests and evaluation sit on top and use everything.

| Layer | Files | Responsibility |
| --- | --- | --- |
| API | `app.py`, `src/api/routes.py`, `src/api/dependencies.py` | App factory, request-ID/latency middleware, exception → HTTP status mapping, `/api/v1` routes, `Depends()` providers for the store, LLM and chain |
| RAG Service | `src/rag/chain.py`, `hybrid.py`, `guardrails.py`, `prompt_templates.py`, `embeddings.py` | Retrieval (dense or BM25 + RRF), relevance gate, prompts, quote and evidence verification, Q&A and screening pipelines |
| Ingestion | `src/ingestion/parser.py`, `splitter.py`, `indexer.py` | PDF/TXT → criterion-level chunks → ChromaDB, replaced idempotently per protocol |
| Configuration & Schemas | `src/config.py`, `src/schemas.py`, `.env.example`, `pyproject.toml`, `requirements*.txt` | Settings, API and LLM output models, tool config, pinned dependencies. The two most-imported files live here |
| Test | `tests/**` (12 files) | Fixtures with fake embeddings and a fake LLM, unit tests per module, HTTP integration tests, slow tests with the real model |
| Evaluation & Tooling | `eval/*`, `scripts/fetch_ctgov.py` | 35-case golden set, dense vs hybrid comparison, gate sweep, optional end-to-end LLM eval, ClinicalTrials.gov downloader |
| Protocol Data | `data/**` | Two synthetic sample protocols; runtime folders for uploads and the Chroma index |
| Infrastructure & CI/CD | `Dockerfile` (2 stages), `.dockerignore`, `.github/workflows/ci.yml` | Non-root image with the embedding model baked in; CI runs lint, types, tests, the eval floor and a Docker build |
| Documentation | `README.md`, `docs/STUDY_BOOK.md` | Usage and results; the 18-chapter study guide |

## Key concepts

- **Criterion-level chunks.** `splitter.py` emits one chunk per inclusion/exclusion criterion, with metadata (`protocol_id`, `criterion_type`, `is_criterion`) and a `[NCT… | Section]` prefix for the embeddings.
- **Trial isolation.** Every query filters on `protocol_id` inside Chroma, and `chain.retrieve()` drops any chunk from another trial as a second line of defence.
- **Relevance gate.** If the best dense match is below `MIN_RELEVANCE_SCORE` (0.30), the service abstains without calling the LLM. In hybrid mode, a code or acronym found verbatim in the protocol (e.g. "MEN2") also passes.
- **Structured output plus verification.** The LLM fills length-bounded Pydantic schemas (`schemas.py`). `guardrails.verify_quote()` checks each quote word for word against the chunk it cites; failures downgrade the status.
- **Q&A can rule out, never rule in.** Top-k retrieval never sees every criterion, so `ELIGIBLE` from `/query` becomes `INDETERMINATE`. Only `/screen` can confirm eligibility.
- **Screening aggregation in code.** The model judges each criterion, which it sees under a short label like `I1` or `E3`. Code applies "all inclusions met AND no exclusion met" and fills in missing verdicts as `INSUFFICIENT_INFO`.
- **Dependency injection for tests.** `dependencies.py` is the only place clients are built, so `conftest.py` swaps in fakes through `app.dependency_overrides`.
- **Audit trail.** Every response records the model, provider, prompt version, retrieval mode, chunk IDs and a context hash.

## Guided tour

Read in this order; each step builds on the one before.

1. **Project overview**: `README.md`, `docs/STUDY_BOOK.md`. What the service does and the "LLM fills a form" principle.
2. **Configuration and dependencies**: `src/config.py`, `.env.example`, `requirements.txt`, `pyproject.toml`. Every setting the rest of the code reads.
3. **Typed contracts**: `src/schemas.py`. The request/response models and the bounded LLM output schemas.
4. **Application entry point**: `app.py`, `src/__init__.py`. Middleware, error mapping, router wiring.
5. **Routes and dependency injection**: `src/api/routes.py`, `src/api/dependencies.py`. Endpoints, upload safety (sanitised names, `%PDF` check, staged atomic save), provider selection.
6. **Ingestion, parse and split**: `src/ingestion/parser.py`, `splitter.py`. Encoding fallbacks, section detection, one chunk per criterion.
7. **Indexing**: `src/ingestion/indexer.py`, `src/rag/embeddings.py`. Delete-then-add per protocol, criteria lookup by metadata, cached MiniLM model.
8. **Hybrid retrieval**: `src/rag/hybrid.py`. Tokenizer, code-term detection, BM25, Reciprocal Rank Fusion.
9. **Prompts and guardrails**: `src/rag/prompt_templates.py`, `guardrails.py`. Delimited context, injection stripping, quote and evidence checks.
10. **The eligibility chain**: `src/rag/chain.py`. Where everything meets: retrieve → gate → LLM → verify → respond, plus screening.
11. **Sample data**: `data/sample_protocols/*`, `scripts/fetch_ctgov.py`. Two synthetic trials; real trials on demand.
12. **Testing with fakes**: `tests/conftest.py`, `test_chain.py`, `test_endpoints.py`, `test_real_embeddings.py`, `test_guardrails.py`.
13. **Evaluation**: `eval/run_eval.py`, `golden_set.jsonl`, `results.md`, `results_ollama_qwen2.5-7b.md`. Hit rate, MRR, gate sweep, LLM accuracy.
14. **Build and CI**: `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`.

## File map

| File | What it does |
| --- | --- |
| `app.py` | ASGI entry point: logging, request-ID/latency middleware, exception → status mapping, routers |
| `src/api/routes.py` | Health/readiness probes; index, query, screen, list and delete endpoints; upload hardening |
| `src/api/dependencies.py` | Builds and caches the Chroma store, the OpenAI or Ollama client, the indexer and the chain |
| `src/config.py` | `Settings` (pydantic-settings) and cached `get_settings()` |
| `src/schemas.py` | API models (queries, screening, citations, audit) and bounded LLM output schemas |
| `src/ingestion/parser.py` | PDF/TXT loading with encoding fallbacks and typed errors |
| `src/ingestion/splitter.py` | Section-aware, criterion-level chunking with metadata and context prefix |
| `src/ingestion/indexer.py` | Idempotent per-protocol writes; chunk, criteria and summary reads; `ingest_file()` |
| `src/rag/embeddings.py` | Cached local MiniLM embeddings, normalised, on CPU |
| `src/rag/hybrid.py` | Tokenizer, code/acronym terms, BM25, Reciprocal Rank Fusion |
| `src/rag/prompt_templates.py` | Versioned Q&A and screening prompts; context and criteria rendering with short labels |
| `src/rag/guardrails.py` | Tag stripping, verbatim quote verification, evidence relevance, log hashing |
| `src/rag/chain.py` | `EligibilityRAGChain`: retrieval, gate, structured LLM call, verification, screening aggregation |
| `tests/conftest.py` | Temp Chroma, fake embeddings, fake structured LLM, TestClient with overrides, PDF builder |
| `eval/run_eval.py` | Retrieval and end-to-end evaluation harness with a CI floor |
| `scripts/fetch_ctgov.py` | Downloads real eligibility criteria from the ClinicalTrials.gov API v2 |
| `Dockerfile` | Builder stage (venv, CPU torch, baked model) and non-root runtime stage |
| `.github/workflows/ci.yml` | Lint, type check, tests, eval floor, Docker build |

## Complexity hotspots

Approach these carefully. Changes here ripple furthest.

| File | Complexity | Why it needs care |
| --- | --- | --- |
| `src/rag/chain.py` | complex | Owns every safety decision: retrieval modes, gate, verification, downgrades, screening. Imported by 7 files. Change it only with tests in `tests/unit/test_chain.py` |
| `eval/run_eval.py` | complex | Builds its own chain and indexes the samples; CI fails if its hit rate drops below 0.95 |
| `tests/unit/test_chain.py` | complex | The behavioural spec of the chain; read it to learn what the chain promises |
| `src/schemas.py` | moderate, highest fan-in (10) | Every layer imports it; changing a field changes the API and the LLM's output schema |
| `src/config.py` | simple, fan-in 9 | Defaults here (e.g. the 0.30 gate) were set from evaluation results |
| `src/ingestion/splitter.py` | moderate, fan-in 7 | Chunk boundaries and metadata drive retrieval quality and screening; chunk IDs must stay stable |
| `src/rag/guardrails.py` | moderate | Loosening a check (quote normalisation, evidence relevance) directly weakens safety |

## First tasks for a new contributor

1. Run `pytest` (116 tests, offline, about 4 s) and `python -m eval.run_eval` to see the baseline.
2. Follow one request end to end: `routes.query_protocol` → `EligibilityRAGChain.query` → `retrieve` → `_invoke_structured` → `verify_quote`.
3. Add a small endpoint, such as `GET /api/v1/protocols/{id}/chunks`. It touches `routes.py` and `indexer.py`, plus a test in `tests/integration/test_endpoints.py`.
