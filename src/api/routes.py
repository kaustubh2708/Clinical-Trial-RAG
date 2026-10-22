"""HTTP endpoints. Blocking work (embedding, Chroma, LLM) runs in `def` handlers, i.e. FastAPI's threadpool."""

import logging
import re
import uuid
from pathlib import Path as FilePath
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile, status
from fastapi.responses import JSONResponse

from src.api.dependencies import get_indexer, get_llm, get_rag_chain, get_vector_store
from src.config import Settings, get_settings
from src.ingestion.indexer import ProtocolIndexer, ingest_file
from src.ingestion.parser import SUPPORTED_EXTENSIONS
from src.rag.chain import EligibilityRAGChain
from src.schemas import (
    NCT_ID_PATTERN,
    EligibilityQueryRequest,
    EligibilityQueryResponse,
    PatientScreeningRequest,
    ProtocolSummary,
    ProtocolUploadResponse,
    ScreeningResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["protocols"])
health_router = APIRouter(tags=["health"])

_READ_CHUNK_BYTES = 1024 * 1024
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_filename(filename: str) -> str:
    """Drop any directory components and unsafe characters (blocks path traversal like '../../etc/passwd')."""
    name = FilePath(filename.replace("\\", "/")).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).lstrip(".")
    return name[:120] or "protocol"


def _save_upload(file: UploadFile, destination: FilePath, max_bytes: int) -> None:
    size = 0
    tmp_path = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.part")
    try:
        with tmp_path.open("wb") as out:
            while chunk := file.file.read(_READ_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {max_bytes // 2**20} MB.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Uploaded file is empty.")
        tmp_path.replace(destination)
    finally:
        tmp_path.unlink(missing_ok=True)


@health_router.get("/health")
def health() -> dict:
    """Liveness: the process is up. Never touches the model or the store."""
    return {"status": "healthy"}


@health_router.get("/ready")
def ready(settings: Settings = Depends(get_settings)) -> JSONResponse:
    """Readiness: the embedding model and vector store load."""
    try:
        get_vector_store(settings).get(limit=1, include=[])
    except Exception as exc:  # surface as 503 so orchestrators stop routing traffic here
        logger.exception("Readiness check failed")
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": type(exc).__name__})
    return JSONResponse(content={"status": "ready", "llm_configured": get_llm(settings) is not None})


@router.post("/protocols/index", response_model=ProtocolUploadResponse, status_code=status.HTTP_201_CREATED)
def index_protocol(
    file: UploadFile = File(..., description="Protocol document (.pdf or .txt)."),
    protocol_id: str = Form(..., pattern=NCT_ID_PATTERN),
    settings: Settings = Depends(get_settings),
    indexer: ProtocolIndexer = Depends(get_indexer),
) -> ProtocolUploadResponse:
    safe_name = sanitize_filename(file.filename or "")
    suffix = FilePath(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Only {sorted(SUPPORTED_EXTENSIONS)} files are accepted.")
    if suffix == ".pdf":
        header = file.file.read(5)
        file.file.seek(0)
        if header != b"%PDF-":
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File has a .pdf extension but is not a PDF.")

    upload_dir = FilePath(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{protocol_id}__{safe_name}"
    _save_upload(file, destination, settings.MAX_UPLOAD_MB * 2**20)

    return ingest_file(destination, protocol_id, safe_name, indexer, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)


@router.post("/protocols/query", response_model=EligibilityQueryResponse)
def query_protocol(
    request: EligibilityQueryRequest,
    chain: EligibilityRAGChain = Depends(get_rag_chain),
) -> EligibilityQueryResponse:
    return chain.query(request)


@router.post("/protocols/screen", response_model=ScreeningResponse)
def screen_patient(
    request: PatientScreeningRequest,
    chain: EligibilityRAGChain = Depends(get_rag_chain),
) -> ScreeningResponse:
    return chain.screen(request)


@router.get("/protocols", response_model=List[ProtocolSummary])
def list_protocols(indexer: ProtocolIndexer = Depends(get_indexer)) -> List[ProtocolSummary]:
    return indexer.list_protocols()


@router.delete("/protocols/{protocol_id}")
def delete_protocol(
    protocol_id: str = Path(..., pattern=NCT_ID_PATTERN),
    indexer: ProtocolIndexer = Depends(get_indexer),
) -> dict:
    deleted = indexer.delete_protocol(protocol_id)
    if deleted == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Protocol {protocol_id} is not indexed.")
    return {"protocol_id": protocol_id, "chunks_deleted": deleted}
