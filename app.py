"""ASGI entrypoint: `uvicorn app:app --host 0.0.0.0 --port 8000`."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src import __version__
from src.api.routes import health_router, router
from src.config import get_settings
from src.ingestion.parser import DocumentParseError, EmptyDocumentError, UnsupportedFileTypeError
from src.rag.chain import LLMInvocationError, LLMNotConfiguredError, NoCriteriaError, ProtocolNotFoundError

logger = logging.getLogger("clinical_trial_rag")

_ERROR_STATUS = {
    UnsupportedFileTypeError: 415,
    EmptyDocumentError: 422,
    DocumentParseError: 422,
    ProtocolNotFoundError: 404,
    NoCriteriaError: 422,
    LLMNotConfiguredError: 503,
    LLMInvocationError: 502,
}
_ERROR_DETAIL = {
    ProtocolNotFoundError: "Protocol {0} is not indexed.",
    NoCriteriaError: "Protocol {0} has no detected inclusion/exclusion criteria.",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings.ensure_directories()
    logger.info("Starting clinical-trial-rag v%s (llm=%s)", __version__, settings.LLM_MODEL_NAME)
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="clinical-trial-rag",
        version=__version__,
        description="Grounded RAG for clinical trial eligibility. Decision support only; not for clinical use.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d latency_ms=%.0f",
            request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000,
        )
        return response

    def _handle(request: Request, exc: Exception) -> JSONResponse:
        exc_type = type(exc)
        template = _ERROR_DETAIL.get(exc_type)
        detail = template.format(exc) if template else str(exc)
        if exc_type is LLMInvocationError:
            logger.error("LLM failure on %s: %s", request.url.path, exc, exc_info=exc.__cause__)
        return JSONResponse(status_code=_ERROR_STATUS[exc_type], content={"detail": detail})

    for exc_class in _ERROR_STATUS:
        application.add_exception_handler(exc_class, _handle)

    application.include_router(health_router)
    application.include_router(router)
    return application


app = create_app()
