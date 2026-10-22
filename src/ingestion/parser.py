"""Load protocol files (.pdf, .txt) into LangChain Documents."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
# Tried in order. latin-1 maps every byte, so decoding always succeeds as a last resort.
TEXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


class UnsupportedFileTypeError(ValueError):
    """Raised for file types the parser cannot load."""


class EmptyDocumentError(ValueError):
    """Raised when a file yields no extractable text (e.g. a scanned PDF without OCR)."""


class DocumentParseError(ValueError):
    """Raised when a file cannot be parsed (corrupt or encrypted PDF, unreadable text)."""


def _load_text(path: Path) -> list[Document]:
    for encoding in TEXT_ENCODINGS:
        try:
            documents = TextLoader(str(path), encoding=encoding).load()
        except RuntimeError as exc:  # TextLoader wraps UnicodeDecodeError in RuntimeError
            if not isinstance(exc.__cause__, UnicodeDecodeError):
                raise
            continue
        if encoding != TEXT_ENCODINGS[0]:
            logger.warning("%s is not UTF-8; decoded as %s", path.name, encoding)
        return documents
    raise DocumentParseError(f"Could not decode '{path.name}' as text.")


def load_document(file_path: str | Path) -> list[Document]:
    """Load a protocol file. PDFs yield one Document per page; text files yield one Document."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    try:
        loaded = PyPDFLoader(str(path)).load() if suffix == ".pdf" else _load_text(path)
    except DocumentParseError:
        raise
    except Exception as exc:  # pypdf raises many error types for corrupt or encrypted files
        raise DocumentParseError(f"Could not read '{path.name}': {type(exc).__name__}.") from exc

    documents = [doc for doc in loaded if doc.page_content.strip()]
    if not documents:
        raise EmptyDocumentError(f"No extractable text found in '{path.name}'. Scanned PDFs need OCR first.")

    logger.info("Loaded %d page(s) from %s", len(documents), path.name)
    return documents
