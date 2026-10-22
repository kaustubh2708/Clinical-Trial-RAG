"""Load protocol files (.pdf, .txt) into LangChain Documents."""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class UnsupportedFileTypeError(ValueError):
    """Raised for file types the parser cannot load."""


class EmptyDocumentError(ValueError):
    """Raised when a file yields no extractable text (e.g. a scanned PDF without OCR)."""


def load_document(file_path: str | Path) -> List[Document]:
    """Load a protocol file. PDFs yield one Document per page; text files yield one Document."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)

    documents = [doc for doc in loader.load() if doc.page_content.strip()]
    if not documents:
        raise EmptyDocumentError(f"No extractable text found in '{path.name}'. Scanned PDFs need OCR first.")

    logger.info("Loaded %d page(s) from %s", len(documents), path.name)
    return documents
