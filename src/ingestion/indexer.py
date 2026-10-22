"""Persist protocol chunks into ChromaDB, idempotently per protocol."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.ingestion.parser import EmptyDocumentError, load_document
from src.ingestion.splitter import split_protocol
from src.schemas import ProtocolSummary, ProtocolUploadResponse

logger = logging.getLogger(__name__)


def build_vector_store(embeddings: Embeddings, persist_directory: str, collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )


class ProtocolIndexer:
    """Writes and reads protocol chunks. Every operation is scoped by protocol_id."""

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

    def _ids_for(self, protocol_id: str) -> list[str]:
        return self.vector_store.get(where={"protocol_id": protocol_id}, include=[])["ids"]

    def delete_protocol(self, protocol_id: str) -> int:
        ids = self._ids_for(protocol_id)
        if ids:
            self.vector_store.delete(ids=ids)
        logger.info("Deleted %d chunk(s) for %s", len(ids), protocol_id)
        return len(ids)

    def index_chunks(self, chunks: Sequence[Document], protocol_id: str) -> tuple[int, bool]:
        """Replace all chunks for `protocol_id` with `chunks`. Returns (chunks_indexed, replaced_existing)."""
        if any(chunk.metadata.get("protocol_id") != protocol_id for chunk in chunks):
            raise ValueError("Every chunk must carry the protocol_id it is indexed under.")
        replaced = self.delete_protocol(protocol_id) > 0
        if chunks:
            self.vector_store.add_documents(list(chunks), ids=[chunk.metadata["chunk_id"] for chunk in chunks])
        logger.info("Indexed %d chunk(s) for %s (replaced_existing=%s)", len(chunks), protocol_id, replaced)
        return len(chunks), replaced

    def get_protocol_chunks(self, protocol_id: str) -> list[Document]:
        """Every chunk of one protocol, in document order."""
        result = self.vector_store.get(where={"protocol_id": protocol_id}, include=["documents", "metadatas"])
        documents = [
            Document(id=chunk_id, page_content=text, metadata=metadata)
            for chunk_id, text, metadata in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
        ]
        return sorted(documents, key=lambda doc: doc.metadata.get("chunk_index", 0))

    def get_criteria(self, protocol_id: str) -> list[Document]:
        """Every inclusion/exclusion criterion of one protocol, in document order (no similarity search)."""
        result = self.vector_store.get(
            where={"$and": [{"protocol_id": protocol_id}, {"is_criterion": True}]},
            include=["documents", "metadatas"],
        )
        documents = [
            Document(id=chunk_id, page_content=text, metadata=metadata)
            for chunk_id, text, metadata in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
        ]
        return sorted(documents, key=lambda doc: doc.metadata.get("chunk_index", 0))

    def protocol_exists(self, protocol_id: str) -> bool:
        return bool(self.vector_store.get(where={"protocol_id": protocol_id}, limit=1, include=[])["ids"])

    def list_protocols(self) -> list[ProtocolSummary]:
        result = self.vector_store.get(include=["metadatas"])
        counts: dict[str, int] = defaultdict(int)
        sources: dict[str, set[str]] = defaultdict(set)
        for metadata in result["metadatas"]:
            protocol_id = metadata["protocol_id"]
            counts[protocol_id] += 1
            sources[protocol_id].add(metadata.get("source_document", "unknown"))
        return [
            ProtocolSummary(protocol_id=pid, source_documents=sorted(sources[pid]), chunk_count=counts[pid])
            for pid in sorted(counts)
        ]


def ingest_file(
    file_path: str | Path,
    protocol_id: str,
    source_filename: str,
    indexer: ProtocolIndexer,
    chunk_size: int,
    chunk_overlap: int,
) -> ProtocolUploadResponse:
    """Parse -> split -> index one protocol file."""
    documents = load_document(file_path)
    chunks = split_protocol(documents, protocol_id, source_filename, chunk_size, chunk_overlap)
    if not chunks:
        raise EmptyDocumentError(f"'{source_filename}' produced no chunks.")
    count, replaced = indexer.index_chunks(chunks, protocol_id)
    return ProtocolUploadResponse(
        message="Protocol re-indexed; previous version replaced." if replaced else "Protocol indexed successfully.",
        protocol_id=protocol_id,
        source_filename=source_filename,
        chunks_indexed=count,
        criteria_indexed=sum(1 for chunk in chunks if chunk.metadata["is_criterion"]),
        replaced_existing=replaced,
    )
