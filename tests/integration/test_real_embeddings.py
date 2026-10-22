"""Optional checks with the real MiniLM model (downloads ~90 MB once). Run with: pytest -m slow"""

import pytest

from src.ingestion.indexer import ProtocolIndexer, build_vector_store
from src.ingestion.parser import load_document
from src.ingestion.splitter import split_protocol
from src.rag.chain import EligibilityRAGChain
from src.rag.embeddings import get_embeddings
from src.schemas import EligibilityQueryRequest, EligibilityStatus
from tests.conftest import NSCLC_ID, NSCLC_PROTOCOL, T2D_ID, T2D_PROTOCOL

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_chain(tmp_path_factory):
    from src.config import Settings

    settings = Settings(_env_file=None, CHROMA_PERSIST_DIR=str(tmp_path_factory.mktemp("chroma")), OPENAI_API_KEY=None)
    store = build_vector_store(get_embeddings(), settings.CHROMA_PERSIST_DIR, "slow")
    indexer = ProtocolIndexer(store)
    for path, protocol_id in ((T2D_PROTOCOL, T2D_ID), (NSCLC_PROTOCOL, NSCLC_ID)):
        indexer.index_chunks(split_protocol(load_document(path), protocol_id, path.name), protocol_id)
    return EligibilityRAGChain(store, indexer, None, settings)


def test_kidney_question_retrieves_egfr_exclusion(real_chain):
    hits = real_chain.retrieve("Are there kidney function requirements?", T2D_ID, 3)
    assert any("glomerular filtration" in doc.page_content for doc, _ in hits)
    assert all(doc.metadata["protocol_id"] == T2D_ID for doc, _ in hits)


def test_off_topic_question_abstains_without_llm(real_chain):
    response = real_chain.query(EligibilityQueryRequest(query="Give me a cookie recipe.", protocol_id=NSCLC_ID))
    assert response.status is EligibilityStatus.CRITERIA_NOT_SPECIFIED
    assert response.audit.llm_called is False
