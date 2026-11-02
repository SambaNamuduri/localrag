""" Unit tests for document chunking and validation. """

import pytest
from services.document_service import DocumentService

def test_chunk_text_returns_chunks():
    """ Handle the test chunk text returns chunks operation. """
    text = "word " * 1000
    chunks = DocumentService.chunk(text, size=200, overlap=30)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)

def test_chunk_rejects_invalid_overlap():
    """ Handle the test chunk rejects invalid overlap operation. """
    with pytest.raises(Exception):
        DocumentService.chunk("hello world", size=100, overlap=100)

def test_chunk_rejects_empty_text():
    """ Handle the test chunk rejects empty text operation. """
    with pytest.raises(Exception):
        DocumentService.chunk("   ")
