"""
Tests for backend/services/reranker.py (Phase 5).
Cross-encoder model is mocked — no real model download needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.services.reranker import rrf_merge, rerank_chunks


# ── rrf_merge ──────────────────────────────────────────────────────────────────

def test_rrf_merge_basic_ranking() -> None:
    """Chunk appearing in all 3 lists at rank 1 should score highest."""
    top_id = "child-aaa"
    low_id = "child-bbb"

    list1 = [
        {"metadata": {"child_id": top_id}},
        {"metadata": {"child_id": low_id}},
    ]
    list2 = [{"metadata": {"child_id": top_id}}]
    list3 = [{"metadata": {"child_id": top_id}}]

    result = rrf_merge([list1, list2, list3])

    assert result[0] == top_id
    assert low_id in result


def test_rrf_merge_deduplication() -> None:
    """Same child_id in multiple lists should appear only once in output."""
    shared_id = "child-shared"
    list1 = [{"metadata": {"child_id": shared_id}}]
    list2 = [{"metadata": {"child_id": shared_id}}]

    result = rrf_merge([list1, list2])

    assert result.count(shared_id) == 1


def test_rrf_merge_empty_lists_returns_empty() -> None:
    """All empty lists should return an empty list."""
    assert rrf_merge([[], [], []]) == []


def test_rrf_merge_missing_child_id_skipped() -> None:
    """Matches without child_id in metadata should be silently skipped."""
    result = rrf_merge([[{"metadata": {}}], [{"metadata": {"child_id": "valid-id"}}]])
    assert result == ["valid-id"]


def test_rrf_merge_lower_rank_scores_less() -> None:
    """A chunk at rank 1 must have a higher RRF score than one at rank 5."""
    id_rank1 = "chunk-rank1"
    id_rank5 = "chunk-rank5"

    single_list = [
        {"metadata": {"child_id": id_rank1}},
        {"metadata": {"child_id": "x"}},
        {"metadata": {"child_id": "y"}},
        {"metadata": {"child_id": "z"}},
        {"metadata": {"child_id": id_rank5}},
    ]

    result = rrf_merge([single_list])

    assert result.index(id_rank1) < result.index(id_rank5)


# ── rerank_chunks ──────────────────────────────────────────────────────────────

def test_rerank_chunks_returns_top_n() -> None:
    """rerank_chunks should return at most top_n chunks sorted by score."""
    chunks = [
        {"chunk_id": f"c{i}", "content": f"Content {i}"}
        for i in range(5)
    ]
    # Scores: chunk c2 highest, c4 second, rest lower
    mock_scores = [0.1, 0.3, 0.9, 0.2, 0.7]

    mock_model = MagicMock()
    mock_model.predict.return_value = mock_scores

    with patch("backend.services.reranker._get_cross_encoder", return_value=mock_model):
        result = rerank_chunks("diabetes query", chunks, top_n=2)

    assert len(result) == 2
    assert result[0]["chunk_id"] == "c2"  # highest score 0.9
    assert result[1]["chunk_id"] == "c4"  # second highest 0.7


def test_rerank_chunks_empty_input_returns_empty() -> None:
    """Empty chunk list should return empty list without calling the model."""
    mock_model = MagicMock()

    with patch("backend.services.reranker._get_cross_encoder", return_value=mock_model):
        result = rerank_chunks("query", [], top_n=5)

    assert result == []
    mock_model.predict.assert_not_called()


def test_rerank_chunks_fewer_than_top_n_returns_all() -> None:
    """If fewer chunks than top_n, all chunks are returned."""
    chunks = [
        {"chunk_id": "c1", "content": "Text A"},
        {"chunk_id": "c2", "content": "Text B"},
    ]
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.8, 0.5]

    with patch("backend.services.reranker._get_cross_encoder", return_value=mock_model):
        result = rerank_chunks("query", chunks, top_n=10)

    assert len(result) == 2
