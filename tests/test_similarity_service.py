"""Unit tests for the similarity_service."""
import sys
from unittest.mock import MagicMock, patch
import pytest


def test_empty_inputs_return_empty():
    from app.services.similarity_service import get_similarity_scores
    assert get_similarity_scores("", []) == []
    assert get_similarity_scores("some jd", []) == []


def test_similarity_service_importable():
    """Ensure the module imports and functions are callable."""
    import app.services.similarity_service as sim_svc
    assert callable(sim_svc.get_similarity_scores)


def test_similarity_scores_with_mocked_st():
    """Test get_similarity_scores without loading heavy neural network weights."""
    from app.services import similarity_service

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock()

    mock_util = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.tolist.return_value = [0.85, 0.42]
    mock_util.cos_sim.return_value = [mock_tensor]

    with patch.object(similarity_service, "_get_st_model", return_value=mock_model), \
         patch.dict(sys.modules, {"sentence_transformers.util": mock_util, "sentence_transformers": MagicMock(util=mock_util)}):
        scores = similarity_service.get_similarity_scores("Job description", ["Resume 1", "Resume 2"])
        assert scores == [0.85, 0.42]
