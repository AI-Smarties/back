from unittest.mock import patch

import pytest

from gemini_tools import fetch_information


class TestFetchInformation:
    """Test cases for fetch_information function."""

    @pytest.mark.asyncio
    async def test_fetch_with_empty_query(self):
        """Empty query should return not_relevant."""
        result = await fetch_information(
            thinking_context="Need prior context",
            query="",
            transcript="Budget discussion transcript",
            query_history=[],
        )
        assert result["status"] == "not_relevant"
        assert "thinking" in result

    @pytest.mark.asyncio
    async def test_fetch_with_no_results(self):
        """No vector matches should return not_relevant."""
        with patch("gemini_tools.search_vectors", return_value=[]):
            result = await fetch_information(
                thinking_context="Need prior budget context",
                query="budget",
                transcript="We are discussing budget now",
                query_history=[],
            )
            assert result["status"] == "not_relevant"
            assert "thinking" in result

    @pytest.mark.asyncio
    async def test_fetch_with_search_error(self):
        """Search failure should return error."""
        with patch(
            "gemini_tools.search_vectors",
            side_effect=Exception("DB error"),
        ):
            result = await fetch_information(
                thinking_context="Need prior budget context",
                query="budget",
                transcript="We are discussing budget now",
                query_history=[],
            )
            assert result["status"] == "error"
            assert "Failed to fetch information" in result["error_message"]

    @pytest.mark.asyncio
    async def test_fetch_with_valid_query(self):
        """Valid search result should be evaluated and returned."""
        mock_result = object()

        async def mock_evaluate(
            transcript,
            vector_database_response,
            thinking_context,
            query_history,
        ):
            assert transcript == "Current transcript"
            assert vector_database_response == [mock_result]
            assert thinking_context == "Need earlier project info"
            assert query_history == []
            return {
                "status": "found",
                "information": "Found information",
                "score": 0.95,
                "thinking": "Relevant prior context found",
            }

        with patch("gemini_tools.search_vectors", return_value=[mock_result]):
            with patch("gemini_tools.evaluate_db_data", side_effect=mock_evaluate):
                result = await fetch_information(
                    thinking_context="Need earlier project info",
                    query="test query",
                    transcript="Current transcript",
                    query_history=[],
                )

        assert result["status"] == "found"
        assert result["information"] == "Found information"
        assert result["score"] == 0.95
        assert result["thinking"] == "Relevant prior context found"
