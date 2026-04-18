from unittest.mock import patch, AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from gemini_tools import fetch_information, fetch_general_knowledge
import db_utils
import db


@pytest.fixture
def isolated_db():
    """Drop and recreate tables before the test, then drop again after."""
    try:
        db.drop_tables()
        db.create_tables()
    except OperationalError:
        pytest.skip("Database not available")
    yield
    db.drop_tables()
    db.create_tables()


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

    @pytest.mark.asyncio
    async def test_fetch_information_user_isolation(self, monkeypatch, isolated_db):  # noqa: ARG002

        class _StubEmbeddingModel:  # pylint: disable=too-few-public-methods,unused-argument
            def get_embeddings(self, texts, output_dimensionality):
                embedding = [0.0] * output_dimensionality
                embedding[0] = 1.0
                return [type("obj", (), {"values": embedding})()]

        monkeypatch.setattr(db_utils, "EMBEDDING_MODEL", _StubEmbeddingModel())

        # user A
        conv_a = db_utils.create_conversation("convA", user_id="user-A")
        db_utils.create_vector("secret A", conv_a.id)

        # user B
        conv_b = db_utils.create_conversation("convB", user_id="user-B")
        db_utils.create_vector("secret B", conv_b.id)

        with patch("gemini_tools.evaluate_db_data") as mock_eval:
            mock_eval.side_effect = lambda transcript, vectors, *_: {
                "status": "found",
                "information": vectors[0].text,
                "score": 1.0,
                "thinking": "",
            }

            res_a = await fetch_information(
                thinking_context="test",
                query="secret",
                transcript="",
                query_history=[],
                user_id="user-A",
            )

            res_b = await fetch_information(
                thinking_context="test",
                query="secret",
                transcript="",
                query_history=[],
                user_id="user-B",
            )

        assert "secret A" in res_a["information"]
        assert "secret B" in res_b["information"]


class TestFetchGeneralKnowledge:
    """Test cases for fetch_general_knowledge function."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_not_relevant(self):
        result = await fetch_general_knowledge(
            thinking_context="user asked something",
            query="",
            transcript="some transcript",
        )
        assert result["status"] == "not_relevant"

    @pytest.mark.asyncio
    async def test_web_search_failure_returns_error(self):
        with patch("gemini_tools.get_search_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.aio.models.generate_content.side_effect = Exception("network error")
            mock_get.return_value = mock_client

            result = await fetch_general_knowledge(
                thinking_context="user asked what HTTP is",
                query="what is HTTP",
                transcript="I don't know what HTTP is",
            )
        assert result["status"] == "error"
        assert "Web search failed" in result["error_message"]

    @pytest.mark.asyncio
    async def test_empty_web_result_returns_not_relevant(self):
        with patch("gemini_tools.get_search_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.text = ""
            mock_client.aio.models.generate_content.return_value = mock_response
            mock_get.return_value = mock_client

            result = await fetch_general_knowledge(
                thinking_context="user asked something",
                query="what is X",
                transcript="I don't know what X is",
            )
        assert result["status"] == "not_relevant"

    @pytest.mark.asyncio
    async def test_web_result_forwarded_to_evaluator(self):
        with patch("gemini_tools.get_search_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.text = "HTTP stands for HyperText Transfer Protocol."
            mock_client.aio.models.generate_content.return_value = mock_response
            mock_get.return_value = mock_client

            async def mock_evaluate(transcript, web_text, thinking_context, query_history):
                assert web_text == "HTTP stands for HyperText Transfer Protocol."
                return {"status": "found", "information": "HTTP on HyperText Transfer Protocol.", "score": 0.9, "thinking": "relevant"}

            with patch("gemini_tools.evaluate_web_data", side_effect=mock_evaluate):
                result = await fetch_general_knowledge(
                    thinking_context="user asked what HTTP is",
                    query="what is HTTP",
                    transcript="I don't know what HTTP is",
                )

        assert result["status"] == "found"
        assert "HTTP" in result["information"]
