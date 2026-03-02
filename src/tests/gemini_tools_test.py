from unittest.mock import patch, MagicMock
from gemini_tools import save_information, fetch_information


class TestSaveInformation:
    """Test cases for save_information function"""

    def test_save_with_valid_information(self):
        """Test saving valid information"""
        with patch('gemini_tools.create_vector') as mock_create:
            result = save_information("Test info", 1)
            assert result["status"] == "success"
            assert "Test info" in result["message"]
            mock_create.assert_called_once_with("Test info", 1)

    def test_save_with_empty_information(self):
        """Test that empty information returns error"""
        result = save_information("", 1)
        assert result["status"] == "error"
        assert "cannot be empty" in result["message"]

    def test_save_with_database_error(self):
        """Test handling of database errors"""
        with patch('gemini_tools.create_vector', side_effect=Exception("DB error")):
            result = save_information("Test info", 1)
            assert result["status"] == "error"
            assert "Failed to save" in result["message"]


class TestFetchInformation:
    """Test cases for fetch_information function"""

    def test_fetch_with_valid_query(self):
        """Test fetching information with valid query"""
        mock_result = MagicMock()
        mock_result.text = "Found information"
        with patch('gemini_tools.search_vectors', return_value=[mock_result]):
            result = fetch_information("test query")
            assert result["status"] == "success"
            assert result["information"] == "Found information"

    def test_fetch_with_empty_query(self):
        """Test that empty query returns error"""
        result = fetch_information("",)
        assert result["status"] == "error"
        assert "cannot be empty" in result["message"]

    def test_fetch_with_no_results(self):
        """Test fetching when no information is found"""
        with patch('gemini_tools.search_vectors', return_value=[]):
            result = fetch_information("test query")
            assert result["status"] == "success"
            assert "No relevant information" in result["information"]
