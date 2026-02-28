"""
Tool functions for Gemini Live API.
These functions can be called by Gemini as tool calls.
"""
from db_utils import search_vectors, create_vector


def save_information(information:str, conversation_id:int) -> dict:
    """
    Save information to vector database.

    Args:
        information: The information to be saved

    Returns:
        A dictionary with the status of the save operation
    """
    if not information or information == "":
        return {"status": "error", "message": "Information cannot be empty."}
    try:
        create_vector(information, conversation_id)
        return {"status": "success", "message": f"Information saved: {information}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save information: {e}"}


def fetch_information(query: str) -> dict:
    """
    Fetch useful information based on a text query from vector database.

    Args:
        query: The text query to search for information

    Returns:
        A dictionary with the retrieved information or error message
    """
    if not query or query == "":
        return {"status": "error", "message": "Query cannot be empty."}
    try:
        results = search_vectors(query, limit=1)
        if results:
            return {"status": "success", "information": results[0].text}
        else:
            return {"status": "success", "information": "No relevant information found."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to fetch information: {e}"}
