from unittest.mock import MagicMock, patch
from ..app.services.extract import extract_action_items, extract_action_items_llm


def test_extract_bullets_and_checkboxes():
    text = """
    Notes from meeting:
    - [ ] Set up database
    * implement API extract endpoint
    1. Write tests
    Some narrative sentence.
    """.strip()

    items = extract_action_items(text)
    assert "Set up database" in items
    assert "implement API extract endpoint" in items
    assert "Write tests" in items


@patch("week2.app.services.extract.chat")
def test_extract_action_items_llm_success(mock_chat):
    # Mock Ollama response
    mock_response = MagicMock()
    mock_response.message.content = '{"action_items": ["Implement authentication", "Add logging"]}'
    mock_chat.return_value = mock_response

    text = "Meeting notes: We need to implement authentication and add logging."
    items = extract_action_items_llm(text)

    assert items == ["Implement authentication", "Add logging"]
    mock_chat.assert_called_once()


@patch("week2.app.services.extract.chat")
def test_extract_action_items_llm_failure(mock_chat):
    # Mock Ollama failure
    mock_chat.side_effect = Exception("Connection error")

    text = "Meeting notes: Some items."
    items = extract_action_items_llm(text)

    assert items == []
