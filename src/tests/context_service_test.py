from context_service import build_context

def test_build_context():
    calendar_context_1 = {
        "title": "Business meeting",
        "description": "Discussing Q3 finances",
        "start": '2026-03-26T09:00:00.000+0200',
        "end": '2026-03-26T10:00:00.000+0200'
    }

    calendar_context_2 = {
        "title": None,
        "description": None,
        "start": None,
        "end": None
    }

    calendar_context_3 = {
        "title": "Budget meeting",
        "description": None,
        "start":  None,
        "end": None
    }

    assert build_context(calendar_context_1) == {
        "context_type": 'calendar_event',
        "title": 'Business meeting',
        "description": 'Discussing Q3 finances',
        "start": '2026-03-26T09:00:00.000+0200',
        "end": '2026-03-26T10:00:00.000+0200'
    }

    assert build_context(calendar_context_2) == {
        "context_type": 'general_conversation',
        "title": None,
        "description": None,
        "start": None,
        "end": None
    }

    assert build_context(calendar_context_3) == {
        "context_type": 'general_conversation',
        "title": 'Budget meeting',
        "description": None,
        "start": None,
        "end": None
    }
