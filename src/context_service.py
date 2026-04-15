"""
Takes calendar payload and return structured context for Gemini Live.
"""

def build_context(calendar_context):
    title = calendar_context.get('title')
    description = calendar_context.get('description')
    start = calendar_context.get('start')
    end = calendar_context.get('end')

    if start is None and end is None:
        context_type = 'general_conversation'

    else:
        context_type = 'calendar_event'

    return {
        "context_type": context_type,
        "title": title,
        "description": description,
        "start": start,
        "end": end
        }
