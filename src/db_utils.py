from db import sessionlocal
from models import Conversation, Vector, Category

# pylint: disable=no-member

async def create_vector(text, conversation_id):
    with sessionlocal.begin() as session:
        pass

async def delete_vector(vector_id):
    with sessionlocal.begin() as session:
        vec = session.get(Vector, vector_id)
        if vec:
            session.delete(vec)

async def get_vector_by_id(vector_id):
    with sessionlocal() as session:
        pass

async def get_vectors_by_conversation_id(conversation_id):
    with sessionlocal() as session:
        pass

async def get_vectors():
    with sessionlocal() as session:
        pass

async def create_conversation(name, date, summary, category_id):
    with sessionlocal.begin() as session:
        pass

async def delete_conversation(conversation_id):
    with sessionlocal.begin() as session:
        conv = session.get(Conversation, conversation_id)
        if conv:
            session.delete(conv)

async def get_conversation_by_id(conversation_id):
    with sessionlocal() as session:
        pass

async def get_conversations_by_category_id(category_id):
    with sessionlocal() as session:
        pass

async def get_conversations_by_category_name(category_name):
    with sessionlocal() as session:
        pass

async def get_conversations():
    with sessionlocal() as session:
        pass

async def create_category(name):
    with sessionlocal.begin() as session:
        pass

async def delete_category(category_id):
    with sessionlocal.begin() as session:
        cat = session.get(Category, category_id)
        if cat:
            session.delete(cat)

async def get_category_by_id(category_id):
    with sessionlocal() as session:
        pass

async def get_category_by_name(name):
    with sessionlocal() as session:
        pass

async def get_categories():
    with sessionlocal() as session:
        pass
