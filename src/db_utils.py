from db import Session
from models import Conversations, Vectors, Categories

async def create_conversation(name, date, summary, category_id):
    with Session.begin() as session:
        pass

async def delete_conversation(conversation_id):
    with Session.begin() as session:
        conv = session.get(Conversations, conversation_id)
        if conv:
            session.delete(conv)

async def get_conversation_by_id(conversation_id):
    with Session() as session:
        pass

async def get_conversations():
    with Session() as session:
        pass

async def create_vector(text, conversation_id):
    with Session.begin() as session:
        pass

async def delete_vector(vector_id):
    with Session.begin() as session:
        vec = session.get(Vectors, vector_id)
        if vec:
            session.delete(vec)

async def get_vector_by_id(vector_id):
    with Session() as session:
        pass

async def get_vectors_by_conversation_id(conversation_id):
    with Session() as session:
        pass

async def get_vectors():
    with Session() as session:
        pass

async def create_category(name):
    with Session.begin() as session:
        pass

async def delete_category(category_id):
    with Session.begin() as session:
        cat = session.get(Categories, category_id)
        if cat:
            session.delete(cat)

async def get_category_by_id(category_id):
    with Session() as session:
        pass

async def get_categories():
    with Session() as session:
        pass
