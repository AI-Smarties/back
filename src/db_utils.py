from sqlalchemy import select
from db import sessionlocal
from models import Conversation, Vector, Category

# pylint: disable=no-member

async def create_vector(text, conv_id):
    with sessionlocal.begin() as session:
        pass

async def delete_vector(vec_id):
    with sessionlocal.begin() as session:
        vec = session.get(Vector, vec_id)
        if vec:
            session.delete(vec)

async def get_vector_by_id(vec_id):
    with sessionlocal() as session:
        pass

async def get_vectors_by_conversation_id(conv_id):
    with sessionlocal() as session:
        pass

async def get_vectors():
    with sessionlocal() as session:
        pass

async def create_conversation(name, timestamp, summary, cat_id):
    with sessionlocal.begin() as session:
        pass

async def delete_conversation(conv_id):
    with sessionlocal.begin() as session:
        conv = session.get(Conversation, conv_id)
        if conv:
            session.delete(conv)

async def get_conversation_by_id(conv_id):
    with sessionlocal() as session:
        pass

async def get_conversations_by_category_id(cat_id):
    with sessionlocal() as session:
        pass

async def get_conversations_by_category_name(cat_name):
    with sessionlocal() as session:
        pass

async def get_conversations():
    with sessionlocal() as session:
        pass

async def create_category(cat_name):
    with sessionlocal.begin() as session:
        pass

async def delete_category_by_id(cat_id):
    with sessionlocal.begin() as session:
        cat = session.get(Category, cat_id)
        if cat:
            session.delete(cat)

async def delete_category_by_name(cat_name):
    with sessionlocal.begin() as session:
        cat = session.execute(
            select(Category).where(Category.name == cat_name)
        ).scalar_one_or_none()
        if cat:
            session.delete(cat)

async def get_category_by_id(cat_id):
    with sessionlocal() as session:
        pass

async def get_category_by_name(cat_name):
    with sessionlocal() as session:
        pass

async def get_categories():
    with sessionlocal() as session:
        pass
