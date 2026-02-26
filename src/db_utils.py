from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import select
from db import sessionlocal
from models import Conversation, Vector, Category

TIMEZONE = "Europe/Helsinki"

# pylint: disable=no-member

async def create_vector(text, conv_id):
    embedding = None
    vec = Vector(text=text, conversation_id=conv_id, embedding=embedding)
    with sessionlocal.begin() as session:
        session.add(vec)
    return vec

async def delete_vector(vec_id):
    with sessionlocal.begin() as session:
        vec = session.get_one(Vector, vec_id)
        session.delete(vec)

async def get_vector_by_id(vec_id):
    with sessionlocal() as session:
        return session.get(Vector, vec_id)

async def get_vectors_by_conversation_id(conv_id):
    with sessionlocal() as session:
        return session.scalars(select(Vector).where(Vector.conversation_id == conv_id)).all()

async def get_vectors():
    with sessionlocal() as session:
        return session.scalars(select(Vector)).all()

async def create_conversation(name, summary, cat_id=None, timestamp=None):
    if not timestamp:
        timestamp = datetime.now(ZoneInfo(TIMEZONE))
    conv = Conversation(name=name, summary=summary, category_id=cat_id, timestamp=timestamp)
    with sessionlocal.begin() as session:
        session.add(conv)
    return conv

async def delete_conversation(conv_id):
    with sessionlocal.begin() as session:
        conv = session.get_one(Conversation, conv_id)
        session.delete(conv)

async def get_conversation_by_id(conv_id):
    with sessionlocal() as session:
        return session.get(Conversation, conv_id)

async def get_conversations_by_category_id(cat_id):
    with sessionlocal() as session:
        return session.scalars(select(Conversation).where(Conversation.category_id == cat_id)).all()

async def get_conversations_by_category_name(cat_name):
    with sessionlocal() as session:
        return session.scalars(
            select(Conversation).join(Category).where(Category.name == cat_name)
        ).all()

async def get_conversations():
    with sessionlocal() as session:
        return session.scalars(select(Conversation)).all()

async def create_category(cat_name):
    cat = Category(name=cat_name)
    with sessionlocal.begin() as session:
        session.add(cat)
    return cat

async def delete_category_by_id(cat_id):
    with sessionlocal.begin() as session:
        cat = session.get_one(Category, cat_id)
        session.delete(cat)

async def delete_category_by_name(cat_name):
    with sessionlocal.begin() as session:
        cat = session.scalars(select(Category).where(Category.name == cat_name)).one()
        session.delete(cat)

async def get_category_by_id(cat_id):
    with sessionlocal() as session:
        return session.get(Category, cat_id)

async def get_category_by_name(cat_name):
    with sessionlocal() as session:
        return session.scalars(select(Category).where(Category.name == cat_name)).one_or_none()

async def get_categories():
    with sessionlocal() as session:
        return session.scalars(select(Category)).all()
