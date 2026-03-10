from datetime import datetime
from zoneinfo import ZoneInfo
import vertexai
from vertexai.language_models import TextEmbeddingModel
import google
from sqlalchemy import select
from db import sessionlocal
from models import Conversation, Vector, Category, EMBEDDING_DIMENSIONS

TIMEZONE = "Europe/Helsinki"
EMBEDDING_MODEL = None

def load_embedding_model():
    global EMBEDDING_MODEL  # pylint: disable=global-statement
    _, project = google.auth.default()
    vertexai.init(project=project, location="europe-north1")
    EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

# pylint: disable=no-member

def create_vector(text, conv_id):
    if not EMBEDDING_MODEL:
        load_embedding_model()
    embedding = EMBEDDING_MODEL.get_embeddings(
        [text],
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )[0].values
    vec = Vector(text=text, conversation_id=conv_id, embedding=embedding)
    with sessionlocal.begin() as session:
        session.add(vec)
        return vec

def delete_vector(vec_id):
    with sessionlocal.begin() as session:
        vec = session.get_one(Vector, vec_id)
        session.delete(vec)

def get_vector_by_id(vec_id):
    with sessionlocal() as session:
        return session.get(Vector, vec_id)

def get_vectors_by_conversation_id(conv_id):
    with sessionlocal() as session:
        return session.scalars(select(Vector).where(Vector.conversation_id == conv_id)).all()

def get_vectors():
    with sessionlocal() as session:
        return session.scalars(select(Vector)).all()

def search_vectors(text, limit=1, max_distance=0.5):
    if not EMBEDDING_MODEL:
        load_embedding_model()
    embedding = EMBEDDING_MODEL.get_embeddings(
        [text],
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )[0].values
    with sessionlocal() as session:
        return session.scalars(
            select(Vector)
            .where(Vector.embedding.cosine_distance(embedding) < max_distance) # how "relevant" the query response should be on scale of 0-2 (float), 0 = identical 1 = unrelated 2 = opposite
            .limit(limit)
        ).all()

def create_conversation(name, summary=None, cat_id=None, timestamp=None):
    if not timestamp:
        timestamp = datetime.now(ZoneInfo(TIMEZONE))
    conv = Conversation(name=name, summary=summary, category_id=cat_id, timestamp=timestamp)
    with sessionlocal.begin() as session:
        session.add(conv)
        return conv

def update_conversation_summary(conv_id, summary):
    with sessionlocal.begin() as session:
        conv = session.get_one(Conversation, conv_id)
        conv.summary = summary
        session.add(conv)
        return conv

def update_conversation_category(conv_id, cat_id):
    with sessionlocal.begin() as session:
        conv = session.get_one(Conversation, conv_id)
        conv.category_id = cat_id
        session.add(conv)
        return conv

def delete_conversation(conv_id):
    with sessionlocal.begin() as session:
        conv = session.get_one(Conversation, conv_id)
        session.delete(conv)

def get_conversation_by_id(conv_id):
    with sessionlocal() as session:
        return session.get(Conversation, conv_id)

def get_conversations_by_category_id(cat_id):
    with sessionlocal() as session:
        return session.scalars(select(Conversation).where(Conversation.category_id == cat_id)).all()

def get_conversations():
    with sessionlocal() as session:
        return session.scalars(select(Conversation)).all()

def create_category(name):
    cat = Category(name=name)
    with sessionlocal.begin() as session:
        session.add(cat)
        return cat

def delete_category_by_id(cat_id):
    with sessionlocal.begin() as session:
        cat = session.get_one(Category, cat_id)
        session.delete(cat)

def delete_category_by_name(name):
    with sessionlocal.begin() as session:
        cat = session.scalars(select(Category).where(Category.name == name)).one()
        session.delete(cat)

def get_category_by_id(cat_id):
    with sessionlocal() as session:
        return session.get(Category, cat_id)

def get_category_by_name(name):
    with sessionlocal() as session:
        return session.scalars(select(Category).where(Category.name == name)).one_or_none()

def get_categories():
    with sessionlocal() as session:
        return session.scalars(select(Category)).all()
