from datetime import datetime

from zoneinfo import ZoneInfo
import vertexai
from vertexai.language_models import TextEmbeddingModel
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from google import auth

from db import sessionlocal
from models import Conversation, Vector, Category, EMBEDDING_DIMENSIONS


# pylint: disable=no-member


TIMEZONE = "Europe/Helsinki"
EMBEDDING_MODEL = None


def load_embedding_model():
    global EMBEDDING_MODEL  # pylint: disable=global-statement
    _, project = auth.default()
    vertexai.init(project=project, location="europe-north1")
    EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained(
        "text-multilingual-embedding-002")


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


def create_vectors_batch(texts, conv_id):
    """Create multiple vectors with a single embedding API call."""
    if not texts:
        return []
    if not EMBEDDING_MODEL:
        load_embedding_model()
    embeddings = EMBEDDING_MODEL.get_embeddings(
        texts,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    vecs = [
        Vector(text=text, conversation_id=conv_id, embedding=emb.values)
        for text, emb in zip(texts, embeddings)
    ]
    with sessionlocal.begin() as session:
        for vec in vecs:
            session.add(vec)
        return vecs


def delete_vector(vec_id, user_id):
    with sessionlocal.begin() as session:
        stmt = (
            select(Vector)
            .join(Conversation)
            .where(
                Vector.id == vec_id,
                Conversation.user_id == user_id,
            )
        )
        vec = session.scalars(stmt).one()
        session.delete(vec)


def get_vector_by_id(vec_id, user_id):
    with sessionlocal() as session:
        stmt = (
            select(Vector)
            .join(Conversation)
            .where(
                Vector.id == vec_id,
                Conversation.user_id == user_id,
            )
        )
        result = session.scalars(stmt).one()
        return result


def get_vectors_by_conversation_id(conv_id, user_id):
    with sessionlocal() as session:
        return session.scalars(
            select(Vector)
            .join(Conversation)
            .where(
                Vector.conversation_id == conv_id,
                Conversation.user_id == user_id,
            )
        ).all()


def get_vectors(user_id):
    with sessionlocal() as session:
        return session.scalars(
            select(Vector)
            .join(Conversation)
            .where(Conversation.user_id == user_id)
        ).all()


def search_vectors(text, user_id, limit=1, max_distance=0.5):
    if user_id is None:
        raise ValueError("user_id required")

    if not EMBEDDING_MODEL:
        load_embedding_model()

    embedding = EMBEDDING_MODEL.get_embeddings(
        [text],
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )[0].values

    with sessionlocal() as session:
        distance = Vector.embedding.cosine_distance(embedding)

        stmt = (
            select(Vector)
            .join(Conversation)
            .options(joinedload(Vector.conversation))
            .where(
                distance < max_distance,
                Conversation.user_id == user_id,
            )
        )

        return session.scalars(
            stmt.order_by(distance).limit(limit)
        ).all()


def create_conversation(name, user_id, summary=None, cat_id=None, timestamp=None):
    if user_id is None:
        raise ValueError("user_id required")
    if not timestamp:
        timestamp = datetime.now(ZoneInfo(TIMEZONE))
    conv = Conversation(
        name=name,
        summary=summary,
        category_id=cat_id,
        timestamp=timestamp,
        user_id=user_id,
    )
    with sessionlocal.begin() as session:
        session.add(conv)
        return conv


def update_conversation_summary(conv_id, summary, user_id):
    with sessionlocal.begin() as session:
        stmt = select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == user_id,
        )
        conv = session.scalars(stmt).one()
        conv.summary = summary
        session.add(conv)
        return conv


def update_conversation_category(conv_id, cat_id, user_id):
    with sessionlocal.begin() as session:
        stmt = select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == user_id,
        )
        conv = session.scalars(stmt).one()
        conv.category_id = cat_id
        session.add(conv)
        return conv


def delete_conversation(conv_id, user_id):
    with sessionlocal.begin() as session:
        stmt = select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == user_id,
        )
        conv = session.scalars(stmt).one()
        session.delete(conv)


def get_conversation_by_id(conv_id, user_id):
    with sessionlocal() as session:
        stmt = select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == user_id,
        )
        result = session.scalars(stmt).one()
        return result


def get_conversations_by_category_id(cat_id, user_id):
    with sessionlocal() as session:
        return session.scalars(
            select(Conversation).where(
                Conversation.category_id == cat_id,
                Conversation.user_id == user_id,
            )
        ).all()


def get_conversations(user_id):
    with sessionlocal() as session:
        return session.scalars(
            select(Conversation).where(Conversation.user_id == user_id)
        ).all()


def create_category(name, user_id):
    cat = Category(name=name, user_id=user_id)
    with sessionlocal.begin() as session:
        session.add(cat)
        return cat


def delete_category_by_id(cat_id, user_id):
    with sessionlocal.begin() as session:
        stmt = select(Category).where(
            Category.id == cat_id,
            Category.user_id == user_id,
        )
        cat = session.scalars(stmt).one()
        session.delete(cat)


def delete_category_by_name(name, user_id):
    with sessionlocal.begin() as session:
        stmt = select(Category).where(
            Category.name == name,
            Category.user_id == user_id,
        )
        cat = session.scalars(stmt).one()
        session.delete(cat)


def get_category_by_id(cat_id, user_id):
    with sessionlocal() as session:
        stmt = select(Category).where(
            Category.id == cat_id,
            Category.user_id == user_id,
        )
        result = session.scalars(stmt).one()
        return result


def get_category_by_name(name, user_id):
    with sessionlocal() as session:
        stmt = select(Category).where(
            Category.name == name,
            Category.user_id == user_id,
        )
        result = session.scalars(stmt).one()
        return result


def get_categories(user_id):
    with sessionlocal() as session:
        return session.scalars(
            select(Category).where(Category.user_id == user_id)
        ).all()
