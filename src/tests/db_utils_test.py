from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError, NoResultFound

import db
import db_utils


@pytest.fixture(scope="module", autouse=True)
def ensure_database_availability():
    try:
        db.drop_tables()
    except OperationalError:
        pytest.skip(
            "Database is not available (expected at localhost:5432). "
            "Run `docker compose up -d` to enable DB tests."
        )


@pytest.fixture(autouse=True)
def clean_tables():
    db.drop_tables()
    db.create_tables()


def test_categories():
    cat = db_utils.create_category("test", user_id="test-user")
    assert cat.id is not None
    assert cat.name == "test"

    fetched = db_utils.get_category_by_id(cat.id, user_id="test-user")
    assert fetched is not None
    assert fetched.id == cat.id
    assert fetched.name == cat.name

    fetched2 = db_utils.get_category_by_name("test", user_id="test-user")
    assert fetched2 is not None
    assert fetched2.id == cat.id
    assert fetched2.name == cat.name

    cats = db_utils.get_categories(user_id="test-user")
    assert isinstance(cats, list)
    assert len(cats) == 1
    assert cats[0].id == cat.id

    with pytest.raises(IntegrityError):
        db_utils.create_category("test", user_id="test-user")

    assert db_utils.delete_category_by_id(cat.id, user_id="test-user") is None
    with pytest.raises(NoResultFound):
        db_utils.delete_category_by_name("test", user_id="test-user")


def test_conversations():
    cat = db_utils.create_category("cat", user_id="test-user")
    conv = db_utils.create_conversation(
        name="name", user_id="test-user", summary="summary", cat_id=cat.id)
    assert conv.id is not None
    assert conv.name == "name"
    assert conv.summary == "summary"
    assert conv.category_id == cat.id
    assert conv.timestamp is not None
    assert conv.timestamp.tzinfo is not None

    fetched = db_utils.get_conversation_by_id(conv.id, user_id="test-user")
    assert fetched is not None
    assert fetched.id == conv.id
    assert fetched.name == conv.name
    assert fetched.summary == conv.summary
    assert fetched.category_id == cat.id
    assert fetched.timestamp == conv.timestamp

    convs = db_utils.get_conversations(user_id="test-user")
    assert isinstance(convs, list)
    assert len(convs) == 1
    assert convs[0].id == conv.id

    convs_by_cat = db_utils.get_conversations_by_category_id(
        cat.id, user_id="test-user")
    assert isinstance(convs_by_cat, list)
    assert len(convs_by_cat) == 1
    assert convs_by_cat[0].id == conv.id

    fetched = db_utils.update_conversation_summary(
        conv.id, "new summary", user_id="test-user")
    fetched = db_utils.update_conversation_category(
        conv.id, None, user_id="test-user")
    assert fetched is not None
    assert fetched.summary == "new summary"
    assert fetched.category_id is None

    assert db_utils.delete_conversation(conv.id, user_id="test-user") is None


def test_vectors_with_stub_embedding_model(monkeypatch):
    class _StubEmbeddingModel:  # pylint: disable=too-few-public-methods
        def get_embeddings(self, texts, output_dimensionality):
            assert isinstance(texts, list)
            assert output_dimensionality == db_utils.EMBEDDING_DIMENSIONS
            embedding = [0.0] * db_utils.EMBEDDING_DIMENSIONS
            embedding[0] = 1.0
            return [SimpleNamespace(values=embedding)]

    monkeypatch.setattr(db_utils, "EMBEDDING_MODEL", _StubEmbeddingModel())

    conv = db_utils.create_conversation(
        "conv", user_id="test-user", summary="summary", cat_id=None)
    vec = db_utils.create_vector("hello", conv.id)
    assert vec.id is not None
    assert vec.text == "hello"
    assert vec.conversation_id == conv.id
    assert len(vec.embedding) == db_utils.EMBEDDING_DIMENSIONS

    fetched = db_utils.get_vector_by_id(vec.id, user_id="test-user")
    assert fetched is not None
    assert fetched.id == vec.id
    assert fetched.text == vec.text
    assert fetched.conversation_id == vec.conversation_id
    assert len(fetched.embedding) == len(vec.embedding)

    vecs = db_utils.get_vectors(user_id="test-user")
    assert isinstance(vecs, list)
    assert len(vecs) == 1
    assert vecs[0].id == vec.id

    vecs_by_conv = db_utils.get_vectors_by_conversation_id(
        conv.id, user_id="test-user")
    assert isinstance(vecs_by_conv, list)
    assert len(vecs_by_conv) == 1
    assert vecs_by_conv[0].id == vec.id

    vecs_by_search = db_utils.search_vectors(
        "hello", user_id="test-user", limit=1)
    assert isinstance(vecs_by_search, list)
    assert len(vecs_by_search) == 1
    assert vecs_by_search[0].id == vec.id

    assert db_utils.delete_vector(vec.id, user_id="test-user") is None
