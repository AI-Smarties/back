from types import SimpleNamespace
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError, NoResultFound
import db
import db_utils


def is_postgres_available():
    try:
        with db.sessionlocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


@pytest.fixture(scope="module", autouse=True)
def ensure_db_schema():
    if not is_postgres_available():
        pytest.skip(
            "Postgres is not available (expected at localhost:5432). "
            "Run `docker compose up -d` to enable DB tests."
        )
    db.drop_tables()
    db.create_tables()


@pytest.fixture(autouse=True)
def clean_tables():
    with db.sessionlocal.begin() as session:  # pylint: disable=no-member
        session.execute(
            text(
                "TRUNCATE TABLE vectors, conversations, categories "
                "RESTART IDENTITY CASCADE"
            )
        )


def test_categories():
    cat = db_utils.create_category("test")
    assert cat.id is not None
    assert cat.name == "test"

    fetched = db_utils.get_category_by_id(cat.id)
    assert fetched is not None
    assert fetched.id == cat.id
    assert fetched.name == cat.name

    fetched2 = db_utils.get_category_by_name("test")
    assert fetched2 is not None
    assert fetched2.id == cat.id
    assert fetched2.name == cat.name

    cats = db_utils.get_categories()
    assert isinstance(cats, list)
    assert len(cats) == 1
    assert cats[0].id == cat.id

    with pytest.raises(IntegrityError):
        db_utils.create_category("test")

    assert db_utils.delete_category_by_id(cat.id) is None
    with pytest.raises(NoResultFound):
        db_utils.delete_category_by_name("test")


def test_conversations():
    cat = db_utils.create_category("cat")
    conv = db_utils.create_conversation("name", "summary", cat_id=cat.id)
    assert conv.id is not None
    assert conv.name == "name"
    assert conv.summary == "summary"
    assert conv.category_id == cat.id
    assert conv.timestamp is not None
    assert conv.timestamp.tzinfo is not None

    fetched = db_utils.get_conversation_by_id(conv.id)
    assert fetched is not None
    assert fetched.id == conv.id
    assert fetched.name == conv.name
    assert fetched.summary == conv.summary
    assert fetched.category_id == cat.id
    assert fetched.timestamp == conv.timestamp

    convs = db_utils.get_conversations()
    assert isinstance(convs, list)
    assert len(convs) == 1
    assert convs[0].id == conv.id

    convs_by_cat = db_utils.get_conversations_by_category_id(cat.id)
    assert isinstance(convs_by_cat, list)
    assert len(convs_by_cat) == 1
    assert convs_by_cat[0].id == conv.id

    fetched = db_utils.update_conversation_summary(conv.id, "new summary")
    fetched = db_utils.update_conversation_category(conv.id, None)
    assert fetched is not None
    assert fetched.summary == "new summary"
    assert fetched.category_id is None

    assert db_utils.delete_conversation(conv.id) is None


def test_vectors_with_stub_embedding_model(monkeypatch):
    class _StubEmbeddingModel:  # pylint: disable=too-few-public-methods
        def get_embeddings(self, texts, output_dimensionality):
            assert isinstance(texts, list)
            assert output_dimensionality == db_utils.EMBEDDING_DIMENSIONS
            return [SimpleNamespace(values=[0.0] * db_utils.EMBEDDING_DIMENSIONS)]

    monkeypatch.setattr(db_utils, "EMBEDDING_MODEL", _StubEmbeddingModel())

    conv = db_utils.create_conversation("conv", "summary")
    vec = db_utils.create_vector("hello", conv.id)
    assert vec.id is not None
    assert vec.text == "hello"
    assert vec.conversation_id == conv.id
    assert len(vec.embedding) == db_utils.EMBEDDING_DIMENSIONS

    fetched = db_utils.get_vector_by_id(vec.id)
    assert fetched is not None
    assert fetched.id == vec.id
    assert fetched.text == vec.text
    assert fetched.conversation_id == vec.conversation_id
    assert len(fetched.embedding) == len(vec.embedding)

    vecs = db_utils.get_vectors()
    assert isinstance(vecs, list)
    assert len(vecs) == 1
    assert vecs[0].id == vec.id

    vecs_by_conv = db_utils.get_vectors_by_conversation_id(conv.id)
    assert isinstance(vecs_by_conv, list)
    assert len(vecs_by_conv) == 1
    assert vecs_by_conv[0].id == vec.id

    vecs_by_search = db_utils.search_vectors("hello", limit=1)
    assert isinstance(vecs_by_search, list)
    assert len(vecs_by_search) == 1
    assert vecs_by_search[0].id == vec.id

    assert db_utils.delete_vector(vec.id) is None
