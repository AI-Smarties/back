from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, Integer, String
from db import Base


EMBEDDING_DIMENSIONS = 768


class Items(Base): # pylint: disable=too-few-public-methods
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    embedding = Column(VECTOR(EMBEDDING_DIMENSIONS))
