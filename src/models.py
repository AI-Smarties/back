from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db import Base


EMBEDDING_DIMENSIONS = 768


class Vectors(Base): # pylint: disable=too-few-public-methods
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    text = Column(String, nullable=False)
    embedding = Column(VECTOR(EMBEDDING_DIMENSIONS), nullable=False)

    conversation = relationship("Conversations", back_populates="vectors")


class Conversations(Base): # pylint: disable=too-few-public-methods
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    name = Column(String, nullable=False)
    summary = Column(String, nullable=False)

    vectors = relationship("Vectors", back_populates="conversation")
    category = relationship("Categories", back_populates="conversations")


class Categories(Base): # pylint: disable=too-few-public-methods
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)

    conversations = relationship("Conversations", back_populates="category")
