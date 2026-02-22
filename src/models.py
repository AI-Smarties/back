from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db import Base


EMBEDDING_DIMENSIONS = 768


# pylint: disable=too-few-public-methods


class Vectors(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    embedding = Column(VECTOR(EMBEDDING_DIMENSIONS), nullable=False)
    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )

    conversation = relationship("Conversations", back_populates="vectors")


class Conversations(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    name = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    category_id = Column(
        Integer,
        ForeignKey("categories.id"),
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )

    category = relationship("Categories", back_populates="conversations")
    vectors = relationship(
        "Vectors",
        back_populates="conversation",
        cascade="all, delete",
        passive_deletes=True,
    )


class Categories(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)

    conversations = relationship(
        "Conversations",
        back_populates="category",
        cascade="all, delete",
        passive_deletes=True,
    )
