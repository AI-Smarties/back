from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db import Base


EMBEDDING_DIMENSIONS = 768


# pylint: disable=too-few-public-methods


class Vector(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    embedding = Column(VECTOR(EMBEDDING_DIMENSIONS), nullable=False)
    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    conversation = relationship("Conversation", back_populates="vectors")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    name = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), index=True)

    category = relationship("Category", back_populates="conversations")
    vectors = relationship(
        "Vector",
        back_populates="conversation",
        cascade="all, delete",
        passive_deletes=True,
    )


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False, unique=True, index=True)

    conversations = relationship("Conversation", back_populates="category",)
