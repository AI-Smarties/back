from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from db import Base  # pylint: disable=cyclic-import


# pylint: disable=too-few-public-methods


EMBEDDING_DIMENSIONS = 768


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
    user_id = Column(Text, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    name = Column(Text, nullable=False)
    summary = Column(Text)
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
    user_id = Column(Text, nullable=False, index=True)
    name = Column(Text, nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("name", "user_id"),
    )

    conversations = relationship("Conversation", back_populates="category")
