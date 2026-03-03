import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ai_smarties_db")

DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
DB_URL += f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(DB_URL)

sessionlocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

Base = declarative_base()

def create_tables():
    import models  # pylint: disable=unused-import  import-outside-toplevel
    with sessionlocal.begin() as session:  # pylint: disable=no-member
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE"))
    Base.metadata.create_all(engine)

def drop_tables():
    import models  # pylint: disable=unused-import  import-outside-toplevel
    Base.metadata.drop_all(engine)
    with sessionlocal.begin() as session:  # pylint: disable=no-member
        session.execute(text("DROP EXTENSION IF EXISTS vector CASCADE"))
