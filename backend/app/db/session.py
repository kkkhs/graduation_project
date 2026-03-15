from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


SessionFactory = sessionmaker



def build_engine(db_path: str):
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )



def build_session_factory(engine) -> SessionFactory:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
