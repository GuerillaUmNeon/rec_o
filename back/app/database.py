import os
from pathlib import Path
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def _get_database_url() -> str:
    """Build PostgreSQL URL from .env vars, or use DATABASE_URL if set."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    host = os.getenv("POSTGRES")
    database = os.getenv("DATABASE")
    user = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT", "5432")

    missing = [
        name
        for name, value in {
            "POSTGRES": host,
            "DATABASE": database,
            "DB_USERNAME": user,
            "DB_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

    user_q = quote_plus(user)
    password_q = quote_plus(password)
    return f"postgresql+psycopg://{user_q}:{password_q}@{host}:{port}/{database}"


engine = create_engine(_get_database_url(), connect_args={"connect_timeout": 300})


def fetch_all(query: str, params: tuple | None = None) -> list:
    """Run a SELECT and return all rows."""
    with engine.connect() as conn:
        # For positional parameters, we need to pass a list or just execute it directly
        # with psycopg2/psycopg3 style query. 
        # But SQLAlchemy text() wants named parameters if we use : syntax.
        # If we use %s, it might be an issue. 
        # Let's try raw execution if it's positional.
        result = conn.execute(text(query), params or {})
        return result.fetchall()
