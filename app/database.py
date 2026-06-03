import os
from pathlib import Path
from urllib.parse import quote_plus

import psycopg
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
    return f"postgresql://{user_q}:{password_q}@{host}:{port}/{database}"


def get_connection():
    """Open a PostgreSQL connection. Close it when done (or use a ``with`` block)."""
    return psycopg.connect(_get_database_url())


def fetch_all(query: str, params: tuple | None = None) -> list:
    """Run a SELECT and return all rows. Opens and closes the connection."""
    with get_connection() as conn: # close the connection when done
        with conn.cursor() as cursor: # close the cursor when done
            cursor.execute(query, params)
            return cursor.fetchall()
