import json
import logging
from pathlib import Path
from typing import Optional

import asyncpg

from .config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.execute("SET search_path TO vessel, public")


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            init=_init_connection,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Apply SQL migrations in lexical order — each at most once.

    Tracks applied migrations in `vessel._migrations`. New migration files
    are auto-discovered; previously-recorded ones are skipped, which makes
    destructive DROPs in later migrations safe across restarts.
    """
    await pool.execute("CREATE SCHEMA IF NOT EXISTS vessel")
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS vessel._migrations (
            name TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    rows = await pool.fetch("SELECT name FROM vessel._migrations")
    applied = {r["name"] for r in rows}

    migrations_dir = Path(__file__).parent / "migrations"
    sql_files = sorted(migrations_dir.glob("*.sql"))
    for sql_file in sql_files:
        if sql_file.name in applied:
            continue
        sql = sql_file.read_text()
        logger.info("Applying migration %s", sql_file.name)
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO vessel._migrations (name) VALUES ($1) "
                    "ON CONFLICT (name) DO NOTHING",
                    sql_file.name,
                )
