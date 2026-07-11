import asyncpg

from ..encryption import decrypt_json, encrypt_json
from ..models import StateData


async def read(pool: asyncpg.Pool, user_id: str) -> StateData:
    """Return the user's StateData, or an empty one if they don't exist yet."""
    row = await pool.fetchrow(
        "SELECT encrypted_data FROM vessel.state WHERE user_id = $1",
        user_id,
    )
    if row is None:
        return StateData()
    raw = decrypt_json(bytes(row["encrypted_data"]))
    return StateData.model_validate(raw)


async def write(pool: asyncpg.Pool, user_id: str, state: StateData) -> None:
    blob = encrypt_json(state.model_dump(mode="json"))
    await pool.execute(
        """
        INSERT INTO vessel.state (user_id, encrypted_data, updated_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (user_id) DO UPDATE
            SET encrypted_data = EXCLUDED.encrypted_data,
                updated_at = NOW()
        """,
        user_id,
        blob,
    )


async def list_user_ids(pool: asyncpg.Pool) -> list[str]:
    rows = await pool.fetch("SELECT user_id FROM vessel.state")
    return [r["user_id"] for r in rows]
