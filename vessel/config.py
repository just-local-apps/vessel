from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = Field(..., alias="DATABASE_URL")

    # Which LLM the agents call. "groq" (default) → cheap/fast Groq Cloud
    # via OpenAI-compatible API. "anthropic" → Claude via the official SDK.
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")

    claude_api_key: Optional[str] = Field(default=None, alias="CLAUDE_API_KEY")
    claude_model: str = Field(default="claude-opus-4-7", alias="CLAUDE_MODEL")

    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")

    # Optional: legacy single-user token; not used for auth comparison anymore
    # but kept so existing .env files don't blow up.
    vessel_auth_token: Optional[str] = Field(default=None, alias="VESSEL_AUTH_TOKEN")
    # Required: 44-char base64 Fernet key used to encrypt every user-scoped
    # artifact at rest (state JSON, event payloads, invocation prompts).
    vessel_encryption_key: str = Field(..., alias="VESSEL_ENCRYPTION_KEY")

    phoenix_api_key: Optional[str] = Field(default=None, alias="PHOENIX_API_KEY")
    phoenix_collector_endpoint: Optional[str] = Field(
        default=None, alias="PHOENIX_COLLECTOR_ENDPOINT"
    )
    phoenix_project: str = Field(default="vessel", alias="PHOENIX_PROJECT")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")

    timezone: str = Field(default="America/Los_Angeles", alias="TIMEZONE")
    wake_hour: int = Field(default=6, alias="WAKE_HOUR")
    # Default bumped from 21 → 23 to match the live Fly secret value.
    # Late-evening tasks (e.g. wash dishes after 7pm) need a wider
    # bedtime window or `_pick_now_card` collapses free_minutes to 0
    # and the focus card disappears at 9pm.
    bedtime_hour: int = Field(default=23, alias="BEDTIME_HOUR")
    workday_start_hour: int = Field(default=9, alias="WORKDAY_START_HOUR")
    workday_end_hour: int = Field(default=17, alias="WORKDAY_END_HOUR")
    # Minutes of completed-task work before vessel suggests a break.
    work_before_break_min: int = Field(
        default=90, alias="WORK_BEFORE_BREAK_MIN"
    )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
