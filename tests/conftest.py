"""Test bootstrap.

Pre-populates the env vars `Settings()` requires so each test file doesn't
have to. Individual tests can still override via `patch.dict`.
"""
import os

from cryptography.fernet import Fernet

os.environ.setdefault("DATABASE_URL", "postgresql://fake/fake")
os.environ.setdefault("VESSEL_AUTH_TOKEN", "test-token-of-sufficient-length")
os.environ.setdefault("VESSEL_ENCRYPTION_KEY", Fernet.generate_key().decode())
os.environ.setdefault("CLAUDE_API_KEY", "")
