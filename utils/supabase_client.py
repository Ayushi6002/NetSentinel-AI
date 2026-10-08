"""Supabase configuration and client helper for NetSentinel AI.

Reads Supabase credentials safely from the environment.
Does NOT force a connection or assume database schema.
"""
import os
from typing import Optional


def get_supabase_config() -> dict:
    """Safely inspects whether Supabase environment variables are provided."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    return {
        "url_configured": bool(url),
        "key_configured": bool(key),
        "ready": bool(url and key),
    }


def get_supabase_client():
    """Initializes and returns a Supabase client if credentials are configured.

    Returns None if credentials are missing or if client creation fails.
    """
    config = get_supabase_config()
    if not config["ready"]:
        return None

    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()
        return create_client(url, key)
    except Exception:
        return None
