import os
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_engine(
    env_file=".env",
    expected_database=None,
    env_prefix="",
):
    """
    Create a SQLAlchemy PostgreSQL engine.

    Values from the specified environment file take precedence
    over matching process environment variables.
    """
    env_path = PROJECT_ROOT / env_file

    if not env_path.is_file():
        raise FileNotFoundError(
            f"Environment file not found: {env_path}"
        )

    file_config = dotenv_values(env_path)

    def get_setting(key, default=None):
        config_key = f"{env_prefix}{key}"

        value = file_config.get(config_key)

        if value is None or value == "":
            value = os.getenv(config_key, default)

        return value

    required = [
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]

    config = {
        key: get_setting(key)
        for key in required
    }

    missing = [
        key for key, value in config.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing database settings: "
            f"{[f'{env_prefix}{key}' for key in missing]}"
        )

    database = config["DB_NAME"]

    if expected_database and database != expected_database:
        raise RuntimeError(
            "Unsafe database configuration. "
            f"Expected '{expected_database}', got '{database}'."
        )

    try:
        port = int(get_setting("DB_PORT", "5432"))
    except ValueError as exc:
        raise ValueError(
            f"{env_prefix}DB_PORT must be a valid integer."
        ) from exc

    if not 1 <= port <= 65535:
        raise ValueError(
            f"{env_prefix}DB_PORT must be between 1 and 65535."
        )

    url = URL.create(
        "postgresql+psycopg2",
        username=config["DB_USER"],
        password=config["DB_PASSWORD"],
        host=config["DB_HOST"],
        port=port,
        database=database,
    )

    return create_engine(url, pool_pre_ping=True)
