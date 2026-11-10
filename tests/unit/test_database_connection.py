import pytest

from src.database import connection


def write_env_file(tmp_path, content):
    env_file = tmp_path / ".env.test"
    env_file.write_text(content, encoding="utf-8")
    return env_file


def test_get_engine_uses_environment_file_and_prefix(
    tmp_path, monkeypatch
):
    write_env_file(
        tmp_path,
        """
TEST_DB_HOST=localhost
TEST_DB_PORT=5433
TEST_DB_NAME=data_quality_detective_test
TEST_DB_USER=test_user
TEST_DB_PASSWORD=test_password
""".strip(),
    )

    monkeypatch.setattr(connection, "PROJECT_ROOT", tmp_path)

    captured = {}

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return "test_engine"

    monkeypatch.setattr(connection, "create_engine", fake_create_engine)

    engine = connection.get_engine(
        env_file=".env.test",
        expected_database="data_quality_detective_test",
        env_prefix="TEST_",
    )

    assert engine == "test_engine"
    assert captured["url"].database == "data_quality_detective_test"
    assert captured["url"].host == "localhost"
    assert captured["url"].port == 5433
    assert captured["url"].username == "test_user"
    assert captured["kwargs"]["pool_pre_ping"] is True


def test_get_engine_rejects_unexpected_database(
    tmp_path, monkeypatch
):
    write_env_file(
        tmp_path,
        """
DB_HOST=localhost
DB_NAME=production_database
DB_USER=test_user
DB_PASSWORD=test_password
""".strip(),
    )

    monkeypatch.setattr(connection, "PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="Unsafe database configuration"):
        connection.get_engine(
            env_file=".env.test",
            expected_database="data_quality_detective_test",
        )


def test_get_engine_rejects_missing_settings(
    tmp_path, monkeypatch
):
    write_env_file(tmp_path, "DB_HOST=localhost")

    monkeypatch.setattr(connection, "PROJECT_ROOT", tmp_path)

    for key in (
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="Missing database settings"):
        connection.get_engine(env_file=".env.test")


def test_get_engine_rejects_non_integer_port(
    tmp_path, monkeypatch
):
    write_env_file(
        tmp_path,
        """
DB_HOST=localhost
DB_NAME=test_database
DB_USER=test_user
DB_PASSWORD=test_password
DB_PORT=invalid
""".strip(),
    )

    monkeypatch.setattr(connection, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="DB_PORT must be a valid integer"):
        connection.get_engine(env_file=".env.test")


def test_get_engine_rejects_port_out_of_range(
    tmp_path, monkeypatch
):
    write_env_file(
        tmp_path,
        """
DB_HOST=localhost
DB_NAME=test_database
DB_USER=test_user
DB_PASSWORD=test_password
DB_PORT=70000
""".strip(),
    )

    monkeypatch.setattr(connection, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="DB_PORT must be between 1 and 65535"):
        connection.get_engine(env_file=".env.test")


def test_get_engine_rejects_missing_environment_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(connection, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="Environment file not found"):
        connection.get_engine(env_file=".missing.env")
