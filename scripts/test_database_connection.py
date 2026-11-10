import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_engine


live_engine = get_engine(
    env_file=".env",
    expected_database="data_quality_detective",
)

print(
    "Live engine URL:",
    live_engine.url.render_as_string(hide_password=True),
)
live_engine.dispose()


test_engine = get_engine(
    env_file=".env.test",
    expected_database="data_quality_detective_test",
    env_prefix="TEST_",
)

print(
    "Test engine URL:",
    test_engine.url.render_as_string(hide_password=True),
)
test_engine.dispose()

print("\nDATABASE ENGINE FACTORY CHECK PASSED.")
