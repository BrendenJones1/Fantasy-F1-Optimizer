from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from f1opt.db import Base


def test_migrations_match_models(engine):
    """The migration chain must produce exactly the schema the ORM models describe."""
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []
