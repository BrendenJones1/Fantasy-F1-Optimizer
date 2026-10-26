from alembic import context
from sqlalchemy import engine_from_config, pool

from f1opt.config import get_settings
from f1opt.db import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    # An explicit sqlalchemy.url (set by tests) wins over the environment.
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=connection.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
