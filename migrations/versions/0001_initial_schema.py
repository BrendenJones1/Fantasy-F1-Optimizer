"""initial schema

Revision ID: 0001
Revises: 
"""
from alembic import op
import sqlalchemy as sa


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table('constructors',
    sa.Column('constructor_id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.PrimaryKeyConstraint('constructor_id')
    )
    op.create_table('drivers',
    sa.Column('driver_id', sa.String(length=64), nullable=False),
    sa.Column('code', sa.String(length=8), nullable=True),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.PrimaryKeyConstraint('driver_id')
    )
    op.create_table('ingestion_runs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('started_at', sa.DateTime(), nullable=False),
    sa.Column('finished_at', sa.DateTime(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('attempt', sa.Integer(), nullable=False),
    sa.Column('seasons', sa.String(length=64), nullable=True),
    sa.Column('races_completed', sa.Integer(), nullable=False),
    sa.Column('results_upserted', sa.Integer(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('races',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('season', sa.Integer(), nullable=False),
    sa.Column('round', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('circuit_id', sa.String(length=64), nullable=False),
    sa.Column('is_sprint_weekend', sa.Boolean(), nullable=False),
    sa.Column('completed', sa.Boolean(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('season', 'round', name='uq_races_season_round')
    )
    op.create_table('constructor_prices',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('race_id', sa.Integer(), nullable=False),
    sa.Column('constructor_id', sa.String(length=64), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('price', sa.Numeric(precision=6, scale=2), nullable=False),
    sa.Column('captured_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['constructor_id'], ['constructors.constructor_id'], ),
    sa.ForeignKeyConstraint(['race_id'], ['races.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('race_id', 'constructor_id', 'source', name='uq_constructor_price')
    )
    op.create_table('driver_prices',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('race_id', sa.Integer(), nullable=False),
    sa.Column('driver_id', sa.String(length=64), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('price', sa.Numeric(precision=6, scale=2), nullable=False),
    sa.Column('captured_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['driver_id'], ['drivers.driver_id'], ),
    sa.ForeignKeyConstraint(['race_id'], ['races.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('race_id', 'driver_id', 'source', name='uq_driver_price')
    )
    op.create_table('results',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('race_id', sa.Integer(), nullable=False),
    sa.Column('driver_id', sa.String(length=64), nullable=False),
    sa.Column('constructor_id', sa.String(length=64), nullable=False),
    sa.Column('grid', sa.Integer(), nullable=False),
    sa.Column('finish_position', sa.Integer(), nullable=False),
    sa.Column('position_text', sa.String(length=8), nullable=False),
    sa.Column('quali_position', sa.Integer(), nullable=True),
    sa.Column('quali_pts', sa.Float(), nullable=False),
    sa.Column('sprint_pts', sa.Float(), nullable=False),
    sa.Column('race_pts', sa.Float(), nullable=False),
    sa.Column('fantasy_points', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['constructor_id'], ['constructors.constructor_id'], ),
    sa.ForeignKeyConstraint(['driver_id'], ['drivers.driver_id'], ),
    sa.ForeignKeyConstraint(['race_id'], ['races.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('race_id', 'driver_id', name='uq_results_race_driver')
    )
    with op.batch_alter_table('results', schema=None) as batch_op:
        batch_op.create_index('ix_results_driver', ['driver_id'], unique=False)




def downgrade() -> None:

    with op.batch_alter_table('results', schema=None) as batch_op:
        batch_op.drop_index('ix_results_driver')

    op.drop_table('results')
    op.drop_table('driver_prices')
    op.drop_table('constructor_prices')
    op.drop_table('races')
    op.drop_table('ingestion_runs')
    op.drop_table('drivers')
    op.drop_table('constructors')

