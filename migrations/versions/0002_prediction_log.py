"""prediction log

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table('prediction_log',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('race_id', sa.Integer(), nullable=False),
    sa.Column('driver_id', sa.String(length=64), nullable=False),
    sa.Column('constructor_id', sa.String(length=64), nullable=False),
    sa.Column('model', sa.String(length=32), nullable=False),
    sa.Column('trained_through', sa.String(length=16), nullable=False),
    sa.Column('predicted_points', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['constructor_id'], ['constructors.constructor_id'], ),
    sa.ForeignKeyConstraint(['driver_id'], ['drivers.driver_id'], ),
    sa.ForeignKeyConstraint(['race_id'], ['races.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('race_id', 'driver_id', 'model', 'trained_through', name='uq_prediction_log')
    )
    op.create_table('team_recommendation_log',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('race_id', sa.Integer(), nullable=False),
    sa.Column('model', sa.String(length=32), nullable=False),
    sa.Column('trained_through', sa.String(length=16), nullable=False),
    sa.Column('budget', sa.Float(), nullable=False),
    sa.Column('price_source', sa.String(length=64), nullable=False),
    sa.Column('expected_points', sa.Float(), nullable=False),
    sa.Column('payload', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['race_id'], ['races.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('race_id', 'model', 'trained_through', 'budget', 'price_source', name='uq_team_recommendation')
    )



def downgrade() -> None:

    op.drop_table('team_recommendation_log')
    op.drop_table('prediction_log')

