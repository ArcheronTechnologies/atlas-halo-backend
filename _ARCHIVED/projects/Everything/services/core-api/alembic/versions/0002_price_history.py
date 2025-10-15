from alembic import op
import sqlalchemy as sa

revision = '0002_price_history'
down_revision = '0001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'price_history',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('component_id', sa.String(length=36), sa.ForeignKey('components.id'), nullable=False),
        sa.Column('supplier_id', sa.String(length=36), sa.ForeignKey('companies.id'), nullable=False),
        sa.Column('quantity_break', sa.Integer(), nullable=False),
        sa.Column('unit_price', sa.Float(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('valid_from', sa.Date(), nullable=True),
        sa.Column('valid_until', sa.Date(), nullable=True),
        sa.Column('source_type', sa.String(length=32), nullable=True),
        sa.Column('source_reference', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_price_history_component', 'price_history', ['component_id'])
    op.create_index('ix_price_history_supplier', 'price_history', ['supplier_id'])


def downgrade() -> None:
    op.drop_index('ix_price_history_supplier', table_name='price_history')
    op.drop_index('ix_price_history_component', table_name='price_history')
    op.drop_table('price_history')

