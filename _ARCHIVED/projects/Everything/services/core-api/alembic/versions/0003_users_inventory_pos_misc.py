from alembic import op
import sqlalchemy as sa

revision = '0003_users_inventory_pos_misc'
down_revision = '0002_price_history'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('email', sa.String(length=255), unique=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_table(
        'roles',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('name', sa.String(length=50), unique=True, nullable=False),
    )
    op.create_table(
        'user_roles',
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('role_id', sa.String(length=36), sa.ForeignKey('roles.id'), primary_key=True),
    )
    op.create_table(
        'api_keys',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('key_hash', sa.String(length=128), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default=sa.sql.expression.true()),
    )
    op.create_table(
        'inventory',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('component_id', sa.String(length=36), sa.ForeignKey('components.id'), nullable=False),
        sa.Column('location', sa.String(length=100), nullable=True),
        sa.Column('quantity_available', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('quantity_reserved', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cost_per_unit', sa.Float(), nullable=True),
        sa.Column('date_code', sa.String(length=20), nullable=True),
        sa.Column('lot_code', sa.String(length=50), nullable=True),
        sa.Column('expiry_date', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_table(
        'purchase_orders',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('supplier_id', sa.String(length=36), sa.ForeignKey('companies.id'), nullable=False),
        sa.Column('po_number', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='draft'),
        sa.Column('total_value', sa.Float(), nullable=True),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('payment_terms', sa.String(length=100), nullable=True),
        sa.Column('delivery_address', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_table(
        'po_items',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('po_id', sa.String(length=36), sa.ForeignKey('purchase_orders.id'), nullable=False),
        sa.Column('component_id', sa.String(length=36), sa.ForeignKey('components.id'), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('unit_price', sa.Float(), nullable=True),
        sa.Column('lead_time_weeks', sa.Integer(), nullable=True),
        sa.Column('manufacturer_lot_code', sa.String(length=50), nullable=True),
        sa.Column('date_code', sa.String(length=20), nullable=True),
        sa.Column('packaging', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_table(
        'processed_emails',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('message_id', sa.String(length=255), unique=True, nullable=False),
        sa.Column('sender_email', sa.String(length=255), nullable=True),
        sa.Column('subject', sa.Text(), nullable=True),
        sa.Column('body_text', sa.Text(), nullable=True),
        sa.Column('body_html', sa.Text(), nullable=True),
        sa.Column('received_at', sa.DateTime(), nullable=True),
        sa.Column('classification', sa.String(length=50), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('extracted_entities', sa.Text(), nullable=True),
        sa.Column('attachments', sa.Text(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
    )
    op.create_table(
        'market_data',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('component_id', sa.String(length=36), sa.ForeignKey('components.id'), nullable=True),
        sa.Column('data_source', sa.String(length=100), nullable=True),
        sa.Column('metric_type', sa.String(length=50), nullable=True),
        sa.Column('metric_value', sa.Float(), nullable=True),
        sa.Column('unit', sa.String(length=20), nullable=True),
        sa.Column('collection_date', sa.Date(), nullable=True),
        sa.Column('source_url', sa.String(length=500), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('market_data')
    op.drop_table('processed_emails')
    op.drop_table('po_items')
    op.drop_table('purchase_orders')
    op.drop_table('inventory')
    op.drop_table('api_keys')
    op.drop_table('user_roles')
    op.drop_table('roles')
    op.drop_table('users')

