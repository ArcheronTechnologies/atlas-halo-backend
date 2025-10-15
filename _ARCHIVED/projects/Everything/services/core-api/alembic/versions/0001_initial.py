from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'companies',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('type', sa.String(length=32), nullable=True),
        sa.Column('website', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'components',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('manufacturer_part_number', sa.String(length=100), nullable=False),
        sa.Column('manufacturer_id', sa.String(length=36), sa.ForeignKey('companies.id'), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('subcategory', sa.String(length=100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('datasheet_url', sa.String(length=500), nullable=True),
        sa.Column('lifecycle_status', sa.String(length=32), nullable=True),
        sa.Column('rohs_compliant', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_components_mpn', 'components', ['manufacturer_part_number'])

    op.create_table(
        'rfqs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('customer_id', sa.String(length=36), sa.ForeignKey('companies.id'), nullable=False),
        sa.Column('rfq_number', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('required_date', sa.Date(), nullable=True),
        sa.Column('total_value', sa.Float(), nullable=True),
        sa.Column('currency', sa.String(length=3), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'rfq_items',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('rfq_id', sa.String(length=36), sa.ForeignKey('rfqs.id'), nullable=False),
        sa.Column('component_id', sa.String(length=36), sa.ForeignKey('components.id'), nullable=True),
        sa.Column('customer_part_number', sa.String(length=100), nullable=True),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('target_price', sa.Float(), nullable=True),
        sa.Column('lead_time_weeks', sa.Integer(), nullable=True),
        sa.Column('packaging', sa.String(length=50), nullable=True),
        sa.Column('date_code_requirement', sa.String(length=50), nullable=True),
    )

    op.create_table(
        'rfq_quotes',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('rfq_id', sa.String(length=36), sa.ForeignKey('rfqs.id'), nullable=False),
        sa.Column('payload', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('rfq_quotes')
    op.drop_table('rfq_items')
    op.drop_table('rfqs')
    op.drop_index('ix_components_mpn', table_name='components')
    op.drop_table('components')
    op.drop_table('companies')

