from alembic import op
import sqlalchemy as sa

revision = '0004_rfq_quote_winner_role_permissions_audit'
down_revision = '0003_users_inventory_pos_misc'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_winner to rfq_quotes
    op.add_column('rfq_quotes', sa.Column('is_winner', sa.Boolean(), nullable=False, server_default=sa.sql.expression.false()))

    # Role permissions table
    op.create_table(
        'role_permissions',
        sa.Column('role_id', sa.String(length=36), sa.ForeignKey('roles.id'), primary_key=True),
        sa.Column('permission', sa.String(length=100), primary_key=True),
    )

    # Audit logs
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('entity_type', sa.String(length=50), nullable=False),
        sa.Column('entity_id', sa.String(length=36), nullable=False),
        sa.Column('action', sa.String(length=50), nullable=False),
        sa.Column('old_value', sa.Text(), nullable=True),
        sa.Column('new_value', sa.Text(), nullable=True),
        sa.Column('user_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('audit_logs')
    op.drop_table('role_permissions')
    op.drop_column('rfq_quotes', 'is_winner')

