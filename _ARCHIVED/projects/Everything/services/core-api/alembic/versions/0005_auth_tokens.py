from alembic import op
import sqlalchemy as sa

revision = '0005_auth_tokens'
down_revision = '0004_rfq_quote_winner_role_permissions_audit'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('token_hash', sa.String(length=128), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('revoked', sa.Boolean(), nullable=False, server_default=sa.sql.expression.false()),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_refresh_tokens_token_hash', 'refresh_tokens', ['token_hash'])

    op.create_table(
        'revoked_jtis',
        sa.Column('jti', sa.String(length=64), primary_key=True),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('revoked_jtis')
    op.drop_index('ix_refresh_tokens_token_hash', table_name='refresh_tokens')
    op.drop_table('refresh_tokens')

