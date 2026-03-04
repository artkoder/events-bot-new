"""Alembic migration: festival_source_channels

- add festival flags to telegram_source and vk_source
- seed known festival Telegram channels
"""
from alembic import op
import sqlalchemy as sa

revision = "20260215_festival_source_channels"
down_revision = "20260104_festival_parser_fields"
branch_labels = None
depends_on = None


FESTIVAL_TG_SOURCES = (
    ("open_fest", "Открытое море"),
    ("festkantata", "Кантата"),
    ("garazhka_kld", "Гаражка"),
)


def _seed_tg_sources(conn, dialect_name: str) -> None:
    if dialect_name == "postgresql":
        values_sql = ",\n".join(
            "(:u{idx}, true, :s{idx})".format(idx=idx)
            for idx, _ in enumerate(FESTIVAL_TG_SOURCES)
        )
        params: dict[str, str] = {}
        for idx, (username, series) in enumerate(FESTIVAL_TG_SOURCES):
            params[f"u{idx}"] = username
            params[f"s{idx}"] = series
        conn.execute(
            sa.text(
                """
                INSERT INTO telegram_source (username, festival_source, festival_series)
                VALUES
                """
                + values_sql
                + """
                ON CONFLICT (username) DO UPDATE
                SET festival_source = TRUE,
                    festival_series = COALESCE(telegram_source.festival_series, EXCLUDED.festival_series)
                """
            ),
            params,
        )
        return

    # SQLite / fallback
    for username, series in FESTIVAL_TG_SOURCES:
        conn.execute(
            sa.text(
                "INSERT OR IGNORE INTO telegram_source (username, festival_source, festival_series) "
                "VALUES (:username, 1, :series)"
            ),
            {"username": username, "series": series},
        )
        conn.execute(
            sa.text(
                "UPDATE telegram_source "
                "SET festival_source = 1, "
                "festival_series = COALESCE(festival_series, :series) "
                "WHERE username = :username"
            ),
            {"username": username, "series": series},
        )


def upgrade() -> None:
    with op.batch_alter_table("telegram_source", schema=None) as batch_op:
        batch_op.add_column(sa.Column("festival_source", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("festival_series", sa.String(), nullable=True))

    with op.batch_alter_table("vk_source", schema=None) as batch_op:
        batch_op.add_column(sa.Column("festival_source", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("festival_series", sa.String(), nullable=True))

    conn = op.get_bind()
    _seed_tg_sources(conn, conn.dialect.name)


def downgrade() -> None:
    with op.batch_alter_table("vk_source", schema=None) as batch_op:
        batch_op.drop_column("festival_series")
        batch_op.drop_column("festival_source")

    with op.batch_alter_table("telegram_source", schema=None) as batch_op:
        batch_op.drop_column("festival_series")
        batch_op.drop_column("festival_source")
