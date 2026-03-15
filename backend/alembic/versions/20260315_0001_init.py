"""init schema

Revision ID: 20260315_0001
Revises: 
Create Date: 2026-03-15 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260315_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("weight_path", sa.Text(), nullable=False),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_models_key", "models", ["key"], unique=True)

    op.create_table(
        "tasks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("model_key", sa.Text(), nullable=True),
        sa.Column("score_thr", sa.Float(), nullable=False),
        sa.Column("input_count", sa.Integer(), nullable=False),
        sa.Column("done_count", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_tasks_status", "tasks", ["status"], unique=False)

    op.create_table(
        "results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("image_name", sa.String(length=255), nullable=False),
        sa.Column("source_model", sa.String(length=64), nullable=False),
        sa.Column("is_fused", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("bbox_x2", sa.Float(), nullable=False),
        sa.Column("bbox_y2", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=False),
    )
    op.create_index("ix_results_task_id", "results", ["task_id"], unique=False)
    op.create_index("ix_results_image_name", "results", ["image_name"], unique=False)
    op.create_index("ix_results_source_model", "results", ["source_model"], unique=False)

    op.create_table(
        "task_files",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.Integer(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
    )
    op.create_index("ix_task_files_task_id", "task_files", ["task_id"], unique=False)
    op.create_index("ix_task_files_kind", "task_files", ["kind"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_task_files_kind", table_name="task_files")
    op.drop_index("ix_task_files_task_id", table_name="task_files")
    op.drop_table("task_files")

    op.drop_index("ix_results_source_model", table_name="results")
    op.drop_index("ix_results_image_name", table_name="results")
    op.drop_index("ix_results_task_id", table_name="results")
    op.drop_table("results")

    op.drop_index("ix_tasks_status", table_name="tasks")
    op.drop_table("tasks")

    op.drop_index("ix_models_key", table_name="models")
    op.drop_table("models")
