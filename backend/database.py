"""
SQLite Database Layer for Bitcoin Forensics Job Management
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config import DB_PATH as DEFAULT_DB_PATH


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create and return a database connection with dictionary-like row factory."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the jobs table if it doesn't exist."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_stage TEXT,
                    progress INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT,
                    raw_dir TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    summary TEXT
                )
                """
            )
            # Add progress column to existing table if missing
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(jobs)")
            columns = [info[1] for info in cursor.fetchall()]
            if "progress" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN progress INTEGER DEFAULT 0")

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at 
                ON jobs(created_at DESC)
                """
            )
    finally:
        conn.close()


def create_job(
    job_id: str,
    raw_dir: str,
    output_dir: str,
    status: str = "processing",
    current_stage: str = "queued",
    progress: int = 0,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create a new job record."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    now = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, current_stage, progress, created_at,
                    completed_at, error_message, raw_dir, output_dir, summary
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, NULL)
                """,
                (job_id, status, current_stage, progress, now, raw_dir, output_dir),
            )
        return {
            "job_id": job_id,
            "status": status,
            "current_stage": current_stage,
            "progress": progress,
            "created_at": now,
            "completed_at": None,
            "error_message": None,
            "raw_dir": raw_dir,
            "output_dir": output_dir,
            "summary": None,
        }
    finally:
        conn.close()


def update_job(
    job_id: str,
    status: Optional[str] = None,
    current_stage: Optional[str] = None,
    progress: Optional[int] = None,
    completed_at: Optional[str] = None,
    error_message: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Update job fields."""
    conn = get_db_connection(db_path)
    try:
        updates = []
        params = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if current_stage is not None:
            updates.append("current_stage = ?")
            params.append(current_stage)
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if summary is not None:
            updates.append("summary = ?")
            params.append(json.dumps(summary))

        if updates:
            params.append(job_id)
            query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
            with conn:
                conn.execute(query, params)
    finally:
        conn.close()


def get_job(job_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve job details by job_id."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        if data.get("summary"):
            try:
                data["summary"] = json.loads(data["summary"])
            except Exception:
                pass
        return data
    finally:
        conn.close()


def list_jobs(
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List all jobs ordered by created_at DESC."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            data = dict(row)
            if data.get("summary"):
                try:
                    data["summary"] = json.loads(data["summary"])
                except Exception:
                    pass
            results.append(data)
        return results
    finally:
        conn.close()
