"""SQLite storage: uploaded images, predictions and retraining jobs."""

import os
import sqlite3
import time

DB_PATH = os.environ.get(
    "BEAN_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "state", "app_data.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    image BLOB NOT NULL,
    size_bytes INTEGER NOT NULL,
    uploaded_at REAL NOT NULL,
    retrain_job_id TEXT              -- NULL until a retrain job uses the image
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    latency_ms REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS retrain_jobs (
    job_id TEXT PRIMARY KEY,
    trigger TEXT NOT NULL,           -- manual or auto
    status TEXT NOT NULL,            -- running, completed or failed
    stage TEXT,
    log TEXT,
    started_at REAL NOT NULL,
    finished_at REAL,
    new_images INTEGER,
    replay_images INTEGER,
    epochs_run INTEGER,
    old_accuracy REAL,
    old_f1 REAL,
    new_accuracy REAL,
    new_f1 REAL,
    promoted INTEGER,
    model_version TEXT,
    error TEXT
);
"""


def run(sql, args=(), fetch=False):
    """Run one SQL statement. With fetch=True, return the rows as dicts."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, args).fetchall()
        conn.commit()
        return [dict(r) for r in rows] if fetch else None
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.close()
    # A job still "running" at startup was cut off by a restart.
    run("UPDATE retrain_jobs SET status='failed', stage='failed', "
        "error='server restarted during the job' WHERE status='running'")


# ---------- uploads ----------

def add_upload(class_name, filename, sha256, image_bytes):
    """Store one image. Returns False if it was already uploaded."""
    try:
        run("INSERT INTO uploads (class_name, filename, sha256, image, size_bytes, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (class_name, filename, sha256, image_bytes, len(image_bytes), time.time()))
        return True
    except sqlite3.IntegrityError:
        return False


def pending_uploads():
    """Images not used by a retrain yet, as (id, class_name, bytes)."""
    rows = run("SELECT id, class_name, image FROM uploads WHERE retrain_job_id IS NULL",
               fetch=True)
    return [(r["id"], r["class_name"], bytes(r["image"])) for r in rows]


def count_pending_uploads():
    rows = run("SELECT class_name, COUNT(*) AS n FROM uploads "
               "WHERE retrain_job_id IS NULL GROUP BY class_name", fetch=True)
    return {r["class_name"]: r["n"] for r in rows}


def mark_uploads_used(ids, job_id):
    for upload_id in ids:
        run("UPDATE uploads SET retrain_job_id=? WHERE id=?", (job_id, upload_id))


def upload_counts_by_class():
    rows = run("SELECT class_name, COUNT(*) AS n FROM uploads GROUP BY class_name", fetch=True)
    return {r["class_name"]: r["n"] for r in rows}


def recent_uploads(limit=50):
    return run("SELECT id, class_name, filename, size_bytes, uploaded_at, retrain_job_id "
               "FROM uploads ORDER BY id DESC LIMIT ?", (limit,), fetch=True)


# ---------- predictions ----------

def log_prediction(predicted_class, confidence, latency_ms):
    run("INSERT INTO predictions (predicted_class, confidence, latency_ms, created_at) "
        "VALUES (?, ?, ?, ?)", (predicted_class, confidence, latency_ms, time.time()))


def prediction_stats():
    total = run("SELECT COUNT(*) AS n, AVG(latency_ms) AS latency FROM predictions", fetch=True)[0]
    by_class = run("SELECT predicted_class, COUNT(*) AS n FROM predictions "
                   "GROUP BY predicted_class", fetch=True)
    last_hour = run("SELECT COUNT(*) AS n FROM predictions WHERE created_at > ?",
                    (time.time() - 3600,), fetch=True)[0]["n"]
    recent = run("SELECT predicted_class, confidence, latency_ms, created_at FROM predictions "
                 "ORDER BY id DESC LIMIT 15", fetch=True)
    return {
        "total": total["n"],
        "avg_latency_ms": round(total["latency"], 1) if total["latency"] else None,
        "last_hour": last_hour,
        "by_class": {r["predicted_class"]: r["n"] for r in by_class},
        "recent": recent,
    }


# ---------- retraining jobs ----------

def start_job(job_id, trigger):
    run("INSERT INTO retrain_jobs (job_id, trigger, status, started_at) "
        "VALUES (?, ?, 'running', ?)", (job_id, trigger, time.time()))


def update_job(job_id, **fields):
    """Set any columns of a job, e.g. update_job(id, stage='training')."""
    columns = ", ".join(f"{name}=?" for name in fields)
    run(f"UPDATE retrain_jobs SET {columns} WHERE job_id=?", (*fields.values(), job_id))


def running_job():
    rows = run("SELECT job_id FROM retrain_jobs WHERE status='running'", fetch=True)
    return rows[0]["job_id"] if rows else None


def latest_job():
    rows = run("SELECT * FROM retrain_jobs ORDER BY started_at DESC LIMIT 1", fetch=True)
    if not rows:
        return None
    job = rows[0]
    job["log"] = job["log"].split("\n") if job["log"] else []
    return job


def job_history(limit=20):
    return run("SELECT job_id, trigger, status, started_at, finished_at, new_images, "
               "replay_images, epochs_run, old_accuracy, old_f1, new_accuracy, new_f1, "
               "promoted, model_version, error "
               "FROM retrain_jobs ORDER BY started_at DESC LIMIT ?", (limit,), fetch=True)
