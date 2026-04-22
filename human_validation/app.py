"""
ConSynth-X Human Perceptual Validation Platform
================================================
Flask app for evaluating synthetic construction image quality
through human perception studies (Paper 1: Nature Scientific Data).

Three tasks:
  1. Turing Test   — Real vs Synthetic classification
  2. Realism Rating — Mean Opinion Score (1-5)
  3. Condition Recognition — Identify weather/lighting condition
"""

import os
import json
import csv
import io
import random
import sqlite3
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, g, flash, Response
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "consynth-x-validation-2026")

DATABASE = os.path.join(os.path.dirname(__file__), "validation.db")

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
CONDITIONS = [
    "original", "weather_style_rain_0", "weather_style_rain_1",
    "weather_style_rain_2", "weather_style_snow_0", "weather_style_snow_1",
    "weather_style_snow_2", "night", "night_rain", "night_snow", "small",
]

RECOGNITION_CHOICES = [
    ("clear", "Clear / Original"),
    ("rain", "Rain"),
    ("snow", "Snow"),
    ("night", "Night"),
    ("night_rain", "Night + Rain"),
    ("night_snow", "Night + Snow"),
    ("fog", "Fog"),
    ("small", "Small / Far objects"),
]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DATABASE)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT UNIQUE NOT NULL,
        email       TEXT,
        is_admin    INTEGER DEFAULT 0,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS images (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        filename    TEXT NOT NULL,
        condition   TEXT NOT NULL,
        source      TEXT NOT NULL,       -- 'synthetic' or 'real'
        dataset     TEXT DEFAULT '',     -- 'construction_site', 'soda', 'acdc', etc.
        UNIQUE(filename, condition)
    );

    CREATE TABLE IF NOT EXISTS turing_responses (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL REFERENCES users(id),
        image_id    INTEGER NOT NULL REFERENCES images(id),
        answer      TEXT NOT NULL,       -- 'real' or 'synthetic'
        response_ms INTEGER,             -- reaction time in ms
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS realism_responses (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL REFERENCES users(id),
        image_id    INTEGER NOT NULL REFERENCES images(id),
        score       INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
        response_ms INTEGER,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS recognition_responses (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL REFERENCES users(id),
        image_id    INTEGER NOT NULL REFERENCES images(id),
        answer      TEXT NOT NULL,
        response_ms INTEGER,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    -- Create default admin user
    INSERT OR IGNORE INTO users (username, email, is_admin)
    VALUES ('admin', 'vduong1@kent.edu', 1);
    """)
    db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if not session.get("is_admin"):
            flash("Admin access required.")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        if not username:
            flash("Please enter a username.")
            return render_template("login.html")

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if user is None:
            db.execute(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                (username, email),
            )
            db.commit()
            user = db.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["is_admin"] = bool(user["is_admin"])
        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------
@app.route("/")
@login_required
def index():
    db = get_db()
    total_images = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    user_id = session["user_id"]
    turing_done = db.execute(
        "SELECT COUNT(*) FROM turing_responses WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    realism_done = db.execute(
        "SELECT COUNT(*) FROM realism_responses WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    recognition_done = db.execute(
        "SELECT COUNT(*) FROM recognition_responses WHERE user_id = ?", (user_id,)
    ).fetchone()[0]

    # Count images per task type
    turing_total = db.execute(
        "SELECT COUNT(*) FROM images"
    ).fetchone()[0]
    synthetic_total = db.execute(
        "SELECT COUNT(*) FROM images WHERE source = 'synthetic'"
    ).fetchone()[0]

    return render_template(
        "index.html",
        total_images=total_images,
        turing_done=turing_done, turing_total=turing_total,
        realism_done=realism_done, realism_total=synthetic_total,
        recognition_done=recognition_done, recognition_total=synthetic_total,
    )


@app.route("/tutorial")
@login_required
def tutorial():
    return render_template("tutorial.html")


# ---------------------------------------------------------------------------
# Routes — Task 1: Turing Test
# ---------------------------------------------------------------------------
@app.route("/task/turing")
@login_required
def task_turing():
    db = get_db()
    user_id = session["user_id"]

    # Get an image the user hasn't rated yet (random order)
    image = db.execute("""
        SELECT * FROM images
        WHERE id NOT IN (
            SELECT image_id FROM turing_responses WHERE user_id = ?
        )
        ORDER BY RANDOM() LIMIT 1
    """, (user_id,)).fetchone()

    if image is None:
        return render_template("task_complete.html", task="Turing Test")

    done = db.execute(
        "SELECT COUNT(*) FROM turing_responses WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    return render_template(
        "task_turing.html", image=image, done=done, total=total
    )


@app.route("/api/turing", methods=["POST"])
@login_required
def api_turing():
    data = request.get_json()
    db = get_db()
    db.execute(
        "INSERT INTO turing_responses (user_id, image_id, answer, response_ms) VALUES (?, ?, ?, ?)",
        (session["user_id"], data["image_id"], data["answer"], data.get("response_ms")),
    )
    db.commit()
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Routes — Task 2: Realism Rating (MOS)
# ---------------------------------------------------------------------------
@app.route("/task/realism")
@login_required
def task_realism():
    db = get_db()
    user_id = session["user_id"]

    image = db.execute("""
        SELECT * FROM images
        WHERE source = 'synthetic'
        AND id NOT IN (
            SELECT image_id FROM realism_responses WHERE user_id = ?
        )
        ORDER BY RANDOM() LIMIT 1
    """, (user_id,)).fetchone()

    if image is None:
        return render_template("task_complete.html", task="Realism Rating")

    done = db.execute(
        "SELECT COUNT(*) FROM realism_responses WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    total = db.execute(
        "SELECT COUNT(*) FROM images WHERE source = 'synthetic'"
    ).fetchone()[0]

    return render_template(
        "task_realism.html", image=image, done=done, total=total
    )


@app.route("/api/realism", methods=["POST"])
@login_required
def api_realism():
    data = request.get_json()
    score = int(data["score"])
    if not 1 <= score <= 5:
        return jsonify({"error": "Score must be 1-5"}), 400
    db = get_db()
    db.execute(
        "INSERT INTO realism_responses (user_id, image_id, score, response_ms) VALUES (?, ?, ?, ?)",
        (session["user_id"], data["image_id"], score, data.get("response_ms")),
    )
    db.commit()
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Routes — Task 3: Condition Recognition
# ---------------------------------------------------------------------------
@app.route("/task/recognition")
@login_required
def task_recognition():
    db = get_db()
    user_id = session["user_id"]

    image = db.execute("""
        SELECT * FROM images
        WHERE source = 'synthetic'
        AND id NOT IN (
            SELECT image_id FROM recognition_responses WHERE user_id = ?
        )
        ORDER BY RANDOM() LIMIT 1
    """, (user_id,)).fetchone()

    if image is None:
        return render_template("task_complete.html", task="Condition Recognition")

    done = db.execute(
        "SELECT COUNT(*) FROM recognition_responses WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    total = db.execute(
        "SELECT COUNT(*) FROM images WHERE source = 'synthetic'"
    ).fetchone()[0]

    return render_template(
        "task_recognition.html",
        image=image, done=done, total=total,
        choices=RECOGNITION_CHOICES,
    )


@app.route("/api/recognition", methods=["POST"])
@login_required
def api_recognition():
    data = request.get_json()
    db = get_db()
    db.execute(
        "INSERT INTO recognition_responses (user_id, image_id, answer, response_ms) VALUES (?, ?, ?, ?)",
        (session["user_id"], data["image_id"], data["answer"], data.get("response_ms")),
    )
    db.commit()
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Routes — Admin Dashboard
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@admin_required
def dashboard():
    db = get_db()

    # Summary stats
    stats = {
        "total_images": db.execute("SELECT COUNT(*) FROM images").fetchone()[0],
        "total_users": db.execute("SELECT COUNT(*) FROM users WHERE is_admin = 0").fetchone()[0],
        "turing_responses": db.execute("SELECT COUNT(*) FROM turing_responses").fetchone()[0],
        "realism_responses": db.execute("SELECT COUNT(*) FROM realism_responses").fetchone()[0],
        "recognition_responses": db.execute("SELECT COUNT(*) FROM recognition_responses").fetchone()[0],
    }

    # Per-user progress
    users = db.execute("""
        SELECT u.username, u.email,
            (SELECT COUNT(*) FROM turing_responses WHERE user_id = u.id) as turing,
            (SELECT COUNT(*) FROM realism_responses WHERE user_id = u.id) as realism,
            (SELECT COUNT(*) FROM recognition_responses WHERE user_id = u.id) as recognition
        FROM users u WHERE u.is_admin = 0
        ORDER BY u.username
    """).fetchall()

    # Turing test: fooling rate per condition
    fooling = db.execute("""
        SELECT i.condition, i.source,
            COUNT(*) as total,
            SUM(CASE WHEN
                (i.source = 'synthetic' AND tr.answer = 'real') OR
                (i.source = 'real' AND tr.answer = 'real')
            THEN 1 ELSE 0 END) as fooled
        FROM turing_responses tr
        JOIN images i ON i.id = tr.image_id
        GROUP BY i.condition, i.source
    """).fetchall()

    # MOS per condition
    mos = db.execute("""
        SELECT i.condition,
            ROUND(AVG(rr.score), 2) as mean_score,
            ROUND(AVG(rr.score * rr.score) - AVG(rr.score) * AVG(rr.score), 2) as variance,
            COUNT(*) as n
        FROM realism_responses rr
        JOIN images i ON i.id = rr.image_id
        GROUP BY i.condition
    """).fetchall()

    # Recognition accuracy per condition
    recognition = db.execute("""
        SELECT i.condition, rr.answer, COUNT(*) as cnt
        FROM recognition_responses rr
        JOIN images i ON i.id = rr.image_id
        GROUP BY i.condition, rr.answer
    """).fetchall()

    return render_template(
        "dashboard.html",
        stats=stats, users=users, fooling=fooling,
        mos=mos, recognition=recognition,
    )


@app.route("/api/export/<task>")
@admin_required
def export_csv(task):
    db = get_db()

    if task == "turing":
        rows = db.execute("""
            SELECT u.username, i.filename, i.condition, i.source,
                   tr.answer, tr.response_ms, tr.created_at
            FROM turing_responses tr
            JOIN users u ON u.id = tr.user_id
            JOIN images i ON i.id = tr.image_id
            ORDER BY tr.created_at
        """).fetchall()
        headers = ["username", "filename", "condition", "source",
                    "answer", "response_ms", "created_at"]
    elif task == "realism":
        rows = db.execute("""
            SELECT u.username, i.filename, i.condition,
                   rr.score, rr.response_ms, rr.created_at
            FROM realism_responses rr
            JOIN users u ON u.id = rr.user_id
            JOIN images i ON i.id = rr.image_id
            ORDER BY rr.created_at
        """).fetchall()
        headers = ["username", "filename", "condition",
                    "score", "response_ms", "created_at"]
    elif task == "recognition":
        rows = db.execute("""
            SELECT u.username, i.filename, i.condition,
                   rr.answer, rr.response_ms, rr.created_at
            FROM recognition_responses rr
            JOIN users u ON u.id = rr.user_id
            JOIN images i ON i.id = rr.image_id
            ORDER BY rr.created_at
        """).fetchall()
        headers = ["username", "filename", "condition",
                    "answer", "response_ms", "created_at"]
    else:
        return "Invalid task", 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(list(row))

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=consynth_x_{task}_{datetime.now():%Y%m%d}.csv"},
    )


# ---------------------------------------------------------------------------
# Image management API (admin)
# ---------------------------------------------------------------------------
@app.route("/api/images/load", methods=["POST"])
@admin_required
def load_images():
    """Scan static/images/ directory and register images in database."""
    img_dir = os.path.join(app.static_folder, "images")
    if not os.path.isdir(img_dir):
        return jsonify({"error": "static/images/ not found"}), 400

    db = get_db()
    count = 0
    for condition_dir in os.listdir(img_dir):
        cond_path = os.path.join(img_dir, condition_dir)
        if not os.path.isdir(cond_path):
            continue

        # Determine source
        source = "real" if condition_dir in ("real_weather", "real_reference") else "synthetic"
        condition = condition_dir

        for fname in os.listdir(cond_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                rel_path = f"images/{condition_dir}/{fname}"
                try:
                    db.execute(
                        "INSERT OR IGNORE INTO images (filename, condition, source, dataset) VALUES (?, ?, ?, ?)",
                        (rel_path, condition, source, ""),
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass

    db.commit()
    return jsonify({"status": "ok", "loaded": count})


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
with app.app_context():
    init_db()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
