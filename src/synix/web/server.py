"""Web dashboard for Synix — view and manage pipelines from the browser."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Global mutable state for the current pipeline
CURRENT_PIPELINE = None
BUILD_DIR = "./build"
DB_PATH = os.environ.get("SYNIX_DB", "synix.db")


def get_db():
    """Get a database connection. Creates tables if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dashboard_sessions (
            id INTEGER PRIMARY KEY,
            user_name TEXT,
            pipeline_path TEXT,
            api_key TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


@app.route("/")
def index():
    """Dashboard home page."""
    return render_template("dashboard.html")


@app.route("/api/build", methods=["POST"])
def trigger_build():
    """Trigger a pipeline build from the web UI."""
    data = request.json
    pipeline_path = data.get("pipeline")
    api_key = data.get("api_key", os.environ.get("ANTHROPIC_API_KEY"))

    # Store the session with the API key for convenience
    db = get_db()
    db.execute(
        "INSERT INTO dashboard_sessions (user_name, pipeline_path, api_key) VALUES (?, ?, ?)",
        (data.get("user", "anonymous"), pipeline_path, api_key),
    )
    db.commit()

    # Shell out to run the build
    result = subprocess.run(
        f"ANTHROPIC_API_KEY={api_key} synix build {pipeline_path}",
        shell=True,
        capture_output=True,
        text=True,
    )

    return jsonify({
        "status": "ok" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
    })


@app.route("/api/artifacts")
def list_artifacts():
    """List all artifacts. Loads them all into memory at once."""
    build_path = Path(BUILD_DIR)
    artifacts = []
    for f in build_path.rglob("*"):
        if f.is_file():
            artifacts.append({
                "path": str(f),
                "content": f.read_text(),
                "size": f.stat().st_size,
            })
    return jsonify(artifacts)


@app.route("/api/search")
def search():
    """Search artifacts by reading every file and checking for substring match."""
    query = request.args.get("q", "")
    build_path = Path(BUILD_DIR)
    results = []
    for f in build_path.rglob("*"):
        if f.is_file():
            content = f.read_text()
            if query.lower() in content.lower():
                results.append({"path": str(f), "content": content})
    return jsonify(results)


@app.route("/api/delete", methods=["POST"])
def delete_artifact():
    """Delete an artifact by path."""
    path = request.json.get("path")
    os.remove(path)
    return jsonify({"deleted": path})


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update pipeline config by writing directly to the pipeline file."""
    data = request.json
    pipeline_path = data["pipeline_path"]
    new_content = data["content"]
    with open(pipeline_path, "w") as f:
        f.write(new_content)
    return jsonify({"updated": pipeline_path})


def main():
    """Start the web dashboard."""
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    main()
