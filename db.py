"""
db.py — SQLite database layer for Medical Chatbot
==================================================

All database access goes through this module.
SQLite is built into Python — no server, no install needed.

DB file location: controlled by SQLITE_DB_PATH in .env
Default:          chat_history.db  (project root)

════════════════════════════════════════════════════════════
SCHEMA
════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────┐
│  sessions                                               │
│  session_id    TEXT  PK                                 │
│  user_id       TEXT  NOT NULL  (FK → main platform)     │
│  created_at    TEXT  NOT NULL  (ISO-8601 UTC)           │
│  last_disease  TEXT  DEFAULT ''                         │
│  last_severity TEXT  DEFAULT ''                         │
│                CRITICAL / URGENT / MODERATE / MILD      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  messages                                               │
│  id          INTEGER  PK AUTOINCREMENT                  │
│  session_id  TEXT     NOT NULL  FK → sessions           │
│  user_id     TEXT     NOT NULL                          │
│  role        TEXT     NOT NULL  'user' | 'assistant'    │
│  content     TEXT     NOT NULL                          │
│  ts          TEXT     NOT NULL  (ISO-8601 UTC)          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  user_profile        (Phase 2 — friend's advisor input) │
│  user_id          TEXT  PK                              │
│  age              INTEGER                               │
│  gender           TEXT      'Male' / 'Female' / 'Other' │
│  state            TEXT      Indian state / region       │
│  diet_preference  TEXT      'Vegetarian' / 'Non-veg' /  │
│                             'Vegan' / 'Eggetarian'      │
│  updated_at       TEXT  NOT NULL  (ISO-8601 UTC)        │
│                                                         │
│  Read by /api/cultural-advice to build friend's input   │
│  payload (Age/Gender/State/Diet preference fields).     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  nutrition_targets            (Phase 2 — already ready) │
│  id             INTEGER  PK AUTOINCREMENT               │
│  user_id        TEXT     NOT NULL                       │
│  session_id     TEXT     NOT NULL  FK → sessions        │
│  disease        TEXT     NOT NULL                       │
│  nutrition_json TEXT     NOT NULL  (JSON string)        │
│  created_at     TEXT     NOT NULL  (ISO-8601 UTC)       │
│                                                         │
│  nutrition_json structure:                              │
│  {                                                      │
│    "disease": "Influenza",                              │
│    "nutrients": {                                       │
│      "carbohydrates": {                                 │
│        "recommended": "50-55% (up vs normal)",          │
│        "direction":   "increase",                       │
│        "note":        "Energy needed for immune resp."  │
│      },                                                 │
│      "protein":  { ... },                               │
│      "fat":      { ... },                               │
│      "vitamins": { ... },                               │
│      "minerals": { ... },                               │
│      "water":    { ... },                               │
│      "fiber":    { ... }                                │
│    }                                                    │
│  }                                                      │
│                                                         │
│  Endpoint: GET /api/nutrition-targets/<user_id>         │
│  Purpose:  Friend's food recommendation system reads    │
│            this to suggest regional meals per disease   │
└─────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════
SEVERITY → UI BEHAVIOUR
════════════════════════════════════════════════════════════

  CRITICAL  →  Emergency modal (auto-opens) + live GPS
               hospital finder (browser geolocation)
               Call 108 button shown
               Home care table: NOT shown

  URGENT    →  Orange severity bar (clickable) in chat
               Home care table: shown
               No modal pop-up

  MODERATE  →  Yellow severity bar in chat
               Home care table: shown
               No modal

  MILD      →  Green severity bar in chat
               Home care table: shown
               No modal

════════════════════════════════════════════════════════════
PIPELINE → DB WRITE FLOW
════════════════════════════════════════════════════════════

  User message
       │
       v
  run_pipeline(query, user_id, session_id)
       │
       ├─ Step 3 (nutrition) ─► _store_nutrition_targets()
       │                              nutrition_targets table
       │
       └─ /get endpoint
               ├─ _append_msgs()        ► messages table
               └─ _update_session_meta() ► sessions table
                    (last_disease, last_severity)

════════════════════════════════════════════════════════════
PHASE 2 INTEGRATION (pending)
════════════════════════════════════════════════════════════

  1. Replace get_current_user() UUID stub → real JWT decode
  2. Sync or replace SQLite with main platform DB
     (swap get_db() below with your DB adapter)
  3. nutrition_targets already structured and ready
     for friend's food recommendation system

════════════════════════════════════════════════════════════
"""

import os
import sqlite3

# ── DB path from environment ──────────────────────────────────────────────────
DB_PATH = os.getenv("SQLITE_DB_PATH", "chat_history.db")


def get_db() -> sqlite3.Connection:
    """
    Return a new SQLite connection for the current request/thread.
    One connection per call — SQLite handles concurrency at file level.
    row_factory = sqlite3.Row allows dict-style access: row["column_name"]
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create all tables and indexes on first run.
    Safe to call on every startup — all statements use IF NOT EXISTS.
    Also migrates older DBs missing newer columns.
    """
    with get_db() as conn:

        # ── sessions ──────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                last_disease  TEXT DEFAULT '',
                last_severity TEXT DEFAULT ''
            )
        """)

        # ── messages ──────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                user_id     TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                ts          TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # ── nutrition_targets ─────────────────────────────────────────────────
        # Populated silently by run_pipeline() Step 3.
        # Exposed via GET /api/nutrition-targets/<user_id>
        # Read by friend's food recommendation system.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nutrition_targets (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        TEXT    NOT NULL,
                session_id     TEXT    NOT NULL,
                disease        TEXT    NOT NULL,
                nutrition_json TEXT    NOT NULL,
                created_at     TEXT    NOT NULL
            )
        """)

        # ── user_profile ──────────────────────────────────────────────────────
        # Holds Age / Gender / State / Diet preference for the cultural-advice
        # microservice (friend's HuggingFace healthAdvisor).
        # Filled via the profile form modal on first "Get Food Suggestions" click.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id          TEXT    PRIMARY KEY,
                age              INTEGER,
                gender           TEXT,
                state            TEXT,
                diet_preference  TEXT,
                updated_at       TEXT    NOT NULL
            )
        """)

        # ── indexes ───────────────────────────────────────────────────────────
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user     ON sessions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session  ON messages(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nutrition_user    ON nutrition_targets(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nutrition_session ON nutrition_targets(session_id)")

        # ── backward-compat migrations ─────────────────────────────────────────
        # Older DBs may not have last_disease / last_severity columns yet.
        for col in ["last_disease", "last_severity"]:
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT DEFAULT ''")
            except Exception:
                pass  # Column already exists — safe to ignore

    print(f"SQLite ready -> {DB_PATH}")
    print("   Tables: sessions | messages | nutrition_targets | user_profile")
