"""Database schema (shared by app.py and tmdb_import.py)."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user');
CREATE TABLE IF NOT EXISTS movies(id INTEGER PRIMARY KEY, title TEXT NOT NULL, year INTEGER,
    genres TEXT, director TEXT, description TEXT,
    keywords TEXT, cast_names TEXT, tmdb_id INTEGER, poster_path TEXT, tmdb_rating REAL, tmdb_votes INTEGER);
CREATE TABLE IF NOT EXISTS ratings(user_id INTEGER, movie_id INTEGER, rating INTEGER,
    PRIMARY KEY(user_id, movie_id));
CREATE TABLE IF NOT EXISTS watchlist(user_id INTEGER, movie_id INTEGER,
    PRIMARY KEY(user_id, movie_id));
"""

NEW_MOVIE_COLUMNS = {"keywords": "TEXT", "cast_names": "TEXT", "tmdb_id": "INTEGER",
                     "poster_path": "TEXT", "tmdb_rating": "REAL", "tmdb_votes": "INTEGER"}


def ensure_schema(con):
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(movies)")}
    for col, typ in NEW_MOVIE_COLUMNS.items():          # upgrade databases made by older versions
        if col not in have:
            con.execute(f"ALTER TABLE movies ADD COLUMN {col} {typ}")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_movies_tmdb ON movies(tmdb_id)")
    con.commit()


def ensure_default_users(con):
    """Creates the admin and demo accounts if they do not exist yet."""
    from werkzeug.security import generate_password_hash
    for name, pw, role in (("admin", "admin123", "admin"), ("demo", "demo123", "user")):
        if not con.execute("SELECT 1 FROM users WHERE username=?", (name,)).fetchone():
            con.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                        (name, generate_password_hash(pw), role))
    con.commit()
