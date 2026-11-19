import os, sqlite3
from functools import wraps
from flask import Flask, g, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from seed_data import MOVIES, synthetic_ratings
from schema import ensure_schema, ensure_default_users
import threading, time
import tmdb_import
import ml_engine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
DB = os.path.join(os.path.dirname(__file__), "movies.db")


# ---------- database ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()


def init_db():
    con = sqlite3.connect(DB)
    ensure_schema(con)
    ensure_default_users(con)
    if not con.execute("SELECT 1 FROM movies").fetchone():     # first run: built-in sample data
        con.executemany("INSERT INTO movies(title,year,genres,director,description) VALUES(?,?,?,?,?)", MOVIES)
        demo_id = con.execute("SELECT id FROM users WHERE username='demo'").fetchone()[0]
        # a few demo ratings so recommendations work out of the box
        con.executemany("INSERT INTO ratings VALUES(?,?,?)", [(demo_id, 1, 5), (demo_id, 2, 5), (demo_id, 4, 4), (demo_id, 5, 2)])
        # synthetic users with different tastes -> training data for the ML model
        shared_hash = generate_password_hash("password")
        for name, _, rated in synthetic_ratings():
            uid = con.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                              (name, shared_hash, "user")).lastrowid
            con.executemany("INSERT INTO ratings VALUES(?,?,?)", [(uid, m, r) for m, r in rated])
    con.commit()
    con.close()


# ---------- auth helpers ----------
def current_user():
    uid = session.get("uid")
    if not uid:
        return None
    return db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


TMDB_IMG = "https://image.tmdb.org/t/p/"


def poster_url(m, size="w342"):
    """TMDB poster URL for a movie (dict or sqlite row), or None."""
    pp = m["poster_path"] if "poster_path" in m.keys() else None
    return f"{TMDB_IMG}{size}{pp}" if pp else None


app.jinja_env.globals["poster_url"] = poster_url


@app.context_processor
def inject_user():
    return {"user": current_user()}


def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if not session.get("uid"):
            flash("Please log in first.")
            return redirect(url_for("login"))
        return f(*a, **k)
    return w


def admin_required(f):
    @wraps(f)
    def w(*a, **k):
        u = current_user()
        if not u or u["role"] != "admin":
            flash("Admin access only.")
            return redirect(url_for("index"))
        return f(*a, **k)
    return w


def all_movies():
    return [dict(r) for r in db().execute("SELECT * FROM movies ORDER BY title")]


_engine = {"key": None, "obj": None}


def get_engine():
    """Retrains the ML models only when the movies/ratings have changed."""
    d = db()
    key = (d.execute("SELECT COUNT(*), COALESCE(SUM(rating*user_id+movie_id),0) FROM ratings").fetchone()[:],
           d.execute("SELECT COUNT(*), COALESCE(SUM(id),0) FROM movies").fetchone()[:])
    key = (tuple(key[0]), tuple(key[1]))
    if _engine["key"] != key:
        ratings = [(r["user_id"], r["movie_id"], r["rating"]) for r in d.execute("SELECT * FROM ratings")]
        _engine["obj"], _engine["key"] = ml_engine.Engine(all_movies(), ratings), key
    return _engine["obj"]


def poster_wall():
    """Movies shown as the poster background on the login/register page."""
    return [dict(r) for r in db().execute("SELECT id, title, year, poster_path FROM movies ORDER BY COALESCE(tmdb_votes,0) DESC, id LIMIT 24")]


# ---------- guest routes ----------
_searched = set()


def tmdb_live_search(q):
    """If the local catalogue has (almost) nothing for this query, look it up on TMDB and store the matches."""
    if not (q and tmdb_enabled()) or q.lower() in _searched:
        return
    _searched.add(q.lower())
    n = db().execute("SELECT COUNT(*) FROM movies WHERE title LIKE ? OR director LIKE ?",
                     (f"%{q}%", f"%{q}%")).fetchone()[0]
    if n < 3:
        tmdb_import.KEY = os.environ["TMDB_API_KEY"]
        try:
            tmdb_import.search_and_add(q)
        except (Exception, SystemExit) as e:
            print("TMDB search failed:", e)


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    genre = request.args.get("genre", "").strip()
    tmdb_live_search(q)
    sql = """SELECT m.*, ROUND(AVG(r.rating),1) AS avg_rating, COUNT(r.rating) AS votes
             FROM movies m LEFT JOIN ratings r ON r.movie_id=m.id WHERE 1=1"""
    args = []
    if q:
        sql += " AND (m.title LIKE ? OR m.director LIKE ?)"
        args += [f"%{q}%", f"%{q}%"]
    if genre:
        sql += " AND m.genres LIKE ?"
        args.append(f"%{genre}%")
    sql += " GROUP BY m.id ORDER BY m.title"
    movies = db().execute(sql, args).fetchall()
    genres = sorted({g.strip() for m in all_movies() for g in m["genres"].split(",")})
    return render_template("index.html", movies=movies, genres=genres, q=q, genre=genre, tmdb_on=tmdb_enabled())


@app.route("/movie/<int:mid>")
def movie(mid):
    m = db().execute("""SELECT m.*, ROUND(AVG(r.rating),1) AS avg_rating, COUNT(r.rating) AS votes
        FROM movies m LEFT JOIN ratings r ON r.movie_id=m.id WHERE m.id=? GROUP BY m.id""", (mid,)).fetchone()
    if not m:
        return redirect(url_for("index"))
    my_rating, in_list = None, False
    u = current_user()
    if u:
        r = db().execute("SELECT rating FROM ratings WHERE user_id=? AND movie_id=?", (u["id"], mid)).fetchone()
        my_rating = r["rating"] if r else None
        in_list = bool(db().execute("SELECT 1 FROM watchlist WHERE user_id=? AND movie_id=?", (u["id"], mid)).fetchone())
    similar = get_engine().similar(mid)
    return render_template("movie.html", m=m, my_rating=my_rating, in_list=in_list, similar=similar)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name, pw = request.form["username"].strip(), request.form["password"]
        if len(name) < 3 or len(pw) < 4:
            flash("Username needs 3+ chars and password 4+ chars.")
        else:
            try:
                db().execute("INSERT INTO users(username,password_hash) VALUES(?,?)",
                             (name, generate_password_hash(pw)))
                db().commit()
                flash("Registered! Please log in.")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Username already taken.")
    return render_template("auth.html", title="Register", posters=poster_wall())


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db().execute("SELECT * FROM users WHERE username=?", (request.form["username"].strip(),)).fetchone()
        if u and check_password_hash(u["password_hash"], request.form["password"]):
            session["uid"] = u["id"]
            return redirect(url_for("admin") if u["role"] == "admin" else url_for("recommendations"))
        flash("Invalid username or password.")
    return render_template("auth.html", title="Login", posters=poster_wall())


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ---------- registered user routes ----------
@app.route("/rate/<int:mid>", methods=["POST"])
@login_required
def rate(mid):
    r = int(request.form["rating"])
    if 1 <= r <= 5:
        db().execute("INSERT OR REPLACE INTO ratings VALUES(?,?,?)", (session["uid"], mid, r))
        db().commit()
    return redirect(url_for("movie", mid=mid))


@app.route("/watchlist/<int:mid>", methods=["POST"])
@login_required
def toggle_watchlist(mid):
    uid = session["uid"]
    if db().execute("SELECT 1 FROM watchlist WHERE user_id=? AND movie_id=?", (uid, mid)).fetchone():
        db().execute("DELETE FROM watchlist WHERE user_id=? AND movie_id=?", (uid, mid))
    else:
        db().execute("INSERT INTO watchlist VALUES(?,?)", (uid, mid))
    db().commit()
    return redirect(url_for("movie", mid=mid))


@app.route("/watchlist")
@login_required
def watchlist():
    rows = db().execute("""SELECT m.* FROM movies m JOIN watchlist w ON w.movie_id=m.id
                           WHERE w.user_id=? ORDER BY m.title""", (session["uid"],)).fetchall()
    return render_template("list.html", title="My Watchlist", movies=rows, empty="Your watchlist is empty.")


@app.route("/recommendations")
@login_required
def recommendations():
    eng = get_engine()
    recs = eng.recommend(session["uid"])
    return render_template("recs.html", recs=recs, taste=eng.taste(session["uid"]),
                           n_rated=len(eng.by_user.get(session["uid"], {})))


@app.route("/how-it-works")
def how_it_works():
    eng = get_engine()
    return render_template("how.html", stats=eng.stats(), metrics=eng.metrics)


# ---------- admin routes ----------
@app.route("/admin", methods=["GET", "POST"])
@admin_required
def admin():
    if request.method == "POST":
        f = request.form
        db().execute("INSERT INTO movies(title,year,genres,director,description) VALUES(?,?,?,?,?)",
                     (f["title"], f["year"] or None, f["genres"], f["director"], f["description"]))
        db().commit()
        flash("Movie added.")
        return redirect(url_for("admin"))
    movies = db().execute("SELECT * FROM movies ORDER BY title").fetchall()
    users = db().execute("SELECT id, username, role FROM users").fetchall()
    return render_template("admin.html", movies=movies, users=users)


@app.route("/admin/delete/<int:mid>", methods=["POST"])
@admin_required
def delete_movie(mid):
    for t, c in (("ratings", "movie_id"), ("watchlist", "movie_id"), ("movies", "id")):
        db().execute(f"DELETE FROM {t} WHERE {c}=?", (mid,))
    db().commit()
    flash("Movie deleted.")
    return redirect(url_for("admin"))


# ---------- TMDB auto-sync (only when the TMDB_API_KEY environment variable is set) ----------
def tmdb_enabled():
    return bool(os.environ.get("TMDB_API_KEY"))


def run_sync():
    tmdb_import.KEY = os.environ["TMDB_API_KEY"]
    try:
        return tmdb_import.sync_new()
    except (Exception, SystemExit) as e:           # never crash the app because TMDB is down
        print("TMDB sync failed:", e)
        return 0


def auto_sync_loop():
    hours = float(os.environ.get("TMDB_SYNC_HOURS", 24))
    while True:
        run_sync()
        time.sleep(hours * 3600)


@app.route("/admin/sync", methods=["POST"])
@admin_required
def admin_sync():
    if tmdb_enabled():
        threading.Thread(target=run_sync, daemon=True).start()
        flash("TMDB sync started - new movies appear in about a minute (refresh the page).")
    else:
        flash("Set the TMDB_API_KEY environment variable and restart the app to enable TMDB sync.")
    return redirect(url_for("admin"))


init_db()
if __name__ == "__main__":
    if tmdb_enabled() and os.environ.get("WERKZEUG_RUN_MAIN") == "true":   # run once, not in the reloader parent
        threading.Thread(target=auto_sync_loop, daemon=True).start()
    app.run(debug=True)
