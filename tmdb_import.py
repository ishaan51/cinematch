"""
Import real movie data from The Movie Database (TMDB) and use it to train the ML models.

    python tmdb_import.py --key YOUR_TMDB_KEY            # 200 top-rated movies
    python tmdb_import.py --key YOUR_KEY --pages 25 --source popular --users 80

Get a free key: themoviedb.org -> Settings -> API. Both the v3 "API Key" and the
v4 "API Read Access Token" work. You can also set the TMDB_API_KEY environment variable.

What comes from TMDB (real data):  title, year, genres, overview, keywords, director,
top cast, poster, vote average / vote count.   -> trains the content-based TF-IDF model.
What is simulated: TMDB has no per-user rating history, so the user x movie rating
table used by Matrix Factorization is generated from 5 taste personas over the REAL
TMDB genres (+ a small bonus for highly rated movies). Ratings from real users of
the app are added to this and the model retrains on them automatically.
"""
import argparse, json, os, shutil, sqlite3, subprocess, sys, time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from werkzeug.security import generate_password_hash
from schema import ensure_schema, ensure_default_users
from seed_data import synthetic_ratings

BASE = "https://api.themoviedb.org/3"
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies.db")
GENRE_FIX = {"Science Fiction": "Sci-Fi", "Music": "Musical", "TV Movie": "Drama"}
KEY = ""


_use_curl = False


def _urllib_fetch(url, headers):
    try:
        with urlopen(Request(url, headers=headers), timeout=20) as r:
            return r.status, r.read().decode("utf-8")
    except HTTPError as e:
        return e.code, ""


def _curl_get(url, headers):
    """Fallback for PCs where Python's HTTPS is blocked/broken but curl (Windows' own SSL) works."""
    exe = shutil.which("curl")
    if not exe:
        raise OSError("curl is not installed")
    cmd = [exe, "-sS", "-L", "--max-time", "25", "-w", "\n%{http_code}"]
    for k, v in headers.items():
        cmd += ["-H", f"{k}: {v}"]
    p = subprocess.run(cmd + [url], capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise OSError(p.stderr.strip() or f"curl failed ({p.returncode})")
    body, _, code = p.stdout.rpartition("\n")
    return int(code), body


def api_get(path, **params):
    global _use_curl
    headers = {"Accept": "application/json"}
    if KEY.startswith("eyJ"):                       # v4 read-access token
        headers["Authorization"] = "Bearer " + KEY
    else:
        params["api_key"] = KEY
    url = f"{BASE}{path}?{urlencode(params)}"
    last = None
    for attempt in range(5):
        try:
            code, body = _curl_get(url, headers) if _use_curl else _urllib_fetch(url, headers)
        except OSError as e:                        # DNS / SSL / reset / timeout
            last = getattr(e, "reason", e)
            if not _use_curl and shutil.which("curl"):
                _use_curl = True
                print("  (Python's HTTPS connection failed - switching to curl)")
            else:
                time.sleep(1 + attempt)
            continue
        if code == 200:
            return json.loads(body)
        if code == 401:
            sys.exit("TMDB rejected the API key (401). Check the key.")
        if code == 404:
            return None
        if code == 429 or code >= 500:              # rate limited / server busy -> wait and retry
            time.sleep(2 * (attempt + 1))
            continue
        sys.exit(f"TMDB returned HTTP {code}")
    sys.exit(f"Could not reach api.themoviedb.org.\nReal reason: {last}\n"
             "See the README section 'TMDB not reachable'.")


def parse_movie(d):
    """TMDB detail JSON (with credits + keywords appended) -> row dict, or None if unusable."""
    if not d or not (d.get("overview") or "").strip():
        return None
    crew = (d.get("credits") or {}).get("crew", [])
    director = next((c["name"] for c in crew if c.get("job") == "Director"), "")
    cast = ", ".join(c["name"] for c in (d.get("credits") or {}).get("cast", [])[:4])
    kws = ", ".join(k["name"] for k in (d.get("keywords") or {}).get("keywords", [])[:12])
    genres = ",".join(GENRE_FIX.get(g["name"], g["name"]) for g in d.get("genres", []))
    if not genres:
        return None
    return dict(title=d["title"], year=int((d.get("release_date") or "0")[:4] or 0) or None,
                genres=genres, director=director, description=d["overview"].strip(),
                keywords=kws, cast_names=cast, tmdb_id=d["id"], poster_path=d.get("poster_path"),
                tmdb_rating=d.get("vote_average"), tmdb_votes=d.get("vote_count"))


def fetch_movies(source, pages, log=print):
    ids = []
    for p in range(1, pages + 1):
        page = api_get(f"/movie/{source}", page=p, language="en-US")
        ids += [m["id"] for m in page["results"]]
        if p >= page.get("total_pages", p):
            break
    rows = []
    for n, tid in enumerate(ids, 1):
        row = parse_movie(api_get(f"/movie/{tid}", append_to_response="credits,keywords", language="en-US"))
        if row:
            rows.append(row)
        if n % 20 == 0:
            log(f"  fetched {n}/{len(ids)} movies")
        time.sleep(0.03)
    return rows


def upsert_movies(con, rows):
    con.executemany("""INSERT INTO movies(title,year,genres,director,description,keywords,cast_names,tmdb_id,poster_path,tmdb_rating,tmdb_votes)
        VALUES(:title,:year,:genres,:director,:description,:keywords,:cast_names,:tmdb_id,:poster_path,:tmdb_rating,:tmdb_votes)
        ON CONFLICT(tmdb_id) DO UPDATE SET title=excluded.title, year=excluded.year, genres=excluded.genres,
        director=excluded.director, description=excluded.description, keywords=excluded.keywords,
        cast_names=excluded.cast_names, poster_path=excluded.poster_path,
        tmdb_rating=excluded.tmdb_rating, tmdb_votes=excluded.tmdb_votes""", rows)


def sync_new(sources=("now_playing", "upcoming", "popular"), pages=2, log=print):
    """Adds movies that are not in the database yet (new releases, newly popular films).
    Existing movies, users and ratings are left untouched. Returns how many movies were added."""
    con = sqlite3.connect(DB)
    ensure_schema(con)
    known = {r[0] for r in con.execute("SELECT tmdb_id FROM movies WHERE tmdb_id IS NOT NULL")}
    ids = []
    for src in sources:
        for p in range(1, pages + 1):
            page = api_get(f"/movie/{src}", page=p, language="en-US")
            ids += [m["id"] for m in page["results"] if m["id"] not in known and m["id"] not in ids]
            if p >= page.get("total_pages", p):
                break
    rows = []
    for tid in ids:
        row = parse_movie(api_get(f"/movie/{tid}", append_to_response="credits,keywords", language="en-US"))
        if row:
            rows.append(row)
        time.sleep(0.03)
    upsert_movies(con, rows)
    con.commit()
    con.close()
    log(f"TMDB sync: {len(rows)} new movies added")
    return len(rows)


def search_and_add(query, limit=5):
    """Live search on TMDB: stores the best matches that are not in the database yet."""
    con = sqlite3.connect(DB)
    ensure_schema(con)
    known = {r[0] for r in con.execute("SELECT tmdb_id FROM movies WHERE tmdb_id IS NOT NULL")}
    res = (api_get("/search/movie", query=query, include_adult="false", language="en-US") or {}).get("results", [])
    rows = []
    for m in res[:limit]:
        if m["id"] in known:
            continue
        row = parse_movie(api_get(f"/movie/{m['id']}", append_to_response="credits,keywords", language="en-US"))
        if row:
            rows.append(row)
    upsert_movies(con, rows)
    con.commit()
    con.close()
    return len(rows)


def save(rows, n_users=60, keep_samples=False, log=print):
    con = sqlite3.connect(DB)
    ensure_schema(con)
    ensure_default_users(con)
    upsert_movies(con, rows)
    if not keep_samples:                            # drop the 24 built-in sample movies
        old = "SELECT id FROM movies WHERE tmdb_id IS NULL"
        con.execute(f"DELETE FROM ratings WHERE movie_id IN ({old})")
        con.execute(f"DELETE FROM watchlist WHERE movie_id IN ({old})")
        con.execute("DELETE FROM movies WHERE tmdb_id IS NULL")

    # regenerate the simulated users on top of the real TMDB catalogue
    fake = "SELECT id FROM users WHERE username GLOB '*_fan[0-9]*'"
    con.execute(f"DELETE FROM ratings WHERE user_id IN ({fake})")
    con.execute("DELETE FROM users WHERE username GLOB '*_fan[0-9]*'")
    movies = [(r[0], r[1], (r[2] or 7.5) - 7.5) for r in
              con.execute("SELECT id, genres, tmdb_rating FROM movies")]
    n = len(movies)
    shared_hash = generate_password_hash("password")
    for name, _, rated in synthetic_ratings(n_users, movies, per_user=(max(9, n // 10), max(14, n // 5))):
        uid = con.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                          (name, shared_hash, "user")).lastrowid
        con.executemany("INSERT INTO ratings VALUES(?,?,?)", [(uid, m, r) for m, r in rated])

    # give the demo account a starter taste (3 sci-fi/action favourites, 1 animation dislike)
    demo = con.execute("SELECT id FROM users WHERE username='demo'").fetchone()
    if demo:
        con.execute("DELETE FROM ratings WHERE user_id=?", (demo[0],))
        def pick(like, k):
            return [r[0] for r in con.execute(
                "SELECT id FROM movies WHERE genres LIKE ? ORDER BY tmdb_votes DESC LIMIT ?", (like, k))]
        for mid in pick("%Sci-Fi%", 3):
            con.execute("INSERT OR REPLACE INTO ratings VALUES(?,?,5)", (demo[0], mid))
        for mid in pick("%Animation%", 1):
            con.execute("INSERT OR REPLACE INTO ratings VALUES(?,?,2)", (demo[0], mid))
    con.commit()
    tot = con.execute("SELECT (SELECT COUNT(*) FROM movies),(SELECT COUNT(*) FROM users),(SELECT COUNT(*) FROM ratings)").fetchone()
    con.close()
    log(f"Done. Database now has {tot[0]} movies, {tot[1]} users, {tot[2]} ratings.")


def main():
    global KEY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default=os.environ.get("TMDB_API_KEY", ""), help="TMDB API key or v4 token")
    ap.add_argument("--source", default="top_rated", choices=["top_rated", "popular", "now_playing", "upcoming"])
    ap.add_argument("--pages", type=int, default=10, help="20 movies per page (default 10 = 200 movies)")
    ap.add_argument("--users", type=int, default=60, help="simulated users to generate (default 60)")
    ap.add_argument("--sync", action="store_true", help="only add NEW movies (keeps everything else)")
    ap.add_argument("--keep-samples", action="store_true", help="keep the 24 built-in sample movies")
    args = ap.parse_args()
    KEY = args.key
    if not KEY:
        sys.exit("Please pass --key YOUR_TMDB_KEY (or set TMDB_API_KEY).")
    if args.sync:
        sync_new()
        return
    print(f"Fetching {args.pages * 20} '{args.source}' movies from TMDB ...")
    rows = fetch_movies(args.source, args.pages)
    print(f"Got {len(rows)} usable movies. Saving + generating training ratings ...")
    save(rows, args.users, args.keep_samples)
    print("Start the app with: python app.py")


if __name__ == "__main__":
    main()
