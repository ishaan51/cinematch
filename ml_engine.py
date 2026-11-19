"""
Machine-learning recommender for CineMatch.

Three models are combined (a *hybrid* recommender):

1. Content-based  : TF-IDF vectors of each movie's genres/director/description,
                    compared with cosine similarity (unsupervised NLP / vector-space model).
2. Collaborative  : Matrix Factorization trained with Stochastic Gradient Descent
                    (supervised learning - it learns hidden "taste factors" that predict ratings).
3. Hybrid blend   : weights the two models depending on how many ratings the user has
                    (cold-start handling).

The model is evaluated with a train/test split (RMSE / MAE) so the accuracy is measurable.
"""
import math
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------ Matrix Factorization
class MatrixFactorization:
    """r_hat(u,i) = mu + b_u + b_i + p_u . q_i   (trained with SGD + L2 regularisation)"""

    def __init__(self, k=6, epochs=80, lr=0.01, reg=0.08, seed=42):
        self.k, self.epochs, self.lr, self.reg, self.seed = k, epochs, lr, reg, seed

    def fit(self, triples):
        rng = np.random.default_rng(self.seed)
        self.users = {u: i for i, u in enumerate(sorted({t[0] for t in triples}))}
        self.items = {m: i for i, m in enumerate(sorted({t[1] for t in triples}))}
        self.mu = float(np.mean([t[2] for t in triples]))
        self.bu = np.zeros(len(self.users))
        self.bi = np.zeros(len(self.items))
        self.P = rng.normal(0, 0.1, (len(self.users), self.k))
        self.Q = rng.normal(0, 0.1, (len(self.items), self.k))
        data = [(self.users[u], self.items[m], r) for u, m, r in triples]
        self.history = []
        for _ in range(self.epochs):
            rng.shuffle(data)
            sq = 0.0
            for u, i, r in data:
                err = r - (self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])
                sq += err * err
            self.history.append(math.sqrt(sq / len(data)))
        return self

    def predict(self, u, m):
        s = self.mu
        ui, ii = self.users.get(u), self.items.get(m)
        if ui is not None:
            s += self.bu[ui]
        if ii is not None:
            s += self.bi[ii]
        if ui is not None and ii is not None:
            s += self.P[ui] @ self.Q[ii]
        return float(min(5, max(1, s)))


# ------------------------------------------------------------------ Hybrid engine
def _movie_text(m):
    genres = m["genres"].replace(",", " ").replace("-", "")
    extra = f"{m.get('keywords') or ''} {m.get('cast_names') or ''}"      # TMDB keywords + cast
    return f"{genres} {genres} {genres} {m.get('director') or ''} {extra} {m.get('description') or ''}"  # genres weighted x3


def _minmax(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    return {k: (v - lo) / (hi - lo) if hi > lo else 0.5 for k, v in d.items()}


class Engine:
    def __init__(self, movies, ratings):
        """movies: list of dicts; ratings: list of (user_id, movie_id, rating)."""
        self.movies = {m["id"]: m for m in movies}
        self.ids = [m["id"] for m in movies]
        self.pos = {mid: i for i, mid in enumerate(self.ids)}
        self.ratings = list(ratings)
        self.by_user = defaultdict(dict)
        for u, m, r in self.ratings:
            self.by_user[u][m] = r

        # model 1: TF-IDF content vectors
        self.tfidf = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z0-9]+")
        self.X = self.tfidf.fit_transform([_movie_text(m) for m in movies])
        self.sim = cosine_similarity(self.X)

        # model 2: matrix factorization
        self.mf = MatrixFactorization().fit(self.ratings) if self.ratings else None
        self.metrics = self._evaluate()

    # ---- evaluation (hold out 20% of ratings)
    def _evaluate(self):
        if len(self.ratings) < 30:
            return None
        rng = np.random.default_rng(1)
        idx = rng.permutation(len(self.ratings))
        cut = int(len(idx) * 0.8)
        train = [self.ratings[i] for i in idx[:cut]]
        test = [self.ratings[i] for i in idx[cut:]]
        mf = MatrixFactorization().fit(train)
        mu = np.mean([r for _, _, r in train])
        item = defaultdict(list)
        for _, m, r in train:
            item[m].append(r)
        item_mean = {m: np.mean(v) for m, v in item.items()}

        def rmse(pred):
            return float(np.sqrt(np.mean([(pred(u, m) - r) ** 2 for u, m, r in test])))

        def mae(pred):
            return float(np.mean([abs(pred(u, m) - r) for u, m, r in test]))

        models = {
            "Global average (baseline)": lambda u, m: mu,
            "Movie average (baseline)": lambda u, m: item_mean.get(m, mu),
            "Matrix Factorization (ML)": mf.predict,
        }
        return {
            "train": len(train), "test": len(test),
            "rows": [(n, rmse(f), mae(f)) for n, f in models.items()],
            "curve": mf.history[::8],
        }

    # ---- stats for the explanation page
    def stats(self):
        n_u, n_m, n_r = len({u for u, _, _ in self.ratings}), len(self.movies), len(self.ratings)
        return {"users": n_u, "movies": n_m, "ratings": n_r,
                "sparsity": 100 * (1 - n_r / (n_u * n_m)) if n_u and n_m else 0,
                "vocab": len(self.tfidf.vocabulary_), "tmdb": sum(1 for m in self.movies.values() if m.get("tmdb_id")), "k": self.mf.k if self.mf else 0}

    # ---- model 1 : content-based
    def content_scores(self, user_id):
        mine = self.by_user.get(user_id, {})
        if not mine:
            return {}
        mean = np.mean(list(mine.values()))
        w = np.zeros(self.X.shape[1])
        for m, r in mine.items():
            w += (r - mean) * self.X[self.pos[m]].toarray().ravel()
        if not w.any():  # all ratings equal -> plain average profile
            for m in mine:
                w += self.X[self.pos[m]].toarray().ravel()
        sims = cosine_similarity(w.reshape(1, -1), self.X).ravel()
        return {m: float(sims[self.pos[m]]) for m in self.ids if m not in mine}

    # ---- hybrid recommendation
    def recommend(self, user_id, n=8):
        mine = self.by_user.get(user_id, {})
        if not mine:  # cold start: Bayesian-average popularity
            allr = defaultdict(list)
            for _, m, r in self.ratings:
                allr[m].append(r)
            gm = np.mean([r for _, _, r in self.ratings]) if self.ratings else 3.0
            pop = {}
            for m, mv in self.movies.items():          # prior = TMDB average (0-10 -> 0-5) when known
                prior = mv["tmdb_rating"] / 2 if mv.get("tmdb_rating") else gm
                v = allr.get(m, [])
                pop[m] = (sum(v) + 3 * prior) / (len(v) + 3)
            top = sorted(pop, key=pop.get, reverse=True)[:n]
            return [{"movie": self.movies[m], "pred": pop[m], "reason": "Popular with all users and well rated on TMDB (no ratings from you yet)"} for m in top]

        content = self.content_scores(user_id)
        cf = {m: self.mf.predict(user_id, m) for m in content}
        w_cf = 0.6 * min(1.0, len(mine) / 8)          # trust collaborative more as the user rates more
        cn, fn = _minmax(content), _minmax(cf)
        final = {m: w_cf * fn[m] + (1 - w_cf) * cn[m] for m in content}
        out = []
        for m in sorted(final, key=final.get, reverse=True)[:n]:
            liked = [(self.sim[self.pos[m], self.pos[x]], x) for x, r in mine.items() if r >= 4]
            reasons = []
            if liked:
                s, x = max(liked)
                if s > 0.05:
                    reasons.append(f"Similar to “{self.movies[x]['title']}” which you rated highly")
            if cf[m] >= 3.8:
                reasons.append("Users with taste like yours rated it highly")
            out.append({"movie": self.movies[m], "pred": cf[m], "content": content[m],
                        "score": final[m], "reason": " · ".join(reasons) or "Matches the genres you rate highest"})
        return out

    def similar(self, movie_id, n=4):
        row = self.sim[self.pos[movie_id]]
        order = [i for i in np.argsort(-row) if self.ids[i] != movie_id][:n]
        return [self.movies[self.ids[i]] for i in order if row[i] > 0]

    def taste(self, user_id, top=4):
        g = defaultdict(list)
        for m, r in self.by_user.get(user_id, {}).items():
            for x in self.movies[m]["genres"].split(","):
                g[x.strip()].append(r)
        ranked = sorted(((sum(v) / len(v), k) for k, v in g.items() if len(v) >= 1), reverse=True)
        return [(k, round(a, 1)) for a, k in ranked[:top]]
