# (title, year, genres, director, description)
MOVIES = [
 ("Inception", 2010, "Sci-Fi,Thriller,Action", "Christopher Nolan", "A thief enters dreams to plant an idea in a target's mind."),
 ("Interstellar", 2014, "Sci-Fi,Drama,Adventure", "Christopher Nolan", "Explorers travel through a wormhole to save humanity."),
 ("The Dark Knight", 2008, "Action,Crime,Thriller", "Christopher Nolan", "Batman faces the Joker in Gotham City."),
 ("The Matrix", 1999, "Sci-Fi,Action", "The Wachowskis", "A hacker discovers reality is a simulation."),
 ("Toy Story", 1995, "Animation,Comedy,Family", "John Lasseter", "Toys come alive when humans are not around."),
 ("Finding Nemo", 2003, "Animation,Family,Adventure", "Andrew Stanton", "A clownfish searches the ocean for his lost son."),
 ("The Shawshank Redemption", 1994, "Drama,Crime", "Frank Darabont", "Two prisoners bond over years and find redemption."),
 ("The Godfather", 1972, "Crime,Drama", "Francis Ford Coppola", "The aging patriarch of a mafia family hands over his empire."),
 ("Pulp Fiction", 1994, "Crime,Drama,Thriller", "Quentin Tarantino", "Interwoven tales of crime in Los Angeles."),
 ("Forrest Gump", 1994, "Drama,Romance,Comedy", "Robert Zemeckis", "A simple man witnesses key moments of American history."),
 ("Titanic", 1997, "Romance,Drama", "James Cameron", "A love story aboard the doomed ship."),
 ("Avengers: Endgame", 2019, "Action,Sci-Fi,Adventure", "Anthony & Joe Russo", "The Avengers assemble one last time to undo the snap."),
 ("Jurassic Park", 1993, "Sci-Fi,Adventure,Thriller", "Steven Spielberg", "A theme park of cloned dinosaurs goes wrong."),
 ("The Lion King", 1994, "Animation,Family,Drama", "Roger Allers", "A lion prince flees his kingdom and returns to claim it."),
 ("3 Idiots", 2009, "Comedy,Drama", "Rajkumar Hirani", "Three engineering students challenge the education system."),
 ("Dangal", 2016, "Drama,Sports,Biography", "Nitesh Tiwari", "A father trains his daughters to become wrestlers."),
 ("Zindagi Na Milegi Dobara", 2011, "Comedy,Drama,Adventure", "Zoya Akhtar", "Three friends go on a road trip across Spain."),
 ("Parasite", 2019, "Thriller,Drama,Comedy", "Bong Joon-ho", "A poor family schemes its way into a wealthy household."),
 ("Spirited Away", 2001, "Animation,Fantasy,Family", "Hayao Miyazaki", "A girl enters a world of spirits to save her parents."),
 ("The Conjuring", 2013, "Horror,Thriller", "James Wan", "Paranormal investigators help a haunted family."),
 ("Get Out", 2017, "Horror,Thriller", "Jordan Peele", "A visit to his girlfriend's family turns sinister."),
 ("La La Land", 2016, "Romance,Musical,Drama", "Damien Chazelle", "A pianist and an actress chase dreams in Los Angeles."),
 ("Mad Max: Fury Road", 2015, "Action,Sci-Fi,Adventure", "George Miller", "A chase across a desert wasteland."),
 ("Coco", 2017, "Animation,Family,Fantasy", "Lee Unkrich", "A boy journeys into the Land of the Dead."),
]


# ---------- synthetic users so the ML model has data to learn from ----------
import random

PERSONAS = {
    "scifi":  {"Sci-Fi": 1, "Action": 1, "Thriller": .6, "Mystery": .3, "Animation": -.5, "Family": -.5, "Romance": -.6},
    "family": {"Animation": 1, "Family": 1, "Fantasy": 1, "Adventure": .7, "Horror": -.8, "Crime": -.7, "War": -.5},
    "crime":  {"Crime": 1, "Drama": .9, "Thriller": 1, "Mystery": .8, "War": .3, "History": .3, "Animation": -.5, "Musical": -.5, "Family": -.4},
    "horror": {"Horror": 1, "Thriller": 1, "Mystery": .5, "Romance": -.6, "Family": -.6, "Musical": -.5},
    "feel":   {"Comedy": 1, "Romance": 1, "Drama": .7, "Musical": 1, "Family": .3, "Horror": -.7, "Action": -.4},
}


def synthetic_ratings(n_users=40, movies=None, seed=7, per_user=(9, 14)):
    """Simulated users with different tastes.
    movies: list of (movie_id, "Genre,Genre", quality) - quality is a small bonus/penalty
    (e.g. derived from the TMDB average rating). Returns [(username, persona, [(movie_id, rating)])]."""
    rng = random.Random(seed)
    if movies is None:
        movies = [(i + 1, m[2], 0.0) for i, m in enumerate(MOVIES)]
    names, out = list(PERSONAS), []
    for i in range(n_users):
        persona = names[i % len(names)]
        k = min(len(movies), rng.randint(*per_user))
        rated = []
        for mid, genres, q in rng.sample(movies, k):
            gs = [g.strip() for g in genres.split(",") if g.strip()]
            aff = sum(PERSONAS[persona].get(g, 0) for g in gs) / max(1, len(gs))
            r = round(3 + 2.2 * aff + 0.6 * q + rng.gauss(0, .5))
            rated.append((mid, max(1, min(5, r))))
        out.append((f"{persona}_fan{i // len(names) + 1}", persona, rated))
    return out
