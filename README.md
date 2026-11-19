🎬 CineMatch – Movie Recommendation System

<p align="center">
  <strong>Discover Movies. Explore Your Taste. Find Your Next Favourite.</strong>
</p>CineMatch is an intelligent, web-based movie recommendation system project. Built using Python Flask and SQLite, it helps users discover movies through personalized recommendations, movie searches, ratings, and watchlists.

The platform integrates the TMDB API to fetch real-world movie information and uses a Hybrid Machine Learning Recommendation Engine combining Content-Based Filtering and Collaborative Filtering to deliver personalized movie suggestions.

---

✨ Key Features

👤 Three User Roles

Role| Features
Guest User| Browse movies, search movies, filter by genre, view movie details
Registered User| Register, login, rate movies (1–5), manage watchlist, get personalized recommendations
Admin| Add movies, delete movies, view registered users, synchronize movies from TMDB

🎥 Movie Discovery

- Browse an extensive movie catalogue.
- Search for movies by title.
- Filter movies by genre.
- View movie descriptions, posters, cast, director, keywords, and ratings.
- Search for movies directly through TMDB when they are not available locally.

⭐ Personalized Recommendations

- Get recommendations based on your interests and ratings.
- Content-based movie recommendations.
- Collaborative filtering using Matrix Factorization.
- Hybrid recommendation algorithm.
- Popularity-based fallback for new users.

❤️ Watchlist & Ratings

- Add movies to your personal watchlist.
- Rate movies from 1 to 5 stars.
- Use your ratings to improve future recommendations.
- Automatically retrain recommendation models when new ratings are added.

🔄 TMDB Integration

- Import real movie information from TMDB.
- Fetch movie posters, genres, descriptions, cast, directors, and keywords.
- Automatically synchronize new and popular releases every 24 hours.
- Manually synchronize movies through the Admin Panel.

---

🧠 Machine Learning & Recommendation System

CineMatch uses a Hybrid Recommendation Engine implemented in "ml_engine.py".

It combines two machine learning techniques:

1. Content-Based Filtering

- Uses TF-IDF (Term Frequency–Inverse Document Frequency).
- Analyzes movie genres, directors, and descriptions.
- Uses Cosine Similarity to identify movies similar to a user's taste.

2. Collaborative Filtering

- Uses Matrix Factorization with Stochastic Gradient Descent (SGD).
- Predicts user ratings using 6 latent factors.
- Includes L2 regularization to reduce overfitting.
- Learns patterns from user–movie rating interactions.

3. Hybrid Recommendation

Combines both approaches into a single recommendation score:

final_score = w * collaborative_score + (1 - w) * content_score

Where:

w = 0.6 * min(1, ratings / 8)

The system gives more importance to collaborative filtering as users provide more ratings. For new users, content-based recommendations and popularity-based fallback help provide relevant suggestions.

4. Model Evaluation

The recommendation engine is evaluated using:

- 80/20 Train-Test Split
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Comparison against baseline models

Users can explore the recommendation process on the How ML Works page.

---

🌐 TMDB API Integration

CineMatch uses the "TMDB API" (https://developer.themoviedb.org/) to retrieve real-world movie information.

Import Real Movies

1. Create a free account at "The Movie Database" (https://www.themoviedb.org/).
2. Navigate to Settings → API.
3. Request your API key.
4. Run the following command:

python tmdb_import.py --key YOUR_TMDB_KEY

Import Options

Command| Description
"--pages 25"| Import up to 500 movies
"--source popular"| Import popular movies
"--users 80"| Generate 80 synthetic users
"--keep-samples"| Preserve existing sample movies
"--sync"| Synchronize new movies from TMDB

Example:

python tmdb_import.py --key YOUR_TMDB_KEY --pages 25 --source popular

Automatic Movie Synchronization

CineMatch can automatically check TMDB every 24 hours for new releases and popular movies.

Configure the synchronization interval using:

TMDB_SYNC_HOURS=24

You can also manually synchronize movies from the Admin Panel.

---

📊 Real Data vs. Simulated Data

An important part of CineMatch is the combination of real movie information and simulated rating history.

Data| Source
Movie titles| TMDB API
Genres| TMDB API
Movie descriptions| TMDB API
Cast and directors| TMDB API
Keywords and posters| TMDB API
Average movie ratings| TMDB API
User rating history for initial training| Synthetic taste personas
Actual application ratings| Registered users

Why simulated data?

TMDB does not provide individual user-to-movie rating histories through its standard movie API. Therefore, CineMatch generates an initial user–movie rating matrix using five synthetic taste personas based on real TMDB genres.

As registered users rate movies, their actual ratings are incorporated into the recommendation system, and the models retrain automatically.

---

🛠️ Technology Stack

Technology| Purpose
Python| Core programming language
Flask| Backend web framework
SQLite| Database management
HTML| Web page structure
CSS| Styling and user interface
TMDB API| Real movie information
Scikit-learn| TF-IDF and cosine similarity
NumPy| Numerical computations
Machine Learning| Personalized movie recommendations

---

📁 Project Structure

CineMatch/
│
├── app.py              # Flask application, routes, authentication
├── tmdb_import.py      # TMDB movie importer and synchronization
├── schema.py           # Database schema and tables
├── ml_engine.py        # Hybrid ML recommendation engine
├── seed_data.py        # Sample movies and synthetic users
├── requirements.txt    # Python dependencies
│
├── templates/          # HTML templates
│
├── static/             # CSS and static assets
│
└── README.md           # Project documentation

---

⚙️ Installation & Setup

Follow these steps to run CineMatch on your local machine.

Step 1: Clone the Repository

git clone YOUR_REPOSITORY_URL
cd CineMatch

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Run the Application

python app.py

Step 4: Open in Browser

Visit:

http://127.0.0.1:5000

Your CineMatch application is now running locally!

---

🔐 Demo Login Credentials

Use the following credentials to explore the application:

Role| Username| Password
Admin| "admin"| "admin123"
Demo User| "demo"| "demo123"

These are default demo credentials. Change them before deploying the application publicly.

---

🔧 Troubleshooting TMDB Connection

If the TMDB importer cannot connect to "api.themoviedb.org", try:

1. Change DNS settings to "1.1.1.1" or "8.8.8.8".
2. Try a mobile hotspot or VPN if your network blocks TMDB.
3. On Windows 10/11, the importer can automatically fall back to "curl" if Python requests fail.
4. Check SSL certificate errors, especially "CERTIFICATE_VERIFY_FAILED".

---

🚀 Future Enhancements

- AI-powered movie chatbot.
- Advanced user preference analysis.
- Improved recommendation accuracy.
- More advanced deep learning recommendation models.
- User profiles and personalized dashboards.
- Social features for sharing movie recommendations.

---

⚠️ Disclaimer

This product uses the TMDB API but is not endorsed or certified by TMDB.

Movie information and images are provided by The Movie Database (TMDB).

---

👨‍💻 Project Information

Project Name: CineMatch – Movie Recommendation System

Project Type: College Project – SRS Experiment No. 2

Backend: Python Flask

Database: SQLite

Recommendation Engine: Hybrid Machine Learning (Content-Based + Collaborative Filtering)

---

<p align="center">
  <strong>🎬 CineMatch – Your Next Favourite Movie Is Just a Recommendation Away!</strong>
</p><p align="center">
  ⭐ If you like this project, consider giving it a star!
</p>
