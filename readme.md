# 🎬 Cinematic Movie Recommender

A premium, hyper-personalized, director-tuned movie recommendation system built with Python, Streamlit, and XGBoost. This system learns from your personal movie ratings to predict match scores, fetching real-time cover art via the OMDb API and presenting recommendations in an immersive glassmorphism card grid.

🔗 **Live Website:** [Streamlit App](https://keshav9926-movie-recommender-app-ofv4l1.streamlit.app/)

---

## 💡 Engine Architecture

The core recommendation logic uses a **Hybrid Profile Affinity + Machine Learning Pipeline** designed to mimic real-world taste profiling. Rather than generic bag-of-words recommendations, the system implements:

### 1. Structured Profile Affinity Engine
Before training the regressor, the engine compiles a structured user profile:
* **Director Affinity**: Computes the user's average rating for directors and scales the boost factor by $\sqrt{\text{count of rated movies by director}}$. This ensures that a single high rating doesn't skew recommendations as much as a sustained pattern of liking a director.
* **Relative Genre Preference**: Scales genre preference weights against their global dataset prevalence (relative TF-IDF weighting). This prevents dominant tags (like *Drama* or *Comedy*) from washing out strong niche signals (like *Mystery*, *Sci-Fi*, or *Thriller*).
* **Star/Actor Affinity**: Compiles an average rating vector across all four cast columns in the dataset.

### 2. XGBoost Baseline Regressor
* Learns textual nuances from plot overviews, decade indicators, directors, cast, and genres using TF-IDF vectorization.
* Predicts a baseline rating score for unrated movies based on metadata features.

### 3. Diversity-Collapsing Filter
* Prevents recommendation listing dumps (e.g. recommending 5 Hitchcock movies in a row).
* Limits recommendations to **at most 1 movie per director** to keep suggestions diverse and fresh.

### 4. Language Whitelist Audit
* The system is audited to filter out non-English films (e.g. *Pan's Labyrinth*, *Uri: The Surgical Strike*) to ensure all recommendations align with English language titles and dialogues as requested. The whitelisted movies are verified using OMDb API queries and cached in `english_movies.json`.

---

## 🎨 Interface Features

* **Glassmorphic Theme**: A dark space-themed background (`#0e111a`) with magenta-pink and orange-yellow neon gradients.
* **Dynamic Cover Art Fetching**: Concurrently queries the OMDb API for high-resolution posters in the background (using a thread pool to avoid slowing down page loads).
* **Interactive Tabs**:
  * **🎯 AI Recommendations**: Instantly generates your Top 12 tailored movies in a responsive two-column grid.
  * **✍️ Rate Movies**: Select any movie to see its poster and overview, adjust your score with a slider, and save/remove ratings.
  * **📋 Rated Catalog**: View a table of all your previously rated films.

---

## 🗂️ Project Structure

```text
movie_recommender/
├── app.py                  # Streamlit application & hybrid recommendation logic
├── english_movies.json     # Whitelisted English-only movie titles catalog
├── imdb_top_1000.csv       # Movie metadata dataset
├── my_ratings.csv          # Local user ratings database
├── web.ipynb               # Model development notebook
├── requirements.txt        # Python library dependencies
└── README.md               # Project documentation
```

---

## 🚀 Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/keshav9926/movie_recommender.git
cd movie_recommender
```

### 2️⃣ Set Up Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
streamlit run app.py
```
*The app will automatically open in your browser at `http://localhost:8501`.*
