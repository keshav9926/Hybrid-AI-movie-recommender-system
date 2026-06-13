import streamlit as st
import pandas as pd
import numpy as np
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
RATING_COL = 'My_rating '  # Trailing space from the dataset schema
RATINGS_FILE = "my_ratings.csv"

# Setup quiet NLTK downloads
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4.zip')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

download_nltk_resources()

# Load & preprocess movies
@st.cache_data
def load_movies(file_path="imdb_top_1000.csv"):
    movies = pd.read_csv(file_path)

    # Filter for English movies / English titles using precompiled mapping
    english_list_file = "english_movies.json"
    if os.path.exists(english_list_file):
        import json
        with open(english_list_file, 'r', encoding='utf-8') as f:
            english_titles = set(json.load(f))
        movies = movies[movies['Series_Title'].isin(english_titles)]

    movies['Meta_score'] = movies['Meta_score'] / 10
    movies = movies[movies['Meta_score'].notnull()]

    # Strip spaces ONLY for text data compilation to merge keywords
    director_clean = movies['Director'].astype(str).str.replace(" ", "")
    star1_clean = movies['Star1'].astype(str).str.replace(" ", "")
    star2_clean = movies['Star2'].astype(str).str.replace(" ", "")
    star3_clean = movies['Star3'].astype(str).str.replace(" ", "")
    star4_clean = movies['Star4'].astype(str).str.replace(" ", "")

    # Extract decade tag for temporal context (e.g. "Decade_1990s")
    decade = movies['Released_Year'].astype(str).str.extract(r'(\d{3})')[0].fillna('200') + '0s'
    decade_tag = 'Decade_' + decade

    movies['text_data'] = (
        movies['Series_Title'].astype(str) + ' ' +
        movies['Overview'].astype(str) + ' ' +
        director_clean + ' ' +
        star1_clean + ' ' +
        star2_clean + ' ' +
        star3_clean + ' ' +
        star4_clean + ' ' +
        decade_tag + ' ' +
        movies['Genre'].astype(str)
    )

    lemmatizer = WordNetLemmatizer()
    movies['text_data'] = movies['text_data'].apply(
        lambda x: ' '.join(lemmatizer.lemmatize(w) for w in x.split())
    )

    if RATING_COL not in movies.columns:
        movies[RATING_COL] = None

    return movies

# Save / Load user ratings (Global file)
def save_user_ratings(movies, path=RATINGS_FILE):
    movies[['Series_Title', RATING_COL]].dropna().to_csv(path, index=False)

def load_user_ratings(movies, path=RATINGS_FILE):
    if not os.path.exists(path):
        return movies

    ratings = pd.read_csv(path)
    # Clear any old ratings columns and merge cleanly
    if RATING_COL in movies.columns:
        movies.drop(columns=[RATING_COL], inplace=True)
    movies = movies.merge(ratings, on='Series_Title', how='left')
    return movies

# Color mapping helper for movie cards
def get_genre_badge_css(genre):
    color_map = {
        'Drama': ('rgba(52, 152, 219, 0.12)', '#3498db'),
        'Action': ('rgba(231, 76, 60, 0.12)', '#e74c3c'),
        'Adventure': ('rgba(230, 126, 34, 0.12)', '#e67e22'),
        'Comedy': ('rgba(241, 196, 15, 0.12)', '#f1c40f'),
        'Sci-Fi': ('rgba(26, 188, 156, 0.12)', '#1abc9c'),
        'Thriller': ('rgba(155, 89, 182, 0.12)', '#9b59b6'),
        'Crime': ('rgba(44, 62, 80, 0.15)', '#5d6d7e'),
        'Romance': ('rgba(255, 105, 180, 0.12)', '#ff69b4'),
        'Animation': ('rgba(142, 68, 173, 0.12)', '#8e44ad'),
        'Biography': ('rgba(22, 160, 133, 0.12)', '#16a085'),
        'History': ('rgba(127, 140, 141, 0.12)', '#7f8c8d'),
        'War': ('rgba(192, 57, 43, 0.12)', '#c0392b'),
        'Mystery': ('rgba(52, 73, 94, 0.12)', '#2c3e50'),
        'Horror': ('rgba(149, 165, 166, 0.12)', '#95a5a6'),
        'Fantasy': ('rgba(224, 86, 253, 0.12)', '#be2edd')
    }
    bg, fg = color_map.get(genre, ('rgba(255, 189, 89, 0.12)', '#ffbd59'))
    return f'background: {bg}; color: {fg}; border: 1px solid {fg}40;'

# Fetch movie poster URL from OMDb API with caching
@st.cache_data(show_spinner=False)
def get_movie_poster(title):
    import urllib.request
    import urllib.parse
    import json
    url_title = urllib.parse.quote(title)
    url = f"http://www.omdbapi.com/?t={url_title}&apikey=thewdb"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as res:
            data = json.loads(res.read().decode('utf-8'))
            if data.get('Response') == 'True':
                poster = data.get('Poster')
                if poster and poster != 'N/A':
                    return poster
    except Exception:
        pass
    return None

# Fetch posters for a list of movies in parallel
def fetch_posters_parallel(titles):
    from concurrent.futures import ThreadPoolExecutor
    def fetch_one(t):
        return t, get_movie_poster(t)
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_one, titles))
    return dict(results)

# Render movie card in HTML
def render_movie_card_html(movie, rank_num=None, predicted_rating=None, poster_url=None):
    genres = [g.strip() for g in str(movie['Genre']).split(',')]
    genre_html = "".join([f'<span class="genre-badge" style="{get_genre_badge_css(g)}">{g}</span>' for g in genres])
    
    imdb_rating = movie['IMDB_Rating']
    runtime = movie['Runtime']
    year = movie['Released_Year']
    director = movie['Director']
    overview = str(movie['Overview']).replace('"', '&quot;')
    title = movie['Series_Title']
    stars = ", ".join([movie['Star1'], movie['Star2'], movie['Star3'], movie['Star4']])
    
    rank_html = ""
    if rank_num is not None:
        rank_html = f'<span class="genre-badge match-badge" style="background: rgba(212, 175, 55, 0.1); color: #d4af37; border: 1px solid rgba(212, 175, 55, 0.35);">🔥 #{rank_num}</span>'
        
    pred_badge_html = ""
    if predicted_rating is not None:
        pred_badge_html = f'<span class="genre-badge match-badge" style="background: rgba(212, 175, 55, 0.1); color: #d4af37; border: 1px solid rgba(212, 175, 55, 0.35);">🎯 Match: {predicted_rating:.2f}</span>'
        
    if poster_url:
        poster_html = f'<img class="movie-poster" src="{poster_url}" alt="{title}">'
    else:
        # Fallback SVG icon
        poster_html = """<div style="display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; background: linear-gradient(135deg, #182030 0%, #0c1018 100%);"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="rgba(255, 255, 255, 0.3)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line><line x1="2" y1="7" x2="7" y2="7"></line><line x1="2" y1="17" x2="7" y2="17"></line><line x1="17" y1="17" x2="22" y2="17"></line><line x1="17" y1="7" x2="22" y2="7"></line></svg></div>"""

    card_html = f"""<div class="movie-card">
<div class="movie-poster-container">
{poster_html}
</div>
<div class="movie-info">
<div>
<div class="movie-title-row">
<h3 class="movie-title">{title}</h3>
</div>
<div class="movie-meta">
📅 {year} &nbsp;•&nbsp; ⏱️ {runtime} &nbsp;•&nbsp; 🎬 Dir: {director}
</div>
<div class="genre-container">
{genre_html}
</div>
<p class="movie-plot">{overview}</p>
</div>
<div style="display: flex; justify-content: space-between; align-items: flex-end; width: 100%; margin-top: 10px;">
<div class="movie-stars" style="border: none; padding: 0; margin: 0; width: 70%; display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis;">
<strong>Cast:</strong> {stars}
</div>
<div class="badge-row" style="display: flex; gap: 5px; flex-shrink: 0;">
{rank_html}
{pred_badge_html}
<span class="genre-badge rating-badge">⭐ {imdb_rating}</span>
</div>
</div>
</div>
</div>"""
    return card_html

# Render detail card for current selection
def render_detail_card(movie, poster_url=None):
    title = movie['Series_Title']
    year = movie['Released_Year']
    runtime = movie['Runtime']
    director = movie['Director']
    overview = movie['Overview']
    genres = [g.strip() for g in str(movie['Genre']).split(',')]
    genre_html = "".join([f'<span class="genre-badge" style="{get_genre_badge_css(g)}">{g}</span>' for g in genres])
    imdb_rating = movie['IMDB_Rating']
    stars = ", ".join([movie['Star1'], movie['Star2'], movie['Star3'], movie['Star4']])
    
    if poster_url:
        poster_html = f'<img class="movie-poster" src="{poster_url}" alt="{title}">'
    else:
        poster_html = """<div style="display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; background: linear-gradient(135deg, #182030 0%, #0c1018 100%);"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="rgba(255, 255, 255, 0.3)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line><line x1="2" y1="7" x2="7" y2="7"></line><line x1="2" y1="17" x2="7" y2="17"></line><line x1="17" y1="17" x2="22" y2="17"></line><line x1="17" y1="7" x2="22" y2="7"></line></svg></div>"""
        
    card_html = f"""<div class="movie-card" style="margin-bottom: 0;">
<div class="movie-poster-container" style="width: 140px; height: 210px;">
{poster_html}
</div>
<div class="movie-info" style="margin-left: 20px;">
<div>
<h3 class="movie-title" style="font-size: 1.3rem;">{title}</h3>
<div class="movie-meta">
📅 {year} &nbsp;•&nbsp; ⏱️ {runtime} &nbsp;•&nbsp; 🎬 Dir: {director}
</div>
<div class="genre-container">
{genre_html}
</div>
<p class="movie-plot" style="-webkit-line-clamp: 4; height: auto; min-height: 4.5em;">{overview}</p>
</div>
<div style="display: flex; justify-content: space-between; align-items: flex-end; width: 100%; margin-top: 10px;">
<div class="movie-stars" style="border: none; padding: 0; margin: 0;">
<strong>Cast:</strong> {stars}
</div>
<span class="genre-badge rating-badge">⭐ {imdb_rating}</span>
</div>
</div>
</div>"""
    return card_html



# Recommendation Engine Logic (Hybrid Profile Affinity + XGBoost with Diversity Filtering)
def train_and_recommend(movies_df, top_n=10):
    # -------------------------------------------------------------------------
    # MACHINE LEARNING CONCEPT: DATA SPLITTING & COLD START HANDLING
    # - "rated" acts as our Training Set (labeled data where the target variable is 'My_Rating').
    # - "unrated" acts as our Test Set/Prediction Target (unlabeled data where we want to predict 'My_Rating').
    # - If the user has rated fewer than 2 movies, we cannot train an XGBoost model or build robust affinity profiles.
    # -------------------------------------------------------------------------
    rated = movies_df[movies_df[RATING_COL].notnull()]
    unrated = movies_df[movies_df[RATING_COL].isnull()]

    if len(rated) < 2:
        return None

    # -------------------------------------------------------------------------
    # ML CONCEPT: FEATURE ENGINEERING - USER PROFILE AFFINITIES
    # We build custom preference dictionaries to capture the user's affinity towards:
    # 1. Directors (Who directed the movie?)
    # 2. Stars/Actors (Who acts in the movie?)
    # 3. Genres (What category does the movie belong to?)
    # -------------------------------------------------------------------------
    
    # Calculate global genre frequencies across the entire dataset to handle bias correction
    global_genre_counts = {}
    for genres_str in movies_df['Genre']:
        if pd.notnull(genres_str):
            for g in genres_str.split(','):
                g_clean = g.strip()
                global_genre_counts[g_clean] = global_genre_counts.get(g_clean, 0) + 1

    # Compute director average ratings and the count of movies rated under each director
    director_ratings = rated.groupby('Director')[RATING_COL].mean().to_dict()
    director_counts = rated.groupby('Director').size().to_dict()

    # Compute star average ratings by checking across all 4 cast members in the dataset
    star_ratings = {}
    for star_col in ['Star1', 'Star2', 'Star3', 'Star4']:
        for star, rating in zip(rated[star_col], rated[RATING_COL]):
            if pd.notnull(star):
                star_ratings.setdefault(star, []).append(rating)
    star_avg_ratings = {k: np.mean(v) for k, v in star_ratings.items()}

    # Compute genre average ratings and genre counts rated by the user
    genre_ratings = {}
    genre_counts = {}
    for genres_str, rating in zip(rated['Genre'], rated[RATING_COL]):
        if pd.notnull(genres_str):
            for g in genres_str.split(','):
                g_clean = g.strip()
                genre_ratings.setdefault(g_clean, []).append(rating)
                genre_counts[g_clean] = genre_counts.get(g_clean, 0) + 1
    genre_avg_ratings = {k: np.mean(v) for k, v in genre_ratings.items()}

    # -------------------------------------------------------------------------
    # ML CONCEPT: NATURAL LANGUAGE PROCESSING (NLP) & TF-IDF VECTORIZATION
    # - TfidfVectorizer converts text overviews/plots into numeric feature matrices.
    # - Term Frequency (TF): Measures how frequently a word appears in a specific movie overview.
    # - Inverse Document Frequency (IDF): Dampens words that appear across almost all movies (like "the", "movie", "story")
    #   and highlights rare, descriptive words (like "detective", "stranger", "space", "crime").
    # - fit_transform() learns the vocabulary and vectorizes the training text (rated).
    # - transform() vectorizes the test text (unrated) using the same learned vocabulary.
    # -------------------------------------------------------------------------
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
    X_rated = tfidf.fit_transform(rated['text_data'])
    X_unrated = tfidf.transform(unrated['text_data'])

    # Combine text features with numeric features (IMDb Rating and Metacritic Score) using sparse matrices (hstack)
    X_train = hstack([X_rated, csr_matrix(rated[['IMDB_Rating', 'Meta_score']])])
    y_train = rated[RATING_COL].values
    X_unrated_xgb = hstack([X_unrated, csr_matrix(unrated[['IMDB_Rating', 'Meta_score']])])

    # -------------------------------------------------------------------------
    # ML CONCEPT: SUPERVISED LEARNING (GRADIENT BOOSTING REGRESSION)
    # - XGBRegressor is an ensemble model that builds sequential decision trees.
    # - Each new tree is trained to correct the errors (residuals) of the previous trees.
    # - Why XGBoost? It handles non-linear relationships, text vector features, and numerical data exceptionally well.
    # - Hyperparameters used:
    #   * objective='reg:squarederror': Optimizes the Mean Squared Error (MSE) loss.
    #   * n_estimators=100: Number of trees built.
    #   * learning_rate=0.05: Shrinks the contribution of each tree to prevent overfitting.
    #   * max_depth=3: Limits tree depth to keep trees simple and prevent memorizing training data.
    # -------------------------------------------------------------------------
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    xgb_preds = model.predict(X_unrated_xgb)

    # -------------------------------------------------------------------------
    # ML CONCEPT: HYBRID ENSEMBLE COMBINATION
    # To combine general content similarities (XGBoost) with hyper-specific preferences:
    # 1. We start with the XGBoost baseline prediction.
    # 2. We apply a Director Boost: Weight scaled by sqrt(movies_count) so that
    #    multiple ratings carry higher significance than a single high rating.
    # 3. We apply an Actor Boost.
    # 4. We apply a Relative Genre Preference Boost: Inspired by TF-IDF, it divides the user's rated genre ratio
    #    by the global genre ratio in the dataset. This highlights niche genres (e.g. Film-Noir) that the user loves
    #    and down-weights generic genres (e.g. Drama) that are everywhere.
    # -------------------------------------------------------------------------
    custom_preds = []
    for idx, row in unrated.iterrows():
        xgb_pred = xgb_preds[len(custom_preds)]
        
        # Calculate Director Boost
        dir_name = row['Director']
        dir_score = director_ratings.get(dir_name, None)
        dir_count = director_counts.get(dir_name, 0)
        
        # Calculate Star/Actor Boost
        stars = [row['Star1'], row['Star2'], row['Star3'], row['Star4']]
        star_scores = [star_avg_ratings.get(s) for s in stars if s in star_avg_ratings]
        star_score = np.mean(star_scores) if star_scores else None
        
        # Calculate Genre Boost with TF-IDF Prevalence Ratio
        genres = [g.strip() for g in row['Genre'].split(',')]
        g_scores = []
        g_weights = []
        for g in genres:
            if g in genre_avg_ratings:
                g_scores.append(genre_avg_ratings[g])
                user_ratio = genre_counts.get(g, 0) / len(rated)
                global_ratio = global_genre_counts.get(g, 0) / len(movies_df)
                rel_pref = user_ratio / (global_ratio + 1e-5)
                g_weights.append(np.log1p(rel_pref))
                
        if g_scores:
            genre_score = np.average(g_scores, weights=g_weights)
            genre_count_factor = np.mean(g_weights)
        else:
            genre_score = None
            genre_count_factor = 0
        
        # Assemble custom hybrid scores (offset from middle rating 5.0 to calculate boosts/penalties)
        score = 0.15 * xgb_pred
        
        if dir_score is not None:
            # sqrt(dir_count) rewards multiple positive occurrences (e.g. liking 3 Hitchcock films carries more weight than 1)
            count_factor = np.sqrt(dir_count)
            score += 3.8 * (dir_score - 5.0) * count_factor
        if star_score is not None:
            score += 0.8 * (star_score - 5.0)
        if genre_score is not None:
            score += 2.0 * (genre_score - 5.0) * genre_count_factor
            
        custom_preds.append(score)

    # -------------------------------------------------------------------------
    # ML CONCEPT: MIN-MAX FEATURE SCALING (NORMALIZATION)
    # Scaled predictions back to a standard user-friendly rating scale (1.0 to 10.0 stars).
    # -------------------------------------------------------------------------
    min_score, max_score = min(custom_preds), max(custom_preds)
    if max_score - min_score > 1e-5:
        custom_preds_scaled = [1.0 + 9.0 * (s - min_score) / (max_score - min_score) for s in custom_preds]
    else:
        custom_preds_scaled = [5.0] * len(custom_preds)

    unrated = unrated.copy()
    unrated['Predicted_My_Rating'] = custom_preds_scaled
    recs_all = unrated.sort_values(by='Predicted_My_Rating', ascending=False)

    # -------------------------------------------------------------------------
    # ML CONCEPT: DIVERSITY FILTERING (COLLAPSING RECOMMENDATIONS)
    # - Recommender systems often suffer from "filter bubbles" where the list is saturated by one category.
    # - Here, we collapse predictions to at most 1 film per director to maximize the variety of the recommendations.
    # -------------------------------------------------------------------------
    diverse_recs = []
    seen_directors = set()

    for idx, row in recs_all.iterrows():
        dir_name = row['Director']
        if dir_name not in seen_directors:
            diverse_recs.append(row)
            seen_directors.add(dir_name)
        if len(diverse_recs) >= top_n:
            break

    return pd.DataFrame(diverse_recs)




# Streamlit UI Configuration
st.set_page_config(page_title="Cinematic Movie Recommender", layout="wide", page_icon="🎬")

# Custom Stylesheet Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;800;900&family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: radial-gradient(circle at top, #12141c 0%, #07080c 100%) !important;
        color: #ccd2e3 !important;
        font-family: 'Outfit', sans-serif !important;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Cinzel', serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }

    /* Cinematic Header Container */
    .main-title-container {
        text-align: center;
        margin-top: 15px;
        margin-bottom: 25px;
        padding: 18px 0;
        border-top: 1px double rgba(212, 175, 55, 0.25);
        border-bottom: 1px double rgba(212, 175, 55, 0.25);
        width: 100%;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }

    .main-title {
        font-family: 'Cinzel', serif !important;
        background: linear-gradient(135deg, #f5f2eb 10%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 5px;
        margin: 0;
        text-shadow: 0px 2px 15px rgba(212, 175, 55, 0.15);
    }

    .main-subtitle {
        font-family: 'Outfit', sans-serif !important;
        color: #a49e93;
        font-size: 0.95rem;
        text-align: center;
        margin-top: 8px;
        margin-bottom: 0px;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
        border-bottom: 1px solid rgba(212, 175, 55, 0.12) !important;
        margin-bottom: 30px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        padding: 12px 28px !important;
        font-family: 'Cinzel', serif !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        color: #8c92a6 !important;
        letter-spacing: 1px;
        transition: all 0.25s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #f5f2eb !important;
        background-color: rgba(212, 175, 55, 0.05) !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #d4af37 !important;
        border-bottom: 2px solid #d4af37 !important;
    }

    /* Movie Grid and Card Layout */
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
        gap: 25px;
        margin-top: 20px;
    }

    @media (max-width: 768px) {
        .movie-grid {
            grid-template-columns: 1fr;
        }
        .movie-card {
            flex-direction: column !important;
            align-items: center;
        }
        .movie-poster-container {
            margin-right: 0 !important;
            margin-bottom: 15px;
            width: 100% !important;
            max-width: 160px;
        }
        .movie-info {
            margin-left: 0 !important;
            text-align: center;
        }
        .movie-title-row {
            justify-content: center !important;
        }
        .genre-container {
            justify-content: center;
        }
        .movie-stars {
            width: 100% !important;
            text-align: center;
            margin-bottom: 10px;
        }
    }

    .movie-card {
        display: flex;
        background: rgba(18, 23, 37, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 18px;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        overflow: hidden;
    }

    .movie-card:hover {
        transform: translateY(-5px);
        border-color: rgba(212, 175, 55, 0.25);
        box-shadow: 0 15px 35px rgba(212, 175, 55, 0.1);
        background: rgba(24, 30, 48, 0.65);
    }

    .movie-poster-container {
        flex-shrink: 0;
        width: 130px;
        height: 195px;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        background: #111422;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .movie-poster {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }

    .movie-card:hover .movie-poster {
        transform: scale(1.06);
    }

    .movie-info {
        flex-grow: 1;
        margin-left: 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .movie-title-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 10px;
    }

    .movie-title {
        margin: 0;
        font-family: 'Cinzel', serif !important;
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.25;
        letter-spacing: 0.5px;
    }

    .movie-meta {
        font-size: 0.78rem;
        color: #8c92a6;
        margin: 6px 0;
    }

    .genre-container {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin: 6px 0;
    }

    .genre-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    .rating-badge {
        background: rgba(255, 189, 89, 0.08);
        color: #ffbd59;
        border: 1px solid rgba(255, 189, 89, 0.25);
    }

    .match-badge {
        background: rgba(212, 175, 55, 0.1);
        color: #d4af37;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }

    .movie-plot {
        font-size: 0.82rem;
        color: #b3b9c9;
        line-height: 1.4;
        margin: 8px 0;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        height: 4.2em;
    }

    .movie-stars {
        font-size: 0.75rem;
        color: #9ab4c9;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 8px;
        margin-top: auto;
    }

    .badge-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }

    /* Button and Input Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #182030 0%, #0c1018 100%) !important;
        color: #ccd2e3 !important;
        border: 1px solid rgba(212, 175, 55, 0.25) !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-family: 'Cinzel', serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #d4af37 0%, #aa841c 100%) !important;
        color: #0c1018 !important;
        border-color: transparent !important;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.3) !important;
        transform: translateY(-2px) !important;
    }

    /* Slick info block */
    .info-card {
        background: rgba(18, 23, 37, 0.4);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'movies_df' not in st.session_state:
    movies = load_movies()
    movies = load_user_ratings(movies)
    st.session_state.movies_df = movies
    st.session_state.recs_dirty = True
    st.session_state.recommendations = None

# Header Title
st.markdown("""
<div class="main-title-container">
    <div class="main-title">CINEMATIC RECOMMENDATIONS</div>
    <div class="main-subtitle">A hyper-personalized, director-tuned AI movie recommendation engine</div>
</div>
""", unsafe_allow_html=True)

# Setup Tabs
tab_recs, tab_rate, tab_catalog = st.tabs(["🎯 AI Recommendations", "✍️ Rate Movies", "📋 Rated Catalog"])

# Calculate ratings count
rated_count = len(st.session_state.movies_df[st.session_state.movies_df[RATING_COL].notnull()])

# ==========================================
# TAB 1: AI RECOMMENDATIONS
# ==========================================
with tab_recs:
    if rated_count < 2:
        st.markdown(
            """<div class="info-card">
<h3>🍿 Welcome to Cinematic Recommendations!</h3>
<p style="color: #8c92a6;">To activate the AI recommendation engine, please click on the <strong>Rate Movies</strong> tab and rate at least 2 films.</p>
</div>""", 
            unsafe_allow_html=True
        )
    else:
        col_hdr1, col_hdr2 = st.columns([3, 1])
        with col_hdr1:
            st.markdown("### 🎯 Your Top 12 Movie Matches")
            st.write("These recommendations are tailored using your personalized director, genre, and critic affinity profiles.")
        with col_hdr2:
            if st.button("🔄 Refresh Recs", use_container_width=True):
                st.session_state.recs_dirty = True
                st.rerun()

        # Recalculate recommendations if dirty
        if st.session_state.recs_dirty or st.session_state.recommendations is None:
            with st.spinner("Analyzing profile & training XGBoost hybrid regressor..."):
                recs = train_and_recommend(st.session_state.movies_df, top_n=12)
                st.session_state.recommendations = recs
                st.session_state.recs_dirty = False
        
        recs = st.session_state.recommendations
        
        if recs is None or recs.empty:
            st.error("Could not generate recommendations. Make sure your ratings are valid.")
        else:
            # Fetch movie posters in parallel
            with st.spinner("Fetching cover art..."):
                posters = fetch_posters_parallel(recs['Series_Title'].tolist())
            
            # Display recommendations as a custom HTML grid
            grid_html = '<div class="movie-grid">'
            for idx, (_, row) in enumerate(recs.iterrows()):
                title = row['Series_Title']
                poster_url = posters.get(title)
                grid_html += render_movie_card_html(row, rank_num=idx+1, predicted_rating=row['Predicted_My_Rating'], poster_url=poster_url)
            grid_html += '</div>'
            
            st.markdown(grid_html, unsafe_allow_html=True)


# ==========================================
# TAB 2: RATE MOVIES
# ==========================================
with tab_rate:
    st.markdown("### ✍️ Search & Rate")
    st.write("Rate movies to immediately update your personalized recommendation model.")
    
    col_rate_left, col_rate_right = st.columns([3, 2], gap="large")
    
    with col_rate_left:
        # Selection controls
        movie_list = sorted(list(st.session_state.movies_df['Series_Title'].unique()))
        selected_movie = st.selectbox("🔍 Search the Catalog", movie_list)
        
        # Get current rating if exists
        movie_row = st.session_state.movies_df[st.session_state.movies_df['Series_Title'] == selected_movie].iloc[0]
        has_rating = pd.notnull(movie_row[RATING_COL])
        current_rating = float(movie_row[RATING_COL]) if has_rating else 5.0
        
        # Fetch poster for selected movie
        poster_url = get_movie_poster(selected_movie)
        
        # Display selected movie detail card
        st.markdown(render_detail_card(movie_row, poster_url=poster_url), unsafe_allow_html=True)
        
    with col_rate_right:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 🌟 Adjust Rating")
        
        if has_rating:
            st.markdown(f"<p style='color: #ffbd59;'>You rated this movie: <strong>{current_rating} ⭐</strong></p>", unsafe_allow_html=True)
        else:
            st.write("You haven't rated this movie yet.")
            
        rating_val = st.slider("Select Score (1.0 to 10.0)", 1.0, 10.0, current_rating, step=0.1)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        col_rate_btn1, col_rate_btn2 = st.columns(2)
        
        with col_rate_btn1:
            if st.button("Save Rating", use_container_width=True):
                st.session_state.movies_df.loc[st.session_state.movies_df['Series_Title'] == selected_movie, RATING_COL] = rating_val
                save_user_ratings(st.session_state.movies_df)
                st.session_state.recs_dirty = True
                st.toast(f"Saved rating for **{selected_movie}**: {rating_val}⭐!")
                st.rerun()
                
        with col_rate_btn2:
            if has_rating:
                if st.button("🗑️ Remove Rating", use_container_width=True):
                    st.session_state.movies_df.loc[st.session_state.movies_df['Series_Title'] == selected_movie, RATING_COL] = None
                    save_user_ratings(st.session_state.movies_df)
                    st.session_state.recs_dirty = True
                    st.toast(f"Removed rating for **{selected_movie}**")
                    st.rerun()


# ==========================================
# TAB 3: RATED CATALOG
# ==========================================
with tab_catalog:
    st.markdown("### 📋 Your Rated Catalog")
    
    rated_df = st.session_state.movies_df[st.session_state.movies_df[RATING_COL].notnull()][['Series_Title', 'IMDB_Rating', RATING_COL]].copy()
    
    if rated_df.empty:
        st.info("You haven't rated any movies yet. Go to the 'Rate Movies' tab to start rating!")
    else:
        st.write(f"You have rated a total of **{len(rated_df)}** movies.")
        rated_df.columns = ['Movie Title', 'IMDb Rating', 'Your Rating']
        st.dataframe(rated_df.sort_values(by='Your Rating', ascending=False), use_container_width=True, hide_index=True)
