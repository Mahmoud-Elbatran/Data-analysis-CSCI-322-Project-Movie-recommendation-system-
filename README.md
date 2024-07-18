# Movie Recommendation System

This project implements a movie recommendation system using collaborative filtering and clustering techniques. The system processes a dataset of movie ratings and titles to provide users with personalized movie recommendations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Clustering](#clustering)
- [Search Engine](#search-engine)
- [Recommendation System](#recommendation-system)
- [Interactive Widgets](#interactive-widgets)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Mahmoud-Elbatran/Data-analysis-CSCI-322-Project-Movie-recommendation-system-
   cd movierecommendation
   ```

2. Install the required Python packages:
   ```sh
   pip install pandas numpy seaborn ipywidgets matplotlib scikit-learn
   ```

3. Ensure you have the `ratings.csv` and `movies.csv` files in the project directory.

## Usage

Run the Jupyter notebook containing the code to initialize the recommendation system and interact with the widgets.

## Project Structure

- `ratings.csv`: Contains user ratings for different movies.
- `movies.csv`: Contains movie titles and genres.
- `README.md`: Project documentation.
- `movie_recommendation.ipynb`: Jupyter notebook containing the project code.

## Data Processing

### Ratings Data

The ratings data is loaded and processed to check for missing values and ensure data integrity.

```python
import pandas as pd

Ratings = pd.read_csv("ratings.csv")
Ratings.info()
Ratings.isna().sum()
```

### Movies Data

The movies data is also loaded and checked for missing values.

```python
Movies = pd.read_csv("movies.csv")
Movies.info()
Movies.isna().sum()
```

## Clustering

The movies are clustered based on their genres using K-Means clustering.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features_for_clustering = Movies[['genres']].copy()
features_for_clustering['genres'] = features_for_clustering['genres'].astype('category').cat.codes
X = StandardScaler().fit_transform(features_for_clustering)

k_clusters = 5
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
Movies['cluster'] = kmeans.fit_predict(X)
```

## Search Engine

The search engine uses a TF-IDF vectorizer to transform movie titles and find the most similar movies based on cosine similarity.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def Clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

Movies["Clean_title"] = Movies["title"].apply(Clean_title)

Vectorizer = TfidfVectorizer(ngram_range=(1,2))
TFIDF = Vectorizer.fit_transform(Movies["Clean_title"])

def search(title):
    title = Clean_title(title)
    Query_vec = Vectorizer.transform([title])
    Similarity = cosine_similarity(Query_vec, TFIDF).flatten()
    Indices = np.argpartition(Similarity, -5)[-5:]
    Results = Movies.iloc[Indices].iloc[::-1]
    return Results
```

## Recommendation System

The recommendation system identifies users with similar tastes and recommends movies highly rated by them.

```python
def find_similar_movies(movie_id):
    similar_users = Ratings[(Ratings["movieId"] == movie_id) & (Ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = Ratings[(Ratings["userId"].isin(similar_users)) & (Ratings["rating"] > 4)]["movieId"]
    
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    
    all_users = Ratings[(Ratings["movieId"].isin(similar_user_recs.index)) & (Ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    
    return rec_percentages.head(10).merge(Movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
```

## Interactive Widgets

The project uses `ipywidgets` to create interactive search and recommendation widgets.

```python
from IPython.display import display
import ipywidgets as widgets

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)
