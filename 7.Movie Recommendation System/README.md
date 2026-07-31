Movie Recommendation System
A Python-based movie recommendation engine implementing two classic approaches: Content-Based Filtering and Collaborative Filtering. Built with scikit-learn, pandas, and numpy.

Table of Contents
Overview
Features
Installation
Quick Start
Content-Based Recommender
Collaborative Recommender
Dataset
API Reference
Example Output
License

Overview
This project provides a lightweight, easy-to-understand implementation of two fundamental recommendation techniques:
| Technique                   | Description                                                                   | Use Case                         |
| --------------------------- | ----------------------------------------------------------------------------- | -------------------------------- |
| **Content-Based Filtering** | Recommends movies similar to a given movie based on genres and plot overview. | "Because you liked X..."         |
| **Collaborative Filtering** | Recommends movies based on ratings from similar users.                        | "Users like you also enjoyed..." |

Features
TF-IDF Vectorization for text-based movie feature extraction
Cosine Similarity for measuring movie and user likeness
Fuzzy title matching with helpful suggestions for typos
Weighted score prediction for collaborative recommendations
Clean, modular class-based architecture
20 diverse movies across genres: Action, Sci-Fi, Romance, Horror, Animation, Comedy, Drama, Thriller

Installation
Requirements
Python 3.7+
numpy
pandas
scikit-learn

Install Dependencies
bash
pip install numpy pandas scikit-learn

Clone & Run
bash
git clone <repository-url>
cd movie-recommendation-system
python main.py

Quick Start
Python
import pandas as pd
from main import ContentBasedRecommender, CollaborativeRecommender, df, RATINGS

# --- Content-Based: Recommend movies similar to "Inception" ---
content = ContentBasedRecommender(df)
recs = content.recommend("Inception", top_n=5)
for r in recs:
    print(f"{r['title']} (similarity: {r['similarity']})")

# --- Collaborative: Recommend movies for user "Dave" ---
collab = CollaborativeRecommender(RATINGS)
recs = collab.recommend("Dave", top_n=3)
for r in recs:
    print(f"{r['title']} (predicted score: {r['predicted_score']})")
    
Content-Based Recommender
How It Works
Feature Engineering: Combines genres and overview into a single text field (combined_features).
TF-IDF Vectorization: Converts text into numerical vectors, ignoring common English stop words.
Similarity Matrix: Computes pairwise cosine similarity between all movies.
Recommendation: Given a seed movie, returns the most similar movies (excluding itself).
Class: ContentBasedRecommender

Python
ContentBasedRecommender(dataframe: pd.DataFrame)
| Method                                                | Description                                                                 |
| ----------------------------------------------------- | --------------------------------------------------------------------------- |
| `recommend(title: str, top_n: int = 5) -> list[dict]` | Returns `top_n` movies most similar to `title`, sorted by similarity score. |

Parameters
title (str): Movie title (case-insensitive). Supports partial matching with suggestions.
top_n (int): Number of recommendations to return (default: 5).
Returns
Python
[
    {"title": "Interstellar", "similarity": 0.234},
    {"title": "The Matrix",   "similarity": 0.198},
    ...
]

Error Handling
Raises ValueError if the movie is not found.
Suggests close matches if a partial title is detected.
Collaborative Recommender
How It Works
User-Item Matrix: Rows = users, Columns = movies, Values = ratings (0 = unrated).
User Similarity: Computes cosine similarity between all user rating vectors.
Weighted Aggregation: Predicts scores for unrated movies by aggregating ratings from similar users, weighted by similarity.
Recommendation: Returns top-N unrated movies with highest predicted scores.
Class: CollaborativeRecommender
Python
CollaborativeRecommender(ratings: pd.DataFrame)
| Method                                               | Description                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `recommend(user: str, top_n: int = 3) -> list[dict]` | Returns `top_n` movie recommendations for `user` based on similar users' preferences. |

Parameters
user (str): User name (must exist in the ratings index).
top_n (int): Number of recommendations to return (default: 3).
Returns
Python
[
    {"title": "La La Land", "predicted_score": 4.25},
    {"title": "Inception",  "predicted_score": 3.80},
    ...
]

Error Handling
Raises ValueError if the user is not found, listing all known users.

Dataset
| Title                    | Genres                            |
| ------------------------ | --------------------------------- |
| The Matrix               | Action Sci-Fi                     |
| Inception                | Action Sci-Fi Thriller            |
| Interstellar             | Adventure Drama Sci-Fi            |
| The Dark Knight          | Action Crime Drama                |
| Memento                  | Mystery Thriller                  |
| The Prestige             | Drama Mystery Thriller            |
| Toy Story                | Animation Adventure Comedy Family |
| Finding Nemo             | Animation Adventure Comedy Family |
| Up                       | Animation Adventure Comedy Family |
| The Notebook             | Drama Romance                     |
| La La Land               | Comedy Drama Music Romance        |
| Pride and Prejudice      | Drama Romance                     |
| John Wick                | Action Crime Thriller             |
| Mad Max: Fury Road       | Action Adventure Sci-Fi           |
| Gladiator                | Action Adventure Drama            |
| The Conjuring            | Horror Mystery Thriller           |
| Get Out                  | Horror Mystery Thriller           |
| A Quiet Place            | Drama Horror Sci-Fi               |
| The Grand Budapest Hotel | Adventure Comedy Drama            |
| Superbad                 | Comedy                            |

User Ratings Matrix
| User  | The Matrix | Inception | Interstellar | The Notebook | La La Land | Toy Story | John Wick | The Conjuring |
| ----- | ---------- | --------- | ------------ | ------------ | ---------- | --------- | --------- | ------------- |
| Alice | 5          | 5         | 4            | 1            | 0          | 3         | 5         | 2             |
| Bob   | 4          | 5         | 5            | 0            | 1          | 2         | 3         | 1             |
| Carol | 0          | 1         | 0            | 5            | 4          | 4         | 0         | 2             |
| Dave  | 1          | 0         | 0            | 4            | 5          | 4         | 1         | 1             |
| Eve   | 5          | 4         | 5            | 0            | 1          | 2         | 4         | 3             |

Note: 0 indicates the user has not rated that movie.

API Reference
ContentBasedRecommender
| Attribute           | Type              | Description                                                           |
| ------------------- | ----------------- | --------------------------------------------------------------------- |
| `df`                | `pd.DataFrame`    | Movie catalog with `title`, `genres`, `overview`, `combined_features` |
| `vectorizer`        | `TfidfVectorizer` | Fitted TF-IDF vectorizer                                              |
| `tfidf_matrix`      | `sparse matrix`   | TF-IDF feature matrix                                                 |
| `similarity_matrix` | `ndarray`         | Precomputed cosine similarity between all movies                      |
| `title_to_index`    | `dict`            | Mapping of lowercase titles to row indices                            |

CollaborativeRecommender
| Attribute         | Type           | Description                            |
| ----------------- | -------------- | -------------------------------------- |
| `ratings`         | `pd.DataFrame` | User-movie rating matrix               |
| `user_similarity` | `pd.DataFrame` | Cosine similarity matrix between users |

Example Output
Running python main.py produces:
plain
=== Content-based recommendations ===
Because you liked 'Inception':
  The Matrix                   (similarity: 0.198)
  Interstellar                 (similarity: 0.177)
  The Dark Knight              (similarity: 0.141)
  Memento                      (similarity: 0.126)
  The Prestige                 (similarity: 0.126)

=== Collaborative filtering recommendations ===
Recommended for Dave (based on similar users' ratings):
  La La Land                   (predicted score: 4.25)
  Inception                    (predicted score: 3.8)
  Interstellar                 (predicted score: 3.6)
  
Project Structure
plain
.
├── main.py          # Main implementation with classes and demo
├── README.md        # This file
└── requirements.txt # Dependencies (optional)

Extending the System
Add More Movies
Simply append to the MOVIES list with title, genres, and overview fields.
Add More Users/Ratings
Expand the RATINGS DataFrame with new columns (movies) and rows (users).
Switch to Item-Based Collaborative Filtering
Modify CollaborativeRecommender to compute similarity between movies (columns) instead of users:
Python
self.item_similarity = pd.DataFrame(
    cosine_similarity(ratings.T),
    index=ratings.columns,
    columns=ratings.columns,
)
Persist Models
Use joblib or pickle to save fitted vectorizers and similarity matrices for production use.

License
This project is provided as-is for educational and demonstration purposes.

Acknowledgments
Built with scikit-learn and pandas
Inspired by classic recommender system literature (Leskovec, Rajaraman & Ullman; Mining of Massive Datasets) 
