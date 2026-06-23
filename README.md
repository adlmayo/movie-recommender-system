# Movie Recommender System

A content-based movie recommendation system built with Python. Given a movie title, it suggests 5 similar movies based on overview, genres, keywords, cast, and director.

## How It Works

1. **Data loading** — merges the TMDB 5000 movies and credits datasets on title.
2. **Feature selection** — keeps `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`.
3. **Cleaning** — drops rows with missing values and parses stringified JSON/list columns (`genres`, `keywords`, `cast`, `crew`) using `ast.literal_eval`.
4. **Feature engineering**:
   - Keeps the top 3 cast members per movie.
   - Extracts only the director from the crew.
   - Strips spaces from multi-word names/tags (e.g. "Science Fiction" → "ScienceFiction") so they're treated as single tokens.
   - Combines `overview`, `genres`, `keywords`, `cast`, and `crew` into a single `tags` column.
5. **Text processing**:
   - Lowercases all tags.
   - Applies Porter stemming (via NLTK) to normalize words.
6. **Vectorization** — converts tags into vectors using `CountVectorizer` (bag-of-words, top 5000 features, English stop words removed).
7. **Similarity** — computes pairwise **cosine similarity** between all movie vectors.
8. **Recommendation** — for a given movie, returns the top 5 most similar movies by cosine similarity score.
9. **Export** — saves the processed dataframe and similarity matrix as pickle files (`movies.pkl`, `movie_dict.pkl`, `similarity.pkl`) for reuse (e.g. in a web app).

## Dataset

This project uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata):
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Download both files from Kaggle and place them in the same directory as the notebook before running it.

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
nltk
```

Install with:
```bash
pip install pandas numpy matplotlib scikit-learn nltk
```

## Usage

1. Place the dataset CSVs in the project directory.
2. Run all cells in `movie-recommender-system.ipynb` in order.
3. Use the `recommend()` function to get suggestions:

```python
recommend('Avatar')
```

Example output:
```
Lifeforce
Aliens vs Predator: Requiem
Battle: Los Angeles
Titan A.E.
Independence Day
```

## Output Files

Running the notebook produces three pickle files that can be loaded by a separate application (e.g. a Flask/Streamlit app):

| File | Contents |
|---|---|
| `movies.pkl` | Processed movie dataframe (id, title, tags) |
| `movie_dict.pkl` | Same data as a dictionary |
| `similarity.pkl` | Precomputed cosine similarity matrix |

## Notes

- This is a **content-based** recommender — it only uses movie metadata, not user ratings or behavior, so it doesn't personalize recommendations per user.
- Recommendation quality depends on the quality and completeness of the `overview`, `genres`, `keywords`, `cast`, and `crew` fields.
