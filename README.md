# Movie Rating Prediction with Python

A complete educational machine-learning project that estimates a movie's historical TMDB `vote_average` (0–10) using metadata such as genre, director, cast, budget, runtime and release year.

## Dataset required

This project uses the TMDB 5000 Movie Metadata dataset. Put these two files in `data/`:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Dataset source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

**The dataset CSVs are not included in this ZIP.** If you already have them in your current `D:\SystemTorn\data` folder, copy them into the `data` folder of this extracted project.

## Windows PowerShell setup

Open PowerShell in the extracted project folder. Example:

```powershell
cd D:\SystemTorn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a working `.venv`, you can activate it and skip creating it.

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the complete project

First make sure both CSV files are in `data/`, then run:

```powershell
python run_pipeline.py
```

This performs:
1. CSV loading and validation.
2. Merge of movies and credits using the movie ID.
3. Exploratory data analysis and chart generation.
4. JSON parsing for genres, director and cast.
5. Missing-value handling and feature engineering.
6. Training of a Dummy baseline, Random Forest and Extra Trees regressors.
7. Evaluation using MAE, RMSE and R².
8. Selection and saving of the best model.

Then start the web application:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Expected project structure

```text
Movie_Rating_Prediction_Complete/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── models/
│   └── movie_rating_model.joblib
├── outputs/
│   ├── eda_summary.json
│   ├── model_comparison.csv
│   ├── model_metrics.json
│   └── EDA chart PNG files
└── src/
    ├── data_loader.py
    ├── eda.py
    ├── preprocessing.py
    └── train_model.py
```

Generated model and output files appear after the pipeline completes.

## Important notes

- The target is `vote_average`, not a future guaranteed audience score.
- The pipeline excludes `vote_count` from input features to avoid using a value that reflects audience activity after release.
- Movies with missing or invalid ratings and zero vote counts are excluded from supervised training.
- Evaluation results depend on dataset quality and the random train/test split.
- A model prediction is an estimate, not an official TMDB rating.
