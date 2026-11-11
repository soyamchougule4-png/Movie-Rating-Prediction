from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.data_loader import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "movie_rating_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="wide")
st.title("🎬 Movie Rating Prediction")
st.write("Estimate a movie's TMDB-style audience rating from historical movie metadata.")
st.info(
    "Educational estimate on a 0–10 historical rating scale. "
    "It is not a guarantee of audience response or an official TMDB rating."
)

if not MODEL_PATH.exists():
    st.warning(
        "The trained model was not found. Add both TMDB CSV files to the data folder "
        "and run `python run_pipeline.py` from the project folder."
    )
    st.code("python run_pipeline.py\nstreamlit run app.py", language="powershell")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.subheader("Predict a movie rating")
with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        budget = st.number_input("Estimated production budget (USD)", min_value=0, value=50000000, step=1000000)
        runtime = st.number_input("Runtime (minutes)", min_value=1, max_value=500, value=120)
        release_year = st.number_input("Release year", min_value=1900, max_value=2100, value=2025)
    with right:
        genres = st.multiselect(
            "Genres",
            ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
             "Drama", "Family", "Fantasy", "History", "Horror", "Music",
             "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller",
             "War", "Western"]
        )
        director = st.text_input("Director", value="Unknown")
        cast = st.text_input("Up to three lead actors (comma-separated)", value="")
    submitted = st.form_submit_button("Predict movie rating", type="primary")

if submitted:
    cast_names = [name.strip() for name in cast.split(",") if name.strip()][:3]
    row = pd.DataFrame([{
        "log_budget": float(np.log1p(budget)) if budget > 0 else np.nan,
        "runtime": float(runtime),
        "release_year": float(release_year),
        "genres_text": " ".join(genres),
        "director": director.strip() or "Unknown",
        "cast_text": " ".join(cast_names)
    }])
    try:
        prediction = float(model.predict(row)[0])
        prediction = max(0.0, min(10.0, prediction))
        st.metric("Estimated audience rating", f"{prediction:.2f} / 10")
        st.caption("This is a model estimate based on historical patterns, not an actual review score.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("Model evaluation")
metrics_file = OUTPUT_DIR / "model_metrics.json"
comparison_file = OUTPUT_DIR / "model_comparison.csv"
if metrics_file.exists():
    import json
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    st.success(f"Selected model: {metrics.get('best_model', 'Unknown')}")
    st.write(
        f"Training records: {metrics.get('training_rows', 'N/A')} · "
        f"Test records: {metrics.get('testing_rows', 'N/A')}"
    )
if comparison_file.exists():
    st.dataframe(pd.read_csv(comparison_file), use_container_width=True)

st.subheader("Exploratory data analysis")
chart_files = [
    "rating_distribution.png", "top_10_genres.png",
    "budget_vs_revenue.png", "runtime_distribution.png"
]
available = [name for name in chart_files if (OUTPUT_DIR / name).exists()]
if available:
    for start in range(0, len(available), 2):
        cols = st.columns(2)
        for col, filename in zip(cols, available[start:start + 2]):
            with col:
                st.image(str(OUTPUT_DIR / filename), caption=filename.replace("_", " ").replace(".png", "").title(), use_container_width=True)
else:
    st.caption("Charts appear here after the training pipeline runs.")
