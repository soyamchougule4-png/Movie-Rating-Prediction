import sys
from src.data_loader import load_data, PROJECT_ROOT
from src.eda import run_eda
from src.train_model import train

def main():
    print("=" * 64)
    print("MOVIE RATING PREDICTION - FULL TRAINING PIPELINE")
    print("=" * 64)
    print(f"Project folder: {PROJECT_ROOT}")
    try:
        movies, credits, merged = load_data()
        print(f"Movies : {movies.shape[0]} rows x {movies.shape[1]} columns")
        print(f"Credits: {credits.shape[0]} rows x {credits.shape[1]} columns")
        print(f"Merged : {merged.shape[0]} rows x {merged.shape[1]} columns")

        print("\n[1/2] Running exploratory data analysis...")
        summary = run_eda(movies)
        print("EDA summary saved to outputs/eda_summary.json")
        print(f"Average rating: {summary.get('average_rating', 'N/A')}")

        print("\n[2/2] Training and comparing regression models...")
        results, metrics = train(merged)
        print("\nMODEL COMPARISON")
        print(results.to_string(index=False))
        print(f"\nBest model: {metrics['best_model']}")
        print("\nSUCCESS: Model saved to models/movie_rating_model.joblib")
        print("Reports and charts saved in outputs/")
        print("\nNext command: streamlit run app.py")
    except Exception as exc:
        print(f"\nPIPELINE FAILED: {exc}")
        print("\nCheck that both TMDB CSV files are inside the project's data folder.")
        raise

if __name__ == "__main__":
    main()
