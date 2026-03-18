import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "knn_model.pkl"
FEATURE_COLUMNS = ["duration", "contact", "previous", "housing", "pdays"]

def encode_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables as expected by the model."""
    df = df.copy()
    
    # Check if necessary columns exist
    missing_cols = set(["contact", "housing"]) - set(df.columns)
    if missing_cols:
        print(f"Error: Target data is missing essential columns: {missing_cols}")
        sys.exit(1)
        
    contact_mapping = {"cellular": 0, "telephone": 1, "unknown": 2}
    housing_mapping = {"no": 0, "yes": 1}
    
    # Apply mapping, defaulting to 2 (unknown) or 0 (no) for unseen values
    df["contact"] = df["contact"].str.lower().map(contact_mapping).fillna(2).astype(int)
    df["housing"] = df["housing"].str.lower().map(housing_mapping).fillna(0).astype(int)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Batch score bank marketing leads.")
    parser.add_argument("input_csv", help="Path to input CSV file containing customer data.")
    parser.add_argument("--output", "-o", default="scored_leads.csv", help="Output file path.")
    
    args = parser.parse_args()
    
    try:
        print("Loading latest KNN classification model...")
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}. Check if knn_model.pkl is present.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(args.input_csv, sep=';')
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(df)} records. Processing...")
    
    # Keep Original for Output
    output_df = df.copy()
    
    # Verify all expected columns are present
    missing_features = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_features:
        print(f"Error: Input dataset is missing required feature columns: {missing_features}")
        sys.exit(1)
        
    # Preprocess
    encoded_df = encode_inputs(df)
    features_df = encoded_df[FEATURE_COLUMNS]
    
    try:
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)
        
    # Append to output dataframe
    output_df["Predicted_Subscription"] = predictions
    output_df["Confidence_Score"] = probabilities.max(axis=1).round(3)
    output_df["Prediction_Label"] = output_df["Predicted_Subscription"].map({1: "High Priority", 0: "Lower Priority"})
    
    # Save
    output_df.to_csv(args.output, index=False)
    print(f"\nSuccess! Scored {len(df)} leads.")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
