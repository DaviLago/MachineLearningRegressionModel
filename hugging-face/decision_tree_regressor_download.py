from huggingface_hub import hf_hub_download
import joblib
from dotenv import load_dotenv
import os
import sys
import pandas as pd

def main():
    # Load environment variables
    load_dotenv()
    repo_id = os.getenv("HF_REPO_ID")

    # Download the model file
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename="decision_tree_pipeline.joblib")
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- Hard-coded input values for prediction ---
    # Adjust these values as needed
    input_data = {
        "age": [19],
        "sex": ["female"],
        "bmi": [27.9],
        "children": [0],
        "smoker": ["yes"],
        "region": ["southwest"]
    }
    X = pd.DataFrame(input_data)

    print("Input features for prediction:")
    print(X.to_string(index=False))

    # Predict
    try:
        predicted_charge = model.predict(X)[0]
        print(f"Predicted insurance charge: {predicted_charge:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()