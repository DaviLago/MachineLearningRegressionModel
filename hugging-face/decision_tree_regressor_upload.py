import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from huggingface_hub import upload_file, login
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load the dataset with error handling
csv_path = 'dataset/insurance.csv'
if not os.path.exists(csv_path):
    print(f"Error: File '{csv_path}' not found.")
    sys.exit(1)

df = pd.read_csv(csv_path).drop_duplicates()

# Features and target
X = df.drop(columns=['charges'])
y = df['charges']

# Categorical columns to encode
categorical_cols = ['smoker', 'region', 'sex']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Full pipeline with DecisionTreeRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=4)
)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate and print R² score
r2_score = pipeline.score(X_test, y_test)
print(f"R² score on test set: {r2_score:.4f}")

# Save your trained pipeline
model_path = "hugging-face/decision_tree_pipeline.joblib"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(pipeline, model_path)

# Login to Hugging Face Hub and upload
if hf_token:
    login(token=hf_token)
    if os.path.exists(model_path):
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo="decision_tree_pipeline.joblib",
            repo_id="DaviLago/MachineLearningRegressionModel",
            repo_type="model"
        )
        print("Model uploaded to Hugging Face Hub.")
    else:
        print(f"Model file '{model_path}' not found. Skipping upload.")
else:
    print("HF_TOKEN not found. Skipping Hugging Face upload.")