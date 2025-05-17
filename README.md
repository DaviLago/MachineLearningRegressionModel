# MachineLearningRegressionModel

This project demonstrates a machine learning regression pipeline using a Decision Tree Regressor to predict insurance charges based on demographic and lifestyle features.

## Dataset

The model is trained on the `insurance.csv` dataset, which contains the following columns:
- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`
- `charges` (target)

## Features

- Data preprocessing with one-hot encoding for categorical variables.
- Train/test split with stratification.
- Decision Tree regression with hyperparameter tuning.
- Model evaluation using R² score.
- Model serialization with `joblib`.
- Optional upload to [Hugging Face Hub](https://huggingface.co/).

## Usage

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare your environment:**
   - Place `insurance.csv` in the `dataset/` folder.
   - Create a `.env` file with your Hugging Face token (optional for model upload):
     ```
     HF_TOKEN=your_huggingface_token
     ```

3. **Run the training and upload script:**
   ```sh
   python hugging-face/decision_tree_regressor_upload.py
   ```

## Model Upload

If `HF_TOKEN` is set, the trained model will be uploaded to the Hugging Face Hub under the repository:
```
DaviLago/MachineLearningRegressionModel
```

## License

This project is licensed under the MIT License.
