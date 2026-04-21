# Marketing Campaign — Preprocessing & ML Model Training

An interactive Streamlit app that walks through a complete machine learning pipeline for predicting customer responses to a marketing campaign — from raw data cleaning to single-customer prediction.

## Features

- **Step-by-step preprocessing** — each transformation is applied interactively via buttons, so you can follow along in real time
- **Feature engineering** — derives `Age` from birth year and `Total_Spending` from product columns
- **Categorical encoding** — one-hot encodes `Education` and `Marital_Status`
- **Feature scaling** — StandardScaler on numerical columns, with the fitted scaler saved for inference
- **Three model options** — Logistic Regression, Random Forest, and XGBoost
- **Evaluation metrics** — Accuracy, F1 Score, and a Confusion Matrix
- **Single-customer prediction** — fill in customer attributes to get a campaign response prediction with confidence score

## Dataset

This app expects the [UCI Bank Marketing / Kaggle Marketing Campaign dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) as a CSV file.

**Update the file path** in `load_raw()` before running:

```python
return pd.read_csv(r"path/to/your/marketing_campaign.csv")
```

Expected columns include: `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Recency`, `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`, `NumWebVisitsMonth`, `Response`, and others.

## Installation

```bash
git clone https://github.com/your-username/marketing-campaign-ml.git
cd marketing-campaign-ml
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
```

Or install directly:

```bash
pip install streamlit pandas numpy scikit-learn xgboost
```

## Usage

1. Run the app and follow each numbered section top to bottom
2. Click each preprocessing button in order (Steps 1–6)
3. Select a model and click **Train & Evaluate**
4. Use the **Predict for a Single Customer** section to test individual inputs
5. Use the example buttons to quickly load a likely-responder or non-responder profile

## Project Structure

```
marketing-campaign-ml/
├── app.py                  # Main Streamlit application
├── README.md
└── images/                 # Output screenshots for README
    ├── 01_dataset_overview.png
    ├── 02_preprocessing.png
    ├── 03_model_evaluation.png
    └── 04_prediction.png
```
DASHBOARD
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a30343fc-5bb0-46ac-aa8a-0435e3e92527" />

DATA PRECROCESSING
<img width="1920" height="1080" alt="Screenshot 2026-04-21 084842" src="https://github.com/user-attachments/assets/dee48901-fd77-432e-9141-180109337841" />

ML MODEL TRAINING
<img width="1920" height="1080" alt="Screenshot 2026-04-15 233223" src="https://github.com/user-attachments/assets/a5811d60-bd70-4fc2-b1f0-9b3e907b8cb9" />

PREDICTION
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/b677639a-025e-47c0-bc3e-6d5ac6933986" />

