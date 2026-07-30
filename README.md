# Medical Insurance Cost Prediction — Capstone Project

Predicts an individual's annual medical insurance charges from demographic
and lifestyle attributes using regression models.

## Dataset
`data/insurance.csv` — 1,338 records, 7 columns:
`age, sex, bmi, children, smoker, region, charges`
(Classic public dataset from *Machine Learning with R*, Brett Lantz.)

## Project Structure
```
insurance_project/
├── data/insurance.csv          # Dataset
├── train_model.py              # Full training pipeline (EDA + 5 models + save best)
├── app.py                      # Streamlit web app: Predict tab + Dashboard tab
├── outputs/
│   ├── best_model.joblib       # Saved trained pipeline (preprocessing + model)
│   ├── best_model_info.json    # Metadata: best model name + metrics
│   ├── model_comparison.csv    # R2 / RMSE / MAE for all 5 models
│   ├── eda_overview.png        # Charges distribution, boxplots, scatterplots
│   ├── correlation_heatmap.png
│   ├── predicted_vs_actual.png
│   └── feature_importance.png
└── README.md
```

## How to Run

### 1. Train the model
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python train_model.py
```
This trains 5 models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting),
compares them, saves the best one to `outputs/best_model.joblib`, and
generates all EDA/evaluation plots.

### 2. Launch the interactive app
```bash
pip install streamlit plotly
streamlit run app.py
```
Opens a two-tab app:
- **🔮 Predict** — enter age, sex, BMI, children, smoker status, and region to
  get an instant predicted insurance cost.
- **📊 Dashboard** — interactive analytics view with:
  - KPI cards (record count, average/median charges, % smokers)
  - Filters for region, smoker status, and age range
  - Model comparison charts (R² and RMSE across all 5 models)
  - Charges distribution and boxplot by smoking status
  - Age vs. charges scatter plot (colored by BMI)
  - Average charges by region
  - Feature importance chart
  - Raw filtered data table

## Results

| Model              | R²     | RMSE   | MAE    |
|---------------------|--------|--------|--------|
| **Gradient Boosting** | **0.879** | **4329.57** | **2443.48** |
| Random Forest        | 0.865  | 4579.22 | 2555.75 |
| Linear Regression     | 0.784  | 5796.28 | 4181.19 |
| Lasso Regression      | 0.784  | 5797.04 | 4182.22 |
| Ridge Regression      | 0.783  | 5800.46 | 4193.20 |

**Gradient Boosting** is the best performer, explaining ~88% of the variance
in insurance charges. Ensemble methods substantially outperform linear
models here because charges have a non-linear relationship with smoking
status combined with age and BMI (smokers with high BMI see costs jump
disproportionately).

## Key Insights
- **Smoking status** is the single biggest driver of insurance cost.
- **Age** and **BMI** are the next most important numeric predictors.
- **Sex, children, and region** have comparatively minor effects.

## Next Steps / Possible Extensions
- Hyperparameter tuning (GridSearchCV) on Gradient Boosting for further gains
- Deploy the Streamlit app (Streamlit Community Cloud / Render / HuggingFace Spaces)
- Add SHAP explainability for individual predictions
- Collect a larger / more recent real-world dataset for production use
