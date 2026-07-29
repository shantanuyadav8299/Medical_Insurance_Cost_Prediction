"""
Medical Insurance Cost Prediction - Capstone Project
======================================================
Predicts individual medical insurance charges using demographic and
lifestyle attributes (age, sex, bmi, children, smoker, region).

Pipeline:
1. Load & explore data
2. Preprocess (encode categoricals, scale numerics)
3. Train multiple regression models
4. Evaluate & compare (R2, RMSE, MAE)
5. Save the best model + preprocessing pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = "data/insurance.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

# ---------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (saved as images)
# ---------------------------------------------------------------
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(df["charges"], kde=True, ax=axes[0, 0], color="#2E86AB")
axes[0, 0].set_title("Distribution of Charges")

sns.boxplot(x="smoker", y="charges", data=df, ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Charges by Smoking Status")

sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, ax=axes[1, 0], palette="Set1", alpha=0.7)
axes[1, 0].set_title("BMI vs Charges (colored by Smoker)")

sns.scatterplot(x="age", y="charges", hue="smoker", data=df, ax=axes[1, 1], palette="Set1", alpha=0.7)
axes[1, 1].set_title("Age vs Charges (colored by Smoker)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_overview.png", dpi=150)
plt.close()

# Correlation heatmap (numeric only)
plt.figure(figsize=(6, 5))
numeric_df = df.copy()
numeric_df["smoker_num"] = (numeric_df["smoker"] == "yes").astype(int)
corr = numeric_df[["age", "bmi", "children", "smoker_num", "charges"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X = df.drop(columns=["charges"])
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_cols = ["sex", "smoker", "region"]
numeric_cols = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
    ]
)

# ---------------------------------------------------------------
# 4. TRAIN MULTIPLE MODELS & COMPARE
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    results.append({"Model": name, "R2": round(r2, 4), "RMSE": round(rmse, 2), "MAE": round(mae, 2)})
    fitted_pipelines[name] = pipe
    print(f"{name}: R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 5. SELECT BEST MODEL & SAVE
# ---------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

joblib.dump(best_pipeline, f"{OUT_DIR}/best_model.joblib")

with open(f"{OUT_DIR}/best_model_info.json", "w") as f:
    json.dump(
        {
            "best_model": best_model_name,
            "metrics": results_df.iloc[0].to_dict(),
            "features": {"numeric": numeric_cols, "categorical": categorical_cols},
        },
        f,
        indent=2,
    )

# ---------------------------------------------------------------
# 6. PREDICTED vs ACTUAL PLOT FOR BEST MODEL
# ---------------------------------------------------------------
preds_best = best_pipeline.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_best, alpha=0.5, color="#2E86AB")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title(f"Predicted vs Actual — {best_model_name}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/predicted_vs_actual.png", dpi=150)
plt.close()

# Feature importance (if Random Forest / Gradient Boosting is best)
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    feature_names = numeric_cols + list(
        best_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_cols)
    )
    importances = best_pipeline.named_steps["model"].feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(x="importance", y="feature", data=feat_imp, palette="viridis")
    plt.title(f"Feature Importance — {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

print("\nAll outputs saved to the 'outputs' folder.")
