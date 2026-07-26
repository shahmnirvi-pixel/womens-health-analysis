Here is the complete, fully formatted final code script. It fulfills all four rubric requirements with no missing line breaks or syntax errors.

Copy and paste this into a fresh code cell in Google Colab or Jupyter Notebook and run it:

Python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# ==========================================
# 1. LOAD DATA
# ==========================================
url = "https://data.cdc.gov/api/views/d2rk-yvas/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)

# ==========================================
# 2. DATA CLEANING (Requirement #1: Clean 5+ Variables)
# ==========================================
# Filter specifically for Women's Health between 2011 and 2013
womens_df = df[
    (df["Class"] == "Women's Health") & (df["Year"].isin([2011, 2012, 2013]))
].copy()

# List of 5 key variables to clean and format
clean_cols = [
    "Data_value",
    "Sample_Size",
    "Year",
    "Break_Out_Category",
    "Locationabbr",
]

# Drop missing values and cast data types explicitly across all 5 variables
womens_df = womens_df.dropna(subset=clean_cols)
womens_df["Data_value"] = pd.to_numeric(womens_df["Data_value"])
womens_df["Sample_Size"] = pd.to_numeric(womens_df["Sample_Size"])
womens_df["Year"] = womens_df["Year"].astype(int)
womens_df["Break_Out_Category"] = womens_df["Break_Out_Category"].astype(str)
womens_df["Locationabbr"] = womens_df["Locationabbr"].astype(str)

print(f"Cleaned Dataset Shape: {womens_df.shape}\n")

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (Requirement #2: Analyzed 5 Variables)
# ==========================================
print("=== EDA Summary (5 Variables) ===")
print(womens_df[clean_cols].describe(include="all"))

# ==========================================
# 4. DATA VISUALIZATION (Requirement #3: Minimum 5 Charts)
# ==========================================
plt.figure(figsize=(15, 12))

# Chart 1: Boxplot of Rates by Year
plt.subplot(3, 2, 1)
sns.boxplot(data=womens_df, x="Year", y="Data_value", palette="Set2")
plt.title("1. Distribution of Health Values by Year")

# Chart 2: Histogram of Sample Sizes
plt.subplot(3, 2, 2)
sns.histplot(womens_df["Sample_Size"], bins=20, kde=True, color="purple")
plt.title("2. Sample Size Distribution")

# Chart 3: Barplot by Demographic Breakdown
plt.subplot(3, 2, 3)
sns.barplot(
    data=womens_df,
    x="Data_value",
    y="Break_Out_Category",
    estimator=np.mean,
    errorbar=None,
)
plt.title("3. Average Value by Demographic Category")

# Chart 4: Scatter Plot (Sample Size vs Data Value)
plt.subplot(3, 2, 4)
sns.scatterplot(
    data=womens_df,
    x="Sample_Size",
    y="Data_value",
    hue="Year",
    alpha=0.6,
    palette="viridis",
)
plt.title("4. Sample Size vs. Health Value Rate")

# Chart 5: Top 10 States by Response Count
plt.subplot(3, 2, 5)
top_locations = womens_df["Locationabbr"].value_counts().head(10)
top_locations.plot(kind="bar", color="skyblue")
plt.title("5. Top 10 Participating States")

plt.tight_layout()
plt.savefig("womens_health_eda_5charts.png")
plt.show()

# ==========================================
# 5. MACHINE LEARNING (Requirement #4: Minimum 5 Models)
# ==========================================
# Features and Target selection
features = ["Year", "Sample_Size", "Break_Out_Category", "Locationabbr"]
X = pd.get_dummies(womens_df[features], drop_first=True)
y = womens_df["Data_value"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize 5 distinct models
models = {
    "1. Linear Regression": LinearRegression(),
    "2. Ridge Regression": Ridge(),
    "3. K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "4. Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "5. Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

print("\n=== Model Performance Evaluation (5 Models) ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} -> MAE: {mae:.2f} | R2 Score: {r2:.2f}")
