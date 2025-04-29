# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load Dataset
df = pd.read_csv('healthcare_dataset.csv')

# 3. Data Cleaning
print("Cleaning data...")

# Drop irrelevant columns
columns_to_drop = ['Patient ID', 'Name', 'Date', 'Timestamp', 'Doctor']
for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Fill missing numeric values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing categorical values with mode and convert to category type
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
    df[col] = df[col].astype('category')

print("\nData Types:\n", df.dtypes)

# 4. Summary Statistics
print("\nSummary:\n", df.describe(include='all'))

# 5. Plot Target Column Distribution
if 'Outcome' in df.columns:
    sns.countplot(x='Outcome', data=df, palette='Set2')
    plt.title("Outcome Distribution")
    plt.show()

# 6. Plot Numeric Feature Distributions
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"{col} Distribution")
    plt.show()

# 7. Correlation Heatmap (Only One Heatmap)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 8. Boxplots by Outcome (if exists)
if 'Outcome' in df.columns:
    for col in numeric_cols:
        sns.boxplot(x='Outcome', y=col, data=df, palette='pastel')
        plt.title(f"{col} by Outcome")
        plt.show()

# 9. Countplots for Categorical Features
cat_cols = df.select_dtypes(include='category').columns
for col in cat_cols:
    if 'Outcome' in df.columns:
        sns.countplot(x=col, hue='Outcome', data=df, palette='muted')
        plt.title(f"{col} vs Outcome")
        plt.show()

# 10. Gender Pie Chart (if exists)
if 'Gender' in df.columns:
    df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    plt.title("Gender Distribution")
    plt.ylabel("")
    plt.show()

# 11. Data Preparation for Modeling
df_ready = pd.get_dummies(df, drop_first=True)
print("\nData ready for modeling. Shape:", df_ready.shape)
