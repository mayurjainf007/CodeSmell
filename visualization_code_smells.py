import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("model/code_smell_detector_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Load the test.csv file
test_data = pd.read_csv("data/test.csv")

# Handle missing values in the code_block column
test_data["code_block"] = test_data["code_block"].fillna("")  # Replace NaN with an empty string
code_blocks = test_data["code_block"]

# Detect code smells in each code block
predictions = []
for block in code_blocks:
    transformed_block = vectorizer.transform([block])
    predictions.append(model.predict(transformed_block)[0])

# Add predictions to the dataframe
test_data["predicted_smell"] = predictions

# Save plots with charts
def save_and_display_chart(filename, plot_func):
    plot_func()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

print("Graphs generated successfully:")

# 1. Bar Chart: Distribution of Code Smells
prediction_counts = pd.Series(predictions).value_counts()
save_and_display_chart("images/code_smell_distribution_bar.png", lambda: prediction_counts.plot(kind="bar", figsize=(10, 6), title="Code Smell Distribution"))
print("- 'code_smell_distribution_bar.png' generated")

# 2. Pie Chart: Proportion of Code Smells
save_and_display_chart("images/code_smell_distribution_pie.png", lambda: prediction_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, legend=True, figsize=(8, 8), title="Proportion of Code Smells"))
print("- 'code_smell_distribution_pie.png' generated")

# 3. Bar Chart: Top 10 Kernel-wise Code Smell Count
kernel_counts = test_data.groupby("kernel_id")["predicted_smell"].count().nlargest(10)
save_and_display_chart("images/kernel_wise_code_smell_count.png", lambda: kernel_counts.plot(kind="bar", figsize=(12, 6), title="Kernel-wise Code Smell Count"))
print("- 'kernel_wise_code_smell_count.png' generated")

# Additional Graph: Code Smell Type vs. Count (Cleaned additional_smells column)
results_df = pd.read_csv("report/code_smell_result_types.csv")

# List of valid code smell types
valid_smell_types = [
    "Long Method", "Duplicate Code", "Magic Numbers",
    "Too Many Arguments", "Dead Code", "Large Classes", "Complex Conditions"
]

# Clean and split the additional_smells column
additional_smells = results_df["additional_smells"].dropna().str.strip("[]").str.replace("'", "").str.split(", ")

# Filter to count only valid code smell types
all_smells = additional_smells.explode()
all_smells = all_smells[all_smells.isin(valid_smell_types)]
smell_counts = all_smells.value_counts()

# Generate and save bar chart for Additional Code Smells
save_and_display_chart("images/cleaned_additional_code_smell_counts.png", lambda: smell_counts.plot(kind="bar", figsize=(10, 6), title="Additional Code Smell Type vs. Count"))
print("- 'cleaned_additional_code_smell_counts.png' generated")
