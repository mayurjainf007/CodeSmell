import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Load test data with error handling
try:
    test_data = pd.read_csv("data/test.csv")
except FileNotFoundError:
    print("Error: 'data/test.csv' not found. Ensure the file exists in the data directory.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: 'data/test.csv' is empty or corrupted.")
    exit()

# Fill any NaN values in 'code_block' with an empty string
test_data['code_block'] = test_data['code_block'].fillna("")

# Load the saved model and vectorizer with error handling
try:
    model = joblib.load("model/code_smell_detector_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
except FileNotFoundError:
    print("Error: Model or vectorizer not found in the 'model' directory.")
    exit()
except joblib.externals.loky.process_executor.TerminatedWorkerError:
    print("Error: Corrupted model or vectorizer files.")
    exit()

# Transform the code snippets in test_data using the vectorizer
X_test = vectorizer.transform(test_data['code_block'])

# Predict code smells
test_data['code_smell_prediction'] = model.predict(X_test)

# Add interpretive labels for readability in the report
test_data['smell_label'] = test_data['code_smell_prediction'].apply(lambda x: 'Code Smell' if x == 1 else 'No Code Smell')

# Generate summary report at kernel level
smell_summary = test_data.groupby('kernel_id').agg(
    total_code_blocks=('code_block', 'count'),
    total_code_smells=('code_smell_prediction', 'sum')
).reset_index()

# Calculate the percentage of code smells for each notebook and format to two decimal places
smell_summary['percent_code_smells'] = (smell_summary['total_code_smells'] / smell_summary['total_code_blocks'] * 100).round(2)

# Create the report directory if it does not exist
os.makedirs("report", exist_ok=True)

# Save the detailed and summary reports
test_data.to_csv("report/detailed_code_smell_report.csv", index=False)
smell_summary.to_csv("report/summary_code_smell_report.csv", index=False)

# Print a clear summary of the report generation
print("Reports generated successfully:")
print("- 'detailed_code_smell_report.csv': Detailed predictions for each code block")
print("- 'summary_code_smell_report.csv': Summary of code smells per notebook with percentages")

# Print concise summaries for each kernel_id for readability
print("\nSummary Report:")
for _, row in smell_summary.iterrows():
    print(f"Kernel ID: {row['kernel_id']} - Total Blocks: {row['total_code_blocks']}, "
          f"Code Smells: {row['total_code_smells']} ({row['percent_code_smells']}%)")
