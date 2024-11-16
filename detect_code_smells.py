
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the saved model and vectorizer
model = joblib.load("model/code_smell_detector_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Define known code smell types with NLP techniques or rules for better classification
code_smell_types = {
    "Long Method": lambda line: len(line.split()) > 20,
    "Duplicate Code": lambda line: "duplicate" in line.lower(),
    "Magic Numbers": lambda line: bool(re.search(r'\b\d+\b', line)) and not any(x in line for x in ["for", "range"]),
    "Too Many Arguments": lambda line: "def " in line and line.count(",") > 3,
    "Dead Code": lambda line: "pass" in line or "TODO" in line,
    "Large Classes": lambda line: line.strip().startswith("class ") and line.count("def ") > 5,
    "Complex Conditions": lambda line: bool(re.search(r"if|while .*and.*or.*", line))
}

# Load the test.csv file
test_data = pd.read_csv("data/test.csv")

# Handle missing values in the code_block column without inplace=True
test_data["code_block"] = test_data["code_block"].fillna("")  # Replace NaN with an empty string
code_blocks = test_data["code_block"]

# Detect code smells in each code block
results = []
for idx, block in enumerate(code_blocks):
    smells = []
    transformed_block = vectorizer.transform([block])
    prediction = model.predict(transformed_block)[0]
    
    # Apply additional rules for specific smell types
    for smell, rule in code_smell_types.items():
        if rule(block):
            smells.append(smell)
    
    results.append({
        "code_blocks_index": test_data.loc[idx, "code_blocks_index"],
        "kernel_id": test_data.loc[idx, "kernel_id"],
        "code_block_id": test_data.loc[idx, "code_block_id"],
        "predicted_smell": prediction,
        "additional_smells": smells
    })

# Save the results to a new file
results_df = pd.DataFrame(results)
results_df.to_csv("report/code_smell_result_types.csv", index=False)
print("Code smell detection completed. Results saved to 'code_smell_result_types.csv'.")
