import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Embedded configuration for code smell IDs
config = {
    "code_smell_ids": [88, 30, 40, 22, 45]  # Adjust these IDs as needed
}

# Load the list of code smell IDs from the embedded configuration
code_smell_ids = config["code_smell_ids"]

# Load datasets with error handling
try:
    code_blocks = pd.read_csv("data/code_blocks.csv")
    kernels_meta = pd.read_csv("data/kernels_meta.csv")
    competitions_meta = pd.read_csv("data/competitions_meta.csv")
    data_with_preds = pd.read_csv("data/data_with_preds.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure all required files are in the 'data' directory.")
    exit()

# Merge datasets to create a comprehensive training dataset
code_data = pd.merge(code_blocks, kernels_meta, on="kernel_id", how="inner")
code_data = pd.merge(code_data, competitions_meta, on="comp_name", how="inner")
code_data = pd.merge(code_data, data_with_preds[['code_blocks_index', 'predicted_graph_vertex_id']], 
                     left_on="code_blocks_index", right_on="code_blocks_index", how="inner")

# Map `predicted_graph_vertex_id` to binary labels for code smells
code_data['label'] = code_data['predicted_graph_vertex_id'].apply(lambda x: 1 if x in code_smell_ids else 0)

# Preprocess code text (assuming 'code_block' has the code snippets)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(code_data['code_block'])
y = code_data['label']

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForestClassifier using RandomizedSearchCV
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Instantiate the Random Forest Classifier and RandomizedSearchCV
model = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, 
                                   n_iter=10, scoring='accuracy', cv=3, random_state=42, n_jobs=-1)

# Train the model with hyperparameter tuning
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Evaluate the model and save the report
y_pred = best_model.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
accuracy_rep = f"Accuracy: {accuracy_score(y_test, y_pred)}"

# Create the model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save classification report and accuracy to a text file
with open("model/training_report.txt", "w") as report_file:
    report_file.write("Hyperparameter Tuning Results:\n")
    report_file.write(f"Best Parameters: {random_search.best_params_}\n\n")
    report_file.write("Classification Report:\n")
    report_file.write(classification_rep + "\n")
    report_file.write(accuracy_rep + "\n")

# Print evaluation summary to console
print("Best Parameters:", random_search.best_params_)
print(classification_rep)
print(accuracy_rep)

# Save the trained model and vectorizer for future use
joblib.dump(best_model, "model/code_smell_detector_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer saved successfully.")
