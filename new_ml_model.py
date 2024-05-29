import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from data_normalization import normalize_landmarks
import pickle

# Load the dataset ** MAKE SURE TO MODIFY FILE PATHS
file_path = './updated_normalized_data.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['Label'])
y = data['Label']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def train_and_evaluate_model(model, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    return acc, report

# Define the models to evaluate
models = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    LogisticRegression(random_state=42, max_iter=1000),
    SVC(random_state=42),
    KNeighborsClassifier()
]

# Train and evaluate each model
scores = []
reports = []
for model in models:
    score, report = train_and_evaluate_model(model, X_scaled, y)
    scores.append(score)
    reports.append(report)
    print(f"Model: {model.__class__.__name__}, Accuracy: {score:.3f}")
    print(f"Classification Report:\n{report}\n")

# Find the best model
best_model_index = scores.index(max(scores))
best_model = models[best_model_index]

best_model.fit(X, y)

model_data = {
    'model': best_model
}

with open('hand_landmark_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print(f"The best model is: {best_model.__class__.__name__}")