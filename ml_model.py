import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = './output.csv'
data = pd.read_csv(file_path)
file_path2 = './outputNew.csv'
data2 = pd.read_csv(file_path2)
file_path3 = './output3.csv'
data3 = pd.read_csv(file_path3)
bigData = pd.concat([data, data2, data3], axis=0)

# Separate features and labels
X = bigData.drop(columns=['Label'])
y = bigData['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
model = DecisionTreeClassifier(random_state=20)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)