import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


# Load the dataset
file_path = './Data/output.csv'
data = pd.read_csv(file_path)
file_path2 = './Data/outputNew.csv'
data2 = pd.read_csv(file_path2)
file_path3 = './Data/outputNewSam.csv'
data3 = pd.read_csv(file_path3)
bigData = pd.concat([data, data2, data3], axis=0)

# Separate features and labels
X = bigData.drop(columns=['Label'])
y = bigData['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
# model = DecisionTreeClassifier(random_state=20)
# model = LogisticRegression(random_state=20, max_iter=100)
model = Perceptron(max_iter=1000, tol=1e-3)
# model = KNeighborsClassifier(n_neighbors=5)

cv_scores = cross_val_score(model, X_train, y_train, cv=10)

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(report)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())
print("Standard deviation of cross-validation scores:", cv_scores.std())

# plt.figure(figsize=(20,10), dpi=300)
# plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
# plt.show()