import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from data_normalization import normalize_landmarks


# Load the dataset ** MAKE SURE TO MODIFY FILE PATHS
file_path = './Data/output.csv'
data = pd.read_csv(file_path)
file_path2 = './Data/outputNew.csv'
data2 = pd.read_csv(file_path2)
file_path3 = './Data/outputNewSam.csv'
data3 = pd.read_csv(file_path3)
file_path4 = './outputMeora.csv'
data4 = pd.read_csv(file_path4)
bigData = pd.concat([data3, data2, data4, data], axis=0)
bigData = bigData.reset_index(drop=True)

# data_path = './Data/bigDataWithHeaders.csv'
# data = pd.read_csv(data_path)
# # Separate features and labels
# X = data4.drop(columns=['Label'])
# y = data4['Label']

normalized_landmarks = normalize_landmarks(bigData)
df_normalized = pd.DataFrame(normalized_landmarks, columns=[f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']])
df_normalized['label'] = bigData['Label']

X = df_normalized.iloc[:, :-1].values
y = df_normalized['label'].values

model = RandomForestClassifier(n_estimators=50, random_state=42)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=skf)

print("Classification Report:")
print(classification_report(y, y_pred))