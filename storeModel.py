import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from data_analysis import normalize_landmarks
import pickle

data_path = './Data/bigDataWithHeaders.csv'
data = pd.read_csv(data_path)

normalized_landmarks = normalize_landmarks(data)
df_normalized = pd.DataFrame(normalized_landmarks, columns=[f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']])
df_normalized['label'] = data['Label']

X_train = df_normalized.iloc[:, :-1].values
y_train = df_normalized['label'].values

model = SVC(kernel='linear', random_state=42)  # You can also try 'rbf' or other kernels

model.fit(X_train, y_train)

model_data = {
    'model': model
}

with open('hand_landmark_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)