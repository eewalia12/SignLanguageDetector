import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from data_normalization import normalize_landmarks
import pickle

data_path = './Data/updated_normalized_data.csv'
data = pd.read_csv(data_path)

X = data.drop(columns=['Label'])
y = data['Label']

model = RandomForestClassifier(n_estimators=50, random_state=42)

model.fit(X, y)

model_data = {
    'model': model
}

with open('hand_landmark_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)