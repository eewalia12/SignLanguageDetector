import numpy as np
import pandas as pd

df = pd.read_csv('./Data/bigDataWithHeaders.csv')

def normalize_landmarks(df):
    landmarks = df.iloc[:, 1:].values.reshape(-1, 21, 3)
    wrist_coords = landmarks[:, 0, :].reshape(-1, 1, 3)
    normalized_landmarks = landmarks - wrist_coords

    max_dist = np.linalg.norm(normalized_landmarks, axis=2).max(axis=1).reshape(-1, 1, 1)  # Shape: (n_samples, 1, 1)
    normalized_landmarks /= max_dist  # Scale coordinates
    return normalized_landmarks.reshape(-1, 63)  # Flatten the array back to (n_samples, 63)

def normalize_landmarks_array(landmarks):
    landmarks = landmarks.reshape(-1, 21, 3)  # Ensure the shape is (n_samples, 21, 3)
    wrist_coords = landmarks[:, 0, :].reshape(-1, 1, 3)  # Extract wrist coordinates
    normalized_landmarks = landmarks - wrist_coords  # Normalize landmarks by subtracting wrist coordinates

    max_dist = np.linalg.norm(normalized_landmarks, axis=2).max(axis=1).reshape(-1, 1, 1)  # Compute max distance
    normalized_landmarks /= max_dist  # Scale coordinates
    return normalized_landmarks.reshape(-1, 63)  # Flatten back to (n_samples, 63)

# normalized_landmarks = normalize_landmarks(df)
# df_normalized = pd.DataFrame(normalized_landmarks, columns=[f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']])
# df_normalized['label'] = df['Label']

