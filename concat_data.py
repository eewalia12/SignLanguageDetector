import pandas as pd
from data_normalization import normalize_landmarks

# MAKE SURE TO MODIFY FILE PATHS

data_path = "./outputAdditional.csv"
data = pd.read_csv(data_path)
data_path2 = "./updatedWithHeaders.csv"
data2 = pd.read_csv(data_path2)

columns = ["Label"] + [f"landmark_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
data.columns = columns

bigData = pd.concat([data, data2], axis=0)



bigData.columns = columns
output_file_path = './finalData.csv'
bigData.to_csv(output_file_path, index=False)
