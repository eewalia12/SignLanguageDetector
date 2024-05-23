import pandas as pd

# MAKE SURE TO MODIFY FILE PATHS

file_path = './Data/output.csv'
data = pd.read_csv(file_path)
file_path2 = './Data/outputNew.csv'
data2 = pd.read_csv(file_path2)
file_path3 = './Data/outputNewSam.csv'
data3 = pd.read_csv(file_path3)
bigData = pd.concat([data, data2, data3], axis=0)

columns = ["Label"] + [f"landmark_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]

bigData.columns = columns
output_file_path = './Data/bigDataWithHeaders.csv'
bigData.to_csv(output_file_path, index=False)
