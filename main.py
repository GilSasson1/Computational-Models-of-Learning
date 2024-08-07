import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
# Load the California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

num_of_columns = df.shape[1]
num_of_rows = df.shape[0]
print('There are', num_of_columns, 'columns and', num_of_rows, 'rows in the dataset.')
if df.isnull().any().any():
    print("There are missing values in the dataset.")
else:
    print('There are no missing values in the dataset.')
mean_med_inc = df['MedInc'].mean()
print('The average median income is:', mean_med_inc)

mask = df['HouseAge'] > 30
num_of_houses = sum(mask)
print("The number of house that their median age is greater than 30 is:", num_of_houses)
highest_num_of_rooms = max(df['AveRooms'])
print("The maximum number of rooms is:", highest_num_of_rooms)

plt.hist(df['HouseAge'], bins=30)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('House Age Distribution')
plt.show()
plt.scatter(df['Latitude'], df['Longitude'], c=df['Target'])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Longitude vs. Latitude')
plt.show()

corr_mat = df.corr()
print(corr_mat)
df['RoomsPerPopulation'] = df['AveRooms'] / df['Population']
mean_rooms_per_population = df['RoomsPerPopulation'].mean()
print('The average number of rooms per population is:', mean_rooms_per_population)

plt.boxplot(df['MedInc'], showmeans=True)
plt.ylabel('MedInc')
plt.show()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Heatmap')
plt.show()
