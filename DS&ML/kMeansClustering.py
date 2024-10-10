# Import the modules and Read the data.
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.cluster.vq import kmeans
from sklearn.cluster import k_means, KMeans
from sklearn.metrics import silhouette_score

car_df = pd.read_csv('https://raw.githubusercontent.com/Arathi-J/DS-ML/refs/heads/main/jkcars.csv')
# Print the first five records
print(car_df.head())

# Get the total number of rows and columns, data types of columns and missing values (if exist) in the dataset.
print(car_df.shape)
print(car_df.dtypes)
print('\n',car_df.isnull().sum())

# Create a new DataFrame consisting of three columns 'Volume', 'Weight', 'CO2'.
new_data = car_df[['Volume','Weight','CO2']]

# Print the first 5 rows of this new DataFrame.
print(new_data.head())

sil_score = []

for k in range(2,11):
   kmeans =  KMeans(n_clusters = k,random_state=1)
   kmeans.fit(new_data)
   cluster_labels = kmeans.predict(new_data)
   s = silhouette_score(new_data,cluster_labels)
   sil_score.append(s)
print('\n')
df_sil_scores = pd.DataFrame({'K': range(2, 11), 'Silhouette Score': sil_score})
print(df_sil_scores)
# final_df = pd.concat([car_df,cluster_labels],axis=1)
# final_df .columns = list(car_df.columns)+['cluster']
# final_df.head()

import matplotlib.pyplot as plt

plt.plot(df_sil_scores['K'], df_sil_scores['Silhouette Score'])
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()
