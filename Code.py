import pandas as pp
import seaborn as ss
import numpy as nn
import matplotlib.pyplot as mm
import warnings
warnings.filterwarnings('ignore')
main_file = pp.read_csv('6317a324-a16b-4a58-ac08-d2004ff5bb89_Data.csv')
main_file.head(5)
main_file.describe()
main_file.info()
# Rename columns using a dictionary
main_file = main_file.rename(columns={'Series Name':'Series_Name', 'Series Code':'Series_Code', 'Country Name':'Country_Name', 'Country Code':'Country_Code', '1990 [YR1990]':'YR1990', '2000 [YR2000]':'YR2000', '2012 [YR2012]':'YR2012', '2013 [YR2013]':'YR2013', '2014 [YR2014]':'YR2014', '2015 [YR2015]':'YR2015', '2016 [YR2016]':'YR2016', '2017 [YR2017]':'YR2017', '2018 [YR2018]':'YR2018', '2019 [YR2019]':'YR2019', '2020 [YR2020]':'YR2020', '2021 [YR2021]':'YR2021'})
# or using a function
main_file = main_file.rename(columns=lambda x: x.replace(' [YR','').replace(']',''))
main_file.head()
main_file1 = main_file[main_file['Series_Name'].isin(['Forest area (sq. km)'])]
main_file1
main_file1.info()
main_file1[['YR1990', 'YR2000', 'YR2012', 'YR2013', 'YR2014', 'YR2015', 'YR2016', 'YR2017', 'YR2018', 'YR2019', 'YR2020', 'YR2021']] = main_file[['YR1990', 'YR2000', 'YR2012', 'YR2013', 'YR2014', 'YR2015', 'YR2016', 'YR2017', 'YR2018', 'YR2019', 'YR2020', 'YR2021']].apply(pp.to_numeric, errors='coerce')
data = main_file1[['YR1990',	'YR2000',	'YR2012',	'YR2013',	'YR2014',	'YR2015',	'YR2016',	'YR2017',	'YR2018',	'YR2019',	'YR2020',	'YR2021']]
data.head()
data.isnull()
data = data.fillna(0)
null_values = data.isna().sum()
print(null_values)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
# Select the 5 columns to use for clustering
data1 = data[['YR1990',	'YR2000',	'YR2012',	'YR2013',	'YR2014',	'YR2015',	'YR2016',	'YR2017',	'YR2018',	'YR2019',	'YR2020',	'YR2021']].values
data.head()
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_pca)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Create a 3D scatter plot
fig = mm.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
mm.show()
# Create a scatter plot
mm.scatter(data1[:, 0], data1[:, 1])
mm.scatter(data1[:, 2], data1[:, 3])
mm.scatter(data1[:, 4], data1[:, 5])
mm.scatter(data1[:, 6], data1[:, 7])
mm.scatter(data1[:, 8], data1[:, 9])
mm.scatter(data1[:, 10], data1[:, 11])
mm.xlabel('Year')
mm.ylabel('Value')
mm.show()
for i in range(data1.shape[1]):
    mm.scatter(range(len(data1)), data1[:, i])
    mm.xlabel('Index')
    mm.ylabel(f'YR{2012+i}')
    mm.show()
data
df = pp.DataFrame(data)

# create a bar plot of the dataset
ss.barplot(data=df)

# display the plot
mm.show()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# assign the columns to the appropriate variables
X = data[['YR1990', 'YR2000', 'YR2012', 'YR2013', 'YR2014', 'YR2015', 'YR2016', 'YR2017', 'YR2018', 'YR2019', 'YR2020', 'YR2021']]
y = data['YR2019']
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# create the linear regression model
reg = LinearRegression()
# fit the model to the training data
reg.fit(X_train, y_train)
# predict the labels for the test data
y_pred = reg.predict(X_test)
# evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: {:.2f}".format(mse))
print("R-Squared Score: {:.2f}".format(r2))
from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create the Lasso regression model
lasso = Lasso()

# fit the model to the training data
lasso.fit(X_train, y_train)

# predict the labels for the test data
y_pred = lasso.predict(X_test)

# evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: {:.2f}".format(mse))
print("R-Squared Score: {:.2f}".format(r2))

