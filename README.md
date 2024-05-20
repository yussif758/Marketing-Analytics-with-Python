# Marketing-Analytics-with-Python
Customer Segmentation and Clustering

#DATA EXPLORATION 
#import pandas because thats our data manipulation library
```
import pandas as pd 
import numpy as np
```
#import warnigs so that the warnings wont always pop-up 
```
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('Segmentation-Mall_Customers.csv')
data.head()
#print(data)

Columns = data.columns 
#print(Columns)
```
#Use describe to give you a summary statistics of the data
```
Description = data.describe()
#print(Description)
```
#RULE BASED SEGMENTATION
#Value_counts() gives you number of genders and if you want it in percentages uses value_counts(normalize =True)
```
Counts = data['Gender'].value_counts()
Counts1 = data['Gender'].value_counts(normalize =True)
#print(Counts)
MaxAge = data['Age'].max()
MinAge = data['Age'].min()
#print(MaxAge)
#print(MinAge)
```
#For percentile use numpy which I imported as 'np'
```
p1, p2 = np.percentile(data['Age'],[33.3,66.6])
#print(p1)
```
#Creating a Bin 
#Bin means taking a continuous data and putting it in a categorical feature like low,medium and high.
#Any age above the p2, High, age above p1, medium else Low
```
data['Age Bin'] = np.where(data['Age']>p2,'High',np.where(data['Age']>p1,'Medium','Low'))
Counts2 = data['Age Bin'].value_counts()
#print(Counts2)
```
#Create a Function to bin our data.
#by using function we make our work simple because we can apply it to other columns(repetition)
```
def Binner(var,data):
    p1, p2 = np.percentile(data[var],[33.3,66.6])
    data[var+ 'Bin'] = np.where(data[var]>p2,'High',np.where(data[var]>p2,'Medium','Low'))
    
Binner('Annual Income (k$)',data)
Binner('Spending Score (1-100)',data)

#data = data.head()
#print(data)
```
#Analyzing data categories 
```
Group = data.groupby(['Gender','Age Bin','Annual Income (k$)Bin','Spending Score (1-100)Bin']).mean()[['Age','Annual Income (k$)','Spending Score (1-100)']]
#print(Group)
```
#We can also group by the count of customer ID.
```
Group1 = data.groupby(['Gender','Age Bin','Annual Income (k$)Bin','Spending Score (1-100)Bin']).count()[['CustomerID']]
#print(Group1)
```
#We can also sort by customerID
```
Group2 = data.groupby(['Gender','Age Bin','Annual Income (k$)Bin','Spending Score (1-100)Bin']).count()[['CustomerID']].sort_values(by='CustomerID')
#print(Group2)
#OR use ascending=False to arrange them in descending order
Group2 = data.groupby(['Gender','Age Bin','Annual Income (k$)Bin','Spending Score (1-100)Bin']).count()[['CustomerID']].sort_values(by='CustomerID', ascending=False)
```
#UNSUPERVISED SEGMENTATION (K-Means Algorithm)
```
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
num_clusters = 3
X = data[['Age','Annual Income (k$)','Spending Score (1-100)']]
#Initialize the kMeans algorithm
#random_state is not necessary but if you want to reproduce the results you have to give a random state
Kmeans = KMeans(n_clusters = num_clusters, random_state = 0)
```
#fit the algorithm to the data: When you fit the data thats the learning phase of the data 
```
Kmeans.fit(X)
```
#Get the cluster labels for each data points using 'predict' to predict what the clusters are
```
labels = Kmeans.predict(X)
```
#Get the Centroids for each cluster 
```
centroids = Kmeans.cluster_centers_
data['Cluster Name'] = labels
#print(data.sample(10))
```
#visualizing clusters: 
```
from mpl_toolkits.mplot3d import Axes3D
```
#plot the cluters in 3D
```
#fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X['Age'],X['Annual Income (k$)'], X['Spending Score (1-100)'], c = Kmeans.labels_)
#ax.set_xlabel('Age')
#ax.set_ylabel('Annual Income (k$)')
#ax.set_zlabel('Spending Score (1-100)')
#print(plt.show())
```
#Plot the clusters in 2D, can choose any 2 variables. 
```
#fig = plt.figure(figsize=(5,5))
#plt.scatter(data['Spending Score (1-100)'],data['Annual Income (k$)'], c = Kmeans.labels_)
#plt.xlabel('Spending score')
#plt.ylabel('Income')
#print(plt.show())
```
#Applying Principal Component Analysis(PCA) to reduce dimensionality to 2 components
#PCA is a dimensionality reduction technique that can be used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set. Can used to compress 3 dimensions into 2. 
```
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(X)
```
#Applying K means clustering
```
Kmeans = KMeans(n_clusters=3, random_state=0)
Kmeans.fit(x_pca)
```
#plot the clusters in 2d dimention 
0, 1 are the first and second feature in the x_pca 
```
plt.scatter(x_pca[:,0],x_pca[:,1], c=Kmeans.labels_)
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
print(plt.show())
```
#Note that PCA is a sneak technique but it is hard to interpret your results beause we dont know the exact features in the components.

#Clustering Insights and personas 
Deriving insights from unsupervised clusters feature
Once the clusters have been identified, the next step is to analyze the characteristics of each cluster and develop rules and criteria based on their common traits. Deriving the rules involves interpreting the results of a clustering analysis to extract meaningful and actionable insights.

1.compute the mean of all the attributes grouped by the clusters 
2.compute the medium of all the attributes grouped by the clusters
3.try to identify the differences in the attribute mean and median
4. Devices strategies based on clusters

```
new_info = data.groupby('Cluster Name')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(new_info)

new_info1 = data.groupby('Cluster Name')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].median()
print(new_info1)

Crosstab = pd.crosstab(data['Cluster Name'], data['Gender'], normalize='index')
print(Crosstab)
```
#What are the insights?
#Cluster 0 
#1. From data,cluster 0 is made up of people who are middle-aged, high income earners with low spending habit. 
#2. offer exclusive savings or investment opportunities
#3.Introduce progams that will motivate them to spend
#4. Also host events appealing to mature audience 
#5. Marketing efforts should be balanced for both gender. 
#Cluster 1
#1. Individuals in this group are younger, high income earners with high spending habits.
#2. Pormote the latest product they might like.
#3. Host unique trend aligned events 
#4. offer subscribtions for trendy items or experience. 
#5.  Slightly pay more attention to female-centric campaigns 
#Cluster 2
#1. Cluster 2 are made up of middled-age, average income earners with average spending. 
#2. provide value for money or bundled offers. 
#3. Tailor marketing efforts to individual preference. 
#4. Focus on products or campaigns appealing to women. 

#Elbow and Silhouette method for clustering:
#Allows you to estimate what the number of clusters should be.
#Elbow method works by plotting the sum of squared distnaces between each data point and its assigned centroid(also know as the within-clsuter sum of squares or WCSS) against the number of clusters. 

#Run Kmeans clustering for k=1 to 10 clusters
```
wcss = []
for k in range(1,11):
    Kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state = 0)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
```
#Plot elbow curve 
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
print(plt.show())
