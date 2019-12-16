# using kmeans clustering to create an unsupervised machine learning
# TODO: Make graphs work for not only sepal data, but also petal data

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# load a popular data set for classification
iris_data = datasets.load_iris()

# split up data
X = iris_data.data[:, :2]
y = iris_data.target

# create a new kmeans algo with 3 clusters, and an random initiation
kmeans = KMeans(n_clusters=3, init="random")
# slap our data into the kmeans algo
kmeans.fit(X)

# the coordinates of the center of our clusters, and the labels of the classes kmeans creates
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# colors of the dots
colors = [".b", ".k", ".c"]

# create a windows with 2 graphs side by side and a title
sepal_fig, (panel1, panel2) = plt.subplots(1, 2)
sepal_fig.suptitle('Iris Sepal data', fontsize=30)

# run through the array and plot each point
for i in range(len(X)):
    panel1.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

# put the centroids on the graph
panel1.scatter(centers[:, 0], centers[:, 1], marker="x", c="yellow", s=60, zorder=10)
panel1.set_title("Predicted for Sepal", fontsize=20)

# in the next pane, put the actual groupings
panel2.scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow')
panel2.set_title("Actual for Sepal", fontsize=20)

# show plot
plt.show()
