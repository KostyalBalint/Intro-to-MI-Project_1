from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"

# Load the SAheart dataset
df = pd.read_csv(url, index_col='row.names')

# Convert binary text data to numbered categories
df['famhist'] = pd.Categorical(df['famhist']).codes

# Extract the name of the attributes (columns)
attributeNames = list(map(lambda x: x.capitalize(), df.columns.tolist()))

# Convert the dataframe to numpy
y = df['chd'].to_numpy() # classification problem of CHD or no CHD
X = df.drop(columns=['chd']).to_numpy() # rest of the attributes

# Compute size of X
N, M = X.shape # N = observations, M = attributes (except 'chd')
N_numbers = np.arange(1, N+1)

# Normalize the datapoints to have a mean of 0 
mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X - mu) / sigma

# Find KNN with k = 2
C = 2 # number of classnames: 'no CHD' and 'CHD'

# Plot the training data points (color-coded) and test data points.
plt.figure(1)
styles = [".y", ".r"]
for c in range(C):
    class_mask = y_train == c
    plt.plot(y_train[class_mask], N1_numbers_train[class_mask], styles[c])
    
# K-nearest neighbors
K = 2

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist = 1
# metric = "minkowski"
# metric_params = {}  # no parameters needed for minkowski

# You can set the metric argument to 'cosine' to determine the cosine distance
# metric = 'cosine'
# metric_params = {} # no parameters needed for cosine

# To use a mahalonobis distance, we need to input the covariance matrix, too:
metric='mahalanobis'
metric_params={'V': np.cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(
    n_neighbors=K, p=dist, metric=metric, metric_params=metric_params
)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)


# Plot the classfication results
styles = ["oy", "or"] 
for c in range(C):
    class_mask = y_est == c
    plt.plot(y_est[class_mask], N1_numbers_test[class_mask], styles[c], markersize=10)
    plt.plot(y_est[class_mask], N1_numbers_test[class_mask], "kx", markersize=8)
plt.title("Synthetic data classification - KNN")
print(X_test[class_mask, 0])

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est)
accuracy = np.round(100 * cm.diagonal().sum() / cm.sum(), 2)
error_rate = np.round(100 - accuracy, 2)
plt.figure(2)
plt.imshow(cm, cmap="binary", interpolation="None")
plt.colorbar()
plt.xticks(range(C))
plt.yticks(range(C))
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title(
    "Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)".format(accuracy, error_rate)
)

plt.show()
