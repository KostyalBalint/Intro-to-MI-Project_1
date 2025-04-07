import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"

# Load the SAheart dataset
df = pd.read_csv(url, index_col='row.names')

# Convert binary text data to numbered categories
df['famhist'] = pd.Categorical(df['famhist']).codes

# Extract the name of the attributes (columns)
attributeNames = list(map(lambda x: x.capitalize(), df.columns.tolist()))

# Convert the dataframe to numpy
y = df['chd'].to_numpy() # classification problem of CHD or no CHD
X = df.drop(columns=['chd']).to_numpy() # rest of the attributes, remove 'CHD' column

# Compute size of X
N, M = X.shape # N = observations, M = attributes (except 'chd')
N_numbers = np.arange(1, N+1)

# Normalize the datapoints to have a mean of 0 
mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X - mu) / sigma

ones_count = np.count_nonzero(y == 1)
zeros_count = np.count_nonzero(y == 0)
perCHD = round(100*ones_count/len(y),1)
print('Males with CHD: {0} ({2}%), Males without CHD: {1}'.format(ones_count, zeros_count,perCHD))

# Two-layer cross validation

# K-fold for cross validation
k = 0 # index of CV
Kfold = 10 # maximum CV
CV = model_selection.KFold(Kfold) # model for CV

# for evaluation of logistic regression
lambdas = np.linspace(10,120,25)
chosen_lambdas = np.zeros(Kfold)

# for evaluation of KNN
ks = np.arange(1,30+1)
chosen_ks = np.zeros(Kfold)

# arrays to save the predicted values
y_true = []
y_est_KNN_All = []
y_est_LOGREG_All = []
y_est_base_All = []

# miss classification
Missclass_KNN = np.zeros(Kfold)
Missclass_LOGREG = np.zeros(Kfold)
Missclass_base = np.zeros(Kfold)

# Outer fold
for train_index_out, test_index_out in CV.split(X,y):
    print("# Outer fold: {0}/{1}".format(k + 1, Kfold))
    
    # Extract the training and test set for the outer-fold
    X_train_out = X[train_index_out, :]
    y_train_out = y[train_index_out]
    X_test_out = X[test_index_out, :]
    y_test_out = y[test_index_out]

    # Inner fold for chosing the right k and lambda
    i = 0

    # to save the errors for the different lambda- and ks-values
    test_error_lambdas = np.zeros(len(lambdas))
    test_error_ks = np.zeros(len(ks))

    for train_index, test_index in CV.split(X_train_out,y_train_out):
        
        # Extract training and test set for current inner fold
        X_train = X_train_out[train_index, :]
        y_train = y_train_out[train_index]
        X_test = X_train_out[test_index, :]
        y_test = y_train_out[test_index]

        # find optimal lambda value for the training set
        for idx, lambdas_values in enumerate(lambdas):
            # train model
            model_logreg = LogisticRegression(penalty="l2", C=1/lambdas_values, max_iter=500)
            model_logreg.fit(X_train, y_train)

            # predict test and training data
            y_est_test_logreg = model_logreg.predict(X_test)

            # find error of predictions
            test_error_lambdas[idx]= np.sum(y_est_test_logreg != y_test)/len(y_test)

        
        # find optimal k-value for KNN
        for idx, ks_values in enumerate(ks): 
            metric = "cosine"
            metric_params = {}  
            #metric='mahalanobis'
            #metric_params={'V': np.cov(X_train, rowvar=False)}
            knclassifier = KNeighborsClassifier(n_neighbors=int(ks_values), p = 1, metric = metric, metric_params = metric_params) # choose the model
            knclassifier.fit(X_train, y_train) # fits the model to X_train, y_train

            # predict test and training data
            y_est_test_KNN = knclassifier.predict(X_test) # estimate y from X_test

            # find error
            test_error_ks[idx] = np.sum(y_est_test_KNN != y_test)/len(y_test)

        i += 1
        
    min_value_lambdas = np.min(test_error_lambdas) # find lowest test error
    min_index_lambdas = np.where(test_error_lambdas == min_value_lambdas)[0] # the indices of min_error
    # lambda value for logistic regression
    chosen_lambdas[k] = lambdas[min_index_lambdas[0]]

    min_value_ks = np.min(test_error_ks) # find lowest test error
    min_index_ks = np.where(test_error_ks == min_value_ks)[0] # the indices of min_error
    # k-value for nearest neighbor, use the largest number. 
    chosen_ks[k] = ks[min_index_ks[0]]

    ### Train the outer-set with the chosen k and lambda ###
    ## The true y-values
    y_true.append(y_test_out)

    ## K-nearest neigbors 
    knclassifier = KNeighborsClassifier(n_neighbors=int(chosen_ks[k]), p = 2, metric = metric, metric_params = metric_params)
    knclassifier.fit(X_train_out, y_train_out)
    # test
    y_est_KNN = knclassifier.predict(X_test_out)
    y_est_KNN_All.append(y_est_KNN)
    Missclass_KNN[k] = 100 * (np.sum(y_est_KNN != y_test_out)/len(y_test_out))
    
    ## Logistic Regression
    logisticclassifier = LogisticRegression(C = 1/chosen_lambdas[k], max_iter = 500)
    logisticclassifier.fit(X_train_out, y_train_out) 
    # test
    y_est_LOGREG = logisticclassifier.predict(X_test_out)
    y_est_LOGREG_All.append(y_est_LOGREG)
    Missclass_LOGREG[k] = 100 * (np.sum(y_est_LOGREG != y_test_out)/len(y_test_out))
    
    ## Baseline
    a = 0
    b = 0
    while True:
        for c in range(len(y_train)):
            if y_train_out[c] == 0:  # No CHD class
                a += 1
            elif y_train_out[c] == 1:  # CHD class
                b += 1
        
        # test
        if a > b: # If the class of no CHD (a) is largest
            y_est_base = np.full(len(y_test_out), 0) # baseline predicts all y's to not have CHD
            y_est_base_All.append(y_est_base)
            break
        else: # If the class of CHD (b) is largest
            y_est_base = np.full(len(y_test_out), 1) # baseline predicts all y's to have CHD
            y_est_base_All.append(y_est_base)
            break
    Missclass_base[k] = 100 * (np.sum(y_est_base != y_test_out)/len(y_test_out)) # error of each innerfold as columns
    
    k += 1



Class_df_columns = pd.MultiIndex.from_tuples([
    ("Outer fold", "i"), 
    ("KNN", "k"),
    ("KNN", "Error_test (%)"),
    ("Logistic regression", "lambda"),
    ("Logistic regression", "Error_test(%)"),
    ("Baseline", "Error_test(%)")])

Outer_fold = np.round(np.arange(1, Kfold + 1),0)
 
Class_df_data = np.array([Outer_fold, 
                          np.round(chosen_ks), # k-value
                          np.round(Missclass_KNN,1), # Error_test of KNN
                          np.round(chosen_lambdas), # Lambda
                          np.round(Missclass_LOGREG,1), # Error_test of logistic function
                          np.round(Missclass_base,1)]).T

Class_df = pd.DataFrame(Class_df_data, columns=Class_df_columns)
print(Class_df)



