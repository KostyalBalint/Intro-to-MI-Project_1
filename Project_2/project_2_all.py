#%%
# exercise 5.1.1
from dtuimldmtools import (
    draw_neural_net,
    train_neural_net,
)
import pandas as pd
from scipy.stats import zscore

url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"

# Load the SAheart dataset
df = pd.read_csv(url, index_col='row.names')


# Convert binary text data to numbered categories
df['famhist'] = pd.Categorical(df['famhist']).codes

y = df['typea'].to_numpy().reshape(-1, 1)

# Drop the typea column, as that is what we try to predict
df = df.drop("typea", axis=1)

X = df.to_numpy()

# Attribute names
attributeNames = list(map(lambda x: x.capitalize(), df.columns.tolist()))

N, M = X.shape
classNames = [0, 1]
C = len(classNames)

X = zscore(X, ddof=1)  # Mean = 0, Std = 1
y = zscore(y, ddof=1)  # Mean = 0, Std = 1

# Step 2: Normalization (Min-Max Scaling to [0,1])
# X = (X_standardized - X_standardized.min()) / (X_standardized.max() - X_standardized.min())
# y = (y_standardized - y_standardized.min()) / (y_standardized.max() - y_standardized.min())
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
import torch
from ann_validate import ann_validate
from dtuimldmtools import rlr_validate

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000
CV = model_selection.KFold(K, shuffle=True)
# CV = model_selection.KFold(K, shuffle=False)

n_hidden_units_range = range(1, 20)

# Initialize variables
# T = len(lambdas)
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))


#for train_index, test_index in CV.split(X, y):
    # Extract training and test set for current CV fold, convert to tensors
    #X_train = torch.Tensor(X[train_index, :])
    #y_train = torch.Tensor(y[train_index])
    #X_test = torch.Tensor(X[test_index, :])
    #y_test = torch.Tensor(y[test_index])

loss_fn = torch.nn.MSELoss()
models = np.empty((len(n_hidden_units_range),), dtype=object)

CV = model_selection.KFold(K, shuffle=True)

# Generate out models with different number of hidden units
for i, n_hidden_units in enumerate(n_hidden_units_range):
    model = torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # Input layer
        torch.nn.Tanh(),  # Hidden activation (ReLU is also good)
        torch.nn.Linear(n_hidden_units, 1)   # Output layer (NO SIGMOID for regression)
    )
    models[i] = model

(
    opt_val_err,
    opt_model_index,
    opt_model,
    opt_network,
    train_err_vs_lambda,
    test_err_vs_lambda
    #This does a folding cross-validation
) = ann_validate(X, y, models, loss_fn, n_replicates, max_iter, K, CV)

Error_train = np.square(y-opt_network(torch.Tensor(X)).detach().numpy()).sum()/y.shape[0]
Error_test = np.square(y-opt_network(torch.Tensor(X)).detach().numpy()).sum()/y.shape[0]

# Display the results for the last cross-validation fold
plt.figure(1, figsize=(14, 8))
plt.title("Optimal h (n.o. hidden units): {0}".format(n_hidden_units_range[opt_model_index]))
plt.plot(
    n_hidden_units_range, train_err_vs_lambda.T, "b.-", n_hidden_units_range, test_err_vs_lambda.T, "r.-"
)
plt.xlabel("Regularization factor")
plt.ylabel("Squared error (crossvalidation)")
plt.legend(["Train error", "Validation error"])
plt.grid()


plt.savefig('regression_lambda.png')
plt.show()
# Display results

print("ANN results")
print("- Training error: {0}".format(train_err_vs_lambda[opt_model_index]))
print("- Test error:     {0}".format(test_err_vs_lambda[opt_model_index]))

#%%
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = [
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:red",
    "tab:blue",
]

model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
    torch.nn.Tanh(),  # 1st transfer function,
    torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
    # no final tranfer function, i.e. "linear output"
)
loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

print("Training model of type:\n\n{}\n".format(str(model())))
errors = []  # make a list for storing generalizaition error in each loop
for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    print("\n\tBest loss: {}\n".format(final_loss))

    # Determine estimated class labels for test set
    y_test_est = net(X_test)

    # Determine errors and errors
    se = (y_test_est.float() - y_test.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
    errors.append(mse)  # store error rate for current CV fold

    # Display the learning curve for the best net in the current fold
    (h,) = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label("CV fold {0}".format(k + 1))
    summaries_axes[0].set_xlabel("Iterations")
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel("Loss")
    summaries_axes[0].set_title("Learning curves")

# Display the MSE across folds
summaries_axes[1].bar(
    np.arange(1, K + 1), np.squeeze(np.asarray(errors)), color=color_list
)
summaries_axes[1].set_xlabel("Fold")
summaries_axes[1].set_xticks(np.arange(1, K + 1))
summaries_axes[1].set_ylabel("MSE")
summaries_axes[1].set_title("Test mean-squared-error")

print("Diagram of best neural net in last fold:")
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print(
    "\nEstimated generalization error, RMSE: {0}".format(
        round(np.sqrt(np.mean(errors)), 4)
    )
)
#%%
plt.figure(figsize=(10, 10))
y_est = y_test_est.data.numpy()
y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true]) - 1, np.max([y_est, y_true]) + 1]
plt.plot(axis_range, axis_range, "k--")
plt.plot(y_true, y_est, "ob", alpha=0.25)
plt.legend(["Perfect estimation", "Model estimations"])
plt.title("Alcohol content: estimated versus true value (for last CV-fold)")
plt.ylim(axis_range)
plt.xlim(axis_range)
plt.xlabel("True value")
plt.ylabel("Estimated value")
plt.grid()

plt.show()
#%%
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Define K-Fold Cross-Validation
K = 5  # Number of folds
kf = KFold(n_splits=K, shuffle=True, random_state=42)

mse_scores = []

# Perform Cross-Validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Compute the mean of y_train as the baseline prediction
    y_baseline = np.mean(y_train)

    # Predict the same mean value for all test samples
    y_pred_baseline = np.full_like(y_test, y_baseline)

    # Compute and store the MSE
    mse = mean_squared_error(y_test, y_pred_baseline)
    mse_scores.append(mse)



# Compute average MSE across all folds
mean_mse = np.mean(mse_scores)

print(f"Baseline Model - Cross-Validation MSE: {mean_mse:.4f}")

#%%

import numpy as np
import torch
from sklearn import model_selection
from tqdm import tqdm

def ann_validate(X, y, models, loss_fn, n_replicates, max_iter, cvf, CV):

    print(models)

    """Validate regularized linear regression model using 'cvf'-fold cross validation.
    Find the optimal lambda (minimizing validation error) from 'lambdas' list.
    The loss function computed as mean squared error on validation set (MSE).
    Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
    average weight values for all lambdas, MSE train&validation errors for all lambdas.
    The cross validation splits are standardized based on the mean and standard
    deviation of the training set when estimating the regularization strength.

    Parameters:
    X       training data set
    y       vector of values
    lambdas vector of lambda values to be validated
    cvf     number of crossvalidation folds

    Returns:
    opt_val_err         validation error for optimum lambda
    opt_lambda          value of optimal lambda
    mean_w_vs_lambda    weights as function of lambda (matrix)
    train_err_vs_lambda train error as function of lambda (vector)
    test_err_vs_lambda  test error as function of lambda (vector)
    """
    M = X.shape[1]

    trained_networks = np.empty((cvf, len(models)), dtype=object)
    train_error = np.empty((cvf, len(models)))
    test_error = np.empty((cvf, len(models)))

    f = 0 # fold counter
    y = y.squeeze()
    for train_index, test_index in tqdm(CV.split(X, y)):
        X_train = torch.Tensor(X[train_index])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index])
        y_test = torch.Tensor(y[test_index])


        # Standardize the training and test set based on training set moments using PyTorch
        mu = torch.mean(X_train[:, 1:], dim=0)
        sigma = torch.std(X_train[:, 1:], dim=0)

        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        for model_index in range(0, len(models)):

            #print("Training model of type:\n{}\n".format(str(models[model_index]())))

             # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                lambda: models[model_index],
                loss_fn,
                X=X_train,
                y=y_train,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )

            #print("\n\tBest loss: {}\n".format(final_loss))

            # Determine estimated class labels for test set
            y_test_est = net(X_test)

            ## Determine the test error
            #se = (y_test_est.float() - y_test.float()) ** 2  # squared error
            #mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean

            train_error[f, model_index] = final_loss
            test_error[f, model_index] = loss_fn(y_test.squeeze(), y_test_est.squeeze()).detach().numpy()
            trained_networks[f, model_index] = net

        f = f + 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_model_index = np.argmin(np.mean(test_error, axis=0))
    opt_model = models[opt_model_index]
    opt_network = trained_networks[np.unravel_index(np.argmin(test_error), test_error.shape)]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)


    return (
        opt_val_err,
        opt_model_index,
        opt_model,
        opt_network,
        train_err_vs_lambda,
        test_err_vs_lambda
    )


import numpy as np
import torch


""" Copy of the function from the dtuimldmtools package, but remove logging"""
def train_neural_net(
    model, loss_fn, X, y, n_replicates=3, max_iter=10000, tolerance=1e-6
):
    """
    Train a neural network with PyTorch based on a training set consisting of
    observations X and class y. The model and loss_fn inputs define the
    architecture to train and the cost-function update the weights based on,
    respectively.

    Args:
        model: A function handle to make a torch.nn.Sequential.
        loss_fn: A torch.nn-loss, e.g. torch.nn.BCELoss() for binary
                 binary classification, torch.nn.CrossEntropyLoss() for
                 multiclass classification, or torch.nn.MSELoss() for
                 regression.
        X: The input observations as a PyTorch tensor.
        y: The target classes as a PyTorch tensor.
        n_replicates: An integer specifying number of replicates to train,
                      the neural network with the lowest loss is returned.
        max_iter: An integer specifying the maximum number of iterations
                  to do (default 10000).
        tolerance: A float describing the tolerance/convergence criterion
                   for minimum relative change in loss (default 1e-6)


    Returns:
        A list of three elements:
            best_net: A trained torch.nn.Sequential that had the lowest
                      loss of the trained replicates.
            final_loss: A float specifying the loss of best performing net.
            learning_curve: A list containing the learning curve of the best net.

    Usage:
        Assuming loaded dataset (X,y) has been split into a training and
        test set called (X_train, y_train) and (X_test, y_test), and
        that the dataset has been cast into PyTorch tensors using e.g.:
            X_train = torch.tensor(X_train, dtype=torch.float)
        Here illustrating a binary classification example based on e.g.
        M=2 features with H=2 hidden units:

        >>> # Define the overall architechture to use
        >>> model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, H),  # M features to H hiden units
                    torch.nn.Tanh(),        # 1st transfer function
                    torch.nn.Linear(H, 1),  # H hidden units to 1 output neuron
                    torch.nn.Sigmoid()      # final tranfer function
                    )
        >>> loss_fn = torch.nn.BCELoss() # define loss to use
        >>> net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3)
        >>> y_test_est = net(X_test) # predictions of network on test set
        >>> # To optain "hard" class predictions, threshold the y_test_est
        >>> See exercise ex8_2_2.py for indepth example.

        For multi-class with C classes, we need to change this model to e.g.:
        >>> model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, H), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            torch.nn.Linear(H, C), # H hidden units to C classes
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )
        >>> loss_fn = torch.nn.CrossEntropyLoss()

        And the final class prediction is based on the argmax of the output
        nodes:
        >>> y_class = torch.max(y_test_est, dim=1)[1]
    """

    # Specify maximum number of iterations for training
    logging_frequency = 1000  # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        #print("\n\tReplicate: {}/{}".format(r + 1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights)
        net = model()

        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)

        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        # optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)

        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())

        # Train the network while displaying and storing the loss
        #print("\t\t{}\t{}\t\t\t{}".format("Iter", "Loss", "Rel. loss"))
        learning_curve = []  # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X)  # forward pass, predict labels on training set
            loss = loss_fn(y_est, y)  # determine loss
            loss_value = loss.data.numpy()  # get numpy array instead of tensor
            learning_curve.append(loss_value)  # record loss for later display

            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value - old_loss) / old_loss
            if p_delta_loss < tolerance:
                break
            old_loss = loss_value

            # display loss with some frequency:
            if (i != 0) & ((i + 1) % logging_frequency == 0):
                print_str = (
                    "\t\t"
                    + str(i + 1)
                    + "\t"
                    + str(loss_value)
                    + "\t"
                    + str(p_delta_loss)
                )
                #print(print_str)
            # do backpropagation of loss and optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # display final loss
        #print("\t\tFinal loss:")
        #print_str = (
        #    "\t\t" + str(i + 1) + "\t" + str(loss_value) + "\t" + str(p_delta_loss)
        #)
        #print(print_str)

        if loss_value < best_final_loss:
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve

    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve


# %%
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
y = df['chd'].to_numpy()  # classification problem of CHD or no CHD
X = df.drop(columns=['chd']).to_numpy()  # rest of the attributes, remove 'CHD' column

# Compute size of X
N, M = X.shape  # N = observations, M = attributes (except 'chd')
N_numbers = np.arange(1, N + 1)

# Normalize the datapoints to have a mean of 0
mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X - mu) / sigma

# count the number of CHD diagnosed males
ones_count = np.count_nonzero(y == 1)
zeros_count = np.count_nonzero(y == 0)
perCHD = round(100 * ones_count / len(y), 1)
print('Males with CHD: {0} ({2}%), Males without CHD: {1}'.format(ones_count, zeros_count, perCHD))

# %%
# Find the best regularization value for logistic regression.ipynb
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import matplotlib.pyplot as plt

lambda_reg = np.linspace(1, 100, 30)

# Using K-fold 10 cross validation
K_fold = 10
CV = model_selection.KFold(K_fold)

# error matrices for collecting the determined errors
train_error = np.zeros((K_fold, len(lambda_reg)))
test_error = np.zeros((K_fold, len(lambda_reg)))
coefficient_norm = np.zeros((K_fold, len(lambda_reg)))

i = 0  # Reset i to 0 at the start of the outer loop
for train_index, test_index in CV.split(X, y):
    print("Crossvalidation fold: {0}/{1}".format(i + 1, 10))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    for l in range(len(lambda_reg)):
        # print("Regularization constant: {0}".format(lambda_reg[l]))

        # train model
        model_logreg = LogisticRegression(penalty="l2", C=1 / lambda_reg[l], max_iter=500)
        model_logreg.fit(X_train, y_train)

        # predict test and training data
        y_est_test = model_logreg.predict(X_test)
        y_est_train = model_logreg.predict(X_train)

        error_rate_test = 100 * np.sum(y_est_test != y_test) / len(y_test)
        error_rate_train = 100 * np.sum(y_est_train != y_train) / len(y_train)

        w_est = model_logreg.coef_[0]
        coeff_norm = np.sqrt(np.sum(w_est ** 2))

        test_error[i, l] = error_rate_test
        train_error[i, l] = error_rate_train
        coefficient_norm[i, l] = coeff_norm

    i += 1

# Plot the classification error rate
plt.figure()
plt.plot(lambda_reg, np.mean(test_error, axis=0), 'r.-', label='Test error')  # plot the mean errors of each lambda
plt.plot(lambda_reg, np.mean(train_error, axis=0), 'b.-', label='Training error')  # plot the mean errors of each lambda
plt.legend()
plt.xlabel('Regularization constant, $\log_{10}(\lambda)$')
plt.ylabel("Error rate (%)")
plt.show()

print("The best lambdas = [10,50]")
print(10 ** 0)
print(10 ** 2)

# %%
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

# Find the optimal number of neighbors

# Maximum number of neighbors
L = np.arange(1, 50 + 1)

# cross validation
CV = model_selection.KFold(10)

# matrices for saving the errors
test_error = np.zeros((10, len(L)))
train_error = np.zeros((10, len(L)))

# index for loop
i = 0

for train_index, test_index in CV.split(X, y):
    print("Crossvalidation fold: {0}/{1}".format(i + 1, 10))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    # Fit classifier and classify the test points for each neighbors
    for idx, l in enumerate(L):
        metric = "cosine"  # distance measure
        knclassifier = KNeighborsClassifier(n_neighbors=l, p=1, metric=metric, metric_params={})  # choose the model
        knclassifier.fit(X_train, y_train)  # fits the model to X_train, y_train

        # predict test and training data
        y_est_test = knclassifier.predict(X_test)  # estimate y from X_test
        y_est_train = knclassifier.predict(X_train)  # estimate y from X_train

        # find error
        error_rate_test = np.sum(y_est_test != y_test) / len(y_test)
        error_rate_train = np.sum(y_est_train != y_train) / len(y_train)

        # adding error
        test_error[i, idx] = error_rate_test
        train_error[i, idx] = error_rate_train

    i += 1

# Plot the classification error rate
plt.figure()
plt.plot(L, 100 * np.mean(test_error, axis=0), 'r.-', label='Test error')
plt.plot(L, 100 * np.mean(train_error, axis=0), 'b.-', label='Training error')
plt.legend()
plt.xlabel("Number of neighbors")
plt.ylabel("Error rate (%)")
plt.show()

print('The optimal k-nearest neigbor is k = [1, 25]')

# %%
# Two-layer cross validation

# K-fold for cross validation
k = 0  # index of outerfold
Kfold = 10  # maximum CV
CV = model_selection.KFold(Kfold)  # model for CV

# for evaluation of logistic regression
lambdas = np.linspace(1, 50, 30)
chosen_lambdas = np.zeros(Kfold)

# for evaluation of KNN
ks = np.arange(10, 50 + 1)
chosen_ks = np.zeros(Kfold)

# arrays to save the predicted values
y_true = []
y_est_KNN_All = []
y_est_LOGREG_All = []
y_est_base_All = []

# miss classification matrices
Missclass_KNN = np.zeros(Kfold)
Missclass_LOGREG = np.zeros(Kfold)
Missclass_base = np.zeros(Kfold)

# Outer fold
for train_index_out, test_index_out in CV.split(X, y):
    print("# Outer fold: {0}/{1}".format(k + 1, Kfold))

    # Extract the training and test set for the outer-fold
    X_train_out = X[train_index_out, :]
    y_train_out = y[train_index_out]
    X_test_out = X[test_index_out, :]
    y_test_out = y[test_index_out]

    # index for inner fold for chosing the right k and lambda
    i = 0

    # to save the errors for the different lambda- and ks-values
    test_error_lambdas = np.zeros((Kfold, len(lambdas)))
    test_error_ks = np.zeros((Kfold, len(ks)))

    # inner fold for chosing the right k and lambda
    for train_index, test_index in CV.split(X_train_out, y_train_out):

        # Extract training and test set for current inner fold
        X_train = X_train_out[train_index, :]
        y_train = y_train_out[train_index]
        X_test = X_train_out[test_index, :]
        y_test = y_train_out[test_index]

        # find optimal lambda value for the training set
        for idx, lambdas_values in enumerate(lambdas):
            # train model
            model_logreg = LogisticRegression(penalty="l2", C=1 / lambdas_values, max_iter=500)
            model_logreg.fit(X_train, y_train)

            # predict test and training data
            y_est_test_logreg = model_logreg.predict(X_test)

            # find and add error of predictions
            test_error_lambdas[i, idx] = np.sum(y_est_test_logreg != y_test) / len(y_test)

        # find optimal k-value for KNN
        for idx, ks_values in enumerate(ks):
            metric = "cosine"
            knclassifier = KNeighborsClassifier(n_neighbors=int(ks_values), p=1, metric=metric,
                                                metric_params={})  # choose the model
            knclassifier.fit(X_train, y_train)  # fits the model to X_train, y_train

            # predict test and training data
            y_est_test_KNN = knclassifier.predict(X_test)  # estimate y from X_test

            # find and error
            test_error_ks[i, idx] = np.sum(y_est_test_KNN != y_test) / len(y_test)

        i += 1

    # find index of lowest mean test error accross the rows
    min_index_lambdas = np.argmin(np.mean(test_error_lambdas, axis=0))
    # lambda value for logistic regression
    chosen_lambdas[k] = lambdas[min_index_lambdas]

    min_index_ks = np.argmin(np.mean(test_error_ks, axis=0))  # find index of lowest test error accross the rows
    # k-value for nearest neighbor
    chosen_ks[k] = ks[min_index_ks]

    ### Train the outer-set with the chosen k and lambda ###
    ## The true y-values
    y_true.append(y_test_out)

    ## K-nearest neigbors
    # train KNN
    knclassifier = KNeighborsClassifier(n_neighbors=int(chosen_ks[k]), p=1, metric=metric, metric_params={})
    knclassifier.fit(X_train_out, y_train_out)
    # test
    y_est_KNN = knclassifier.predict(X_test_out)
    y_est_KNN_All.append(y_est_KNN)
    # find error
    Missclass_KNN[k] = 100 * (np.sum(y_est_KNN != y_test_out) / len(y_test_out))

    ## Logistic Regression
    # train Logistic Regression
    logisticclassifier = LogisticRegression(C=1 / chosen_lambdas[k], max_iter=500)
    logisticclassifier.fit(X_train_out, y_train_out)
    # test
    y_est_LOGREG = logisticclassifier.predict(X_test_out)
    y_est_LOGREG_All.append(y_est_LOGREG)
    # find error
    Missclass_LOGREG[k] = 100 * (np.sum(y_est_LOGREG != y_test_out) / len(y_test_out))

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
        if a > b:  # If the class of no CHD (a) is largest
            y_est_base = np.full(len(y_test_out), 0)  # baseline predicts all y's to not have CHD
            y_est_base_All.append(y_est_base)
            break
        else:  # If the class of CHD (b) is largest
            y_est_base = np.full(len(y_test_out), 1)  # baseline predicts all y's to have CHD
            y_est_base_All.append(y_est_base)
            break
    # add error to matrix
    Missclass_base[k] = 100 * (np.sum(y_est_base != y_test_out) / len(y_test_out))  # error of each innerfold as columns

    # index for outerfold
    k += 1

# set the results up in datafram table.
Class_df_columns = pd.MultiIndex.from_tuples([
    ("Outer fold", "i"),
    ("KNN", "k"),
    ("KNN", "Error_test (%)"),
    ("Logistic regression", "lambda"),
    ("Logistic regression", "Error_test(%)"),
    ("Baseline", "Error_test(%)")])

Outer_fold = np.round(np.arange(1, Kfold + 1), 0)

Class_df_data = np.array([Outer_fold,
                          np.round(chosen_ks),  # k-value
                          np.round(Missclass_KNN, 1),  # Error_test of KNN
                          np.round(chosen_lambdas),  # Lambda
                          np.round(Missclass_LOGREG, 1),  # Error_test of logistic function
                          np.round(Missclass_base, 1)]).T

Class_df = pd.DataFrame(Class_df_data, columns=Class_df_columns)
print(Class_df)

# %%
# Comparison of models via McNemar

# import the McNemar test function from provided dtutools
from dtuimldmtools import mcnemar

alpha = 0.05  # confidence level
print("\n")

# Convert lists to numpy arrays and stack them properly
y_true_array = np.hstack(y_true)
y_est_KNN_All_array = np.hstack(y_est_KNN_All)
y_est_LOGREG_array = np.hstack(y_est_LOGREG_All)
y_est_base_array = np.hstack(y_est_base_All)

# KNN vs. LOGREG
[thetahat, CI, p] = mcnemar(y_true_array, y_est_KNN_All_array, y_est_LOGREG_array, alpha=alpha)
print(thetahat)

# KNN vs. Base
[thetahat, CI, p] = mcnemar(y_true_array, y_est_KNN_All_array, y_est_base_array, alpha=alpha)
print(thetahat)

# Base vs. LOGREG
[thetahat, CI, p] = mcnemar(y_true_array, y_est_base_array, y_est_LOGREG_array, alpha=alpha)
print(thetahat)

# %%
# Calculate the precision and recall for predictions from CV
from sklearn.metrics import precision_score, recall_score

precision_KNN = precision_score(y_true_array, y_est_KNN_All_array)
recall_KNN = recall_score(y_true_array, y_est_KNN_All_array)

precision_LOGREG = precision_score(y_true_array, y_est_LOGREG_array)
recall_LOGREG = recall_score(y_true_array, y_est_LOGREG_array)

precision_base = precision_score(y_true_array, y_est_base_array)
recall_base = recall_score(y_true_array, y_est_base_array)

print(f"Precision_KNN: {precision_KNN:.2f}")
print(f"Recall_KNN: {recall_KNN:.2f}")

print(f"Precision_LOGREG: {precision_LOGREG:.2f}")
print(f"Recall_LOGREG: {recall_LOGREG:.2f}")

print(f"Precision_base: {precision_base:.2f}")
print(f"Recall_base: {recall_base:.2f}")
print("Positive labels in y_true:", np.sum(y_true_array))
print("Positive predictions in y_pred_base:", np.sum(y_est_base_array))
# %%
# Find weights of features with logistic regression with lambda = 20

# Chosen lambda
lambda_value = 20

# Add a column of 1's to express the intercept
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1

# hold out method
test_proportion = 1 / 3
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=test_proportion
)

# estimations of w's for the chosen lambda
model_LOGREG = LogisticRegression(penalty="l2", C=1 / lambda_value)

# train model
model_LOGREG.fit(X_train, y_train)

# find the estimated fitted weight values of the features
w_est = model_LOGREG.coef_[0]

# print weights
print("Weights:")
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], w_est[m]))

#%%
from rlr_validate_custom import rlr_validate_custom
from sklearn import model_selection
from ann_validate import ann_validate
import torch
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from scipy.stats import zscore

url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"

# Load the SAheart dataset
df = pd.read_csv(url, index_col='row.names')

# Convert binary text data to numbered categories
df['famhist'] = pd.Categorical(df['famhist']).codes

# Extract the target attribute, and remove it from the training data
y = np.asarray(np.asmatrix(df["typea"].values).T).squeeze()
df = df.drop(columns=["typea"])

# Attribute names
attributeNames = list(map(lambda x: x.capitalize(), df.columns.tolist()))

# Convert the training data to a numpy array
X = df.to_numpy()
N, M = X.shape

X = zscore(X, ddof=1)  # Mean = 0, Std = 1
y = zscore(y, ddof=1)  # Mean = 0, Std = 1

# ---
# End of data loading
# ---


# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
# CV = model_selection.KFold(K, shuffle=False)

lambda_count = 20

# Values of lambda
lambdas = np.logspace(0, 4, lambda_count)
n_hidden_units_range = range(1, 30)
max_iter = 10000

#Actual y-s
y_true = []

# Linear regression
lr_error_train = np.empty((K, 1))
lr_error_test = np.empty((K, 1))
best_lambda = np.empty((K, 1))
y_est_lr = []

# ANN
ann_error_train = np.empty((K, 1))
ann_error_test = np.empty((K, 1))
best_hidden_units = np.empty((K, 1))
y_est_ann = []

# Baseline
baseline_error_test = np.empty((K, 1))
y_est_base = []


w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    innerCV = model_selection.KFold(K, shuffle=True)

    y_true.append(y_test)


    # ----
    # Linear regression
    # ----

    (   opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate_custom(X_train, y_train, lambdas, K, innerCV)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Compute mean squared error with regularization with optimal lambda
    lr_error_train[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    lr_error_test[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )
    best_lambda[k] = opt_lambda
    y_est_lr.append(X_test @ w_rlr[:, k])

    # ----
    # ANN
    # ----

    loss_fn = torch.nn.MSELoss()
    models = np.empty((len(n_hidden_units_range),), dtype=object)

    # Generate out models with different number of hidden units
    for i, n_hidden_units in enumerate(n_hidden_units_range):
        model = torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units),  # Input layer
            torch.nn.Tanh(),  # Hidden activation (ReLU is also good)
            torch.nn.Linear(n_hidden_units, 1)   # Output layer (NO SIGMOID for regression)
        )
        models[i] = model

    (
        opt_val_err,
        opt_model_index,
        opt_model,
        opt_network,
        train_err_vs_lambda,
        test_err_vs_lambda
        #This does a folding cross-validation
    ) = ann_validate(torch.Tensor(X_train), torch.Tensor(y_train), models, loss_fn, 1, max_iter, K, innerCV)

    #ann_error_train[k] = np.square(y_train-opt_network(torch.Tensor(X_train)).detach().numpy()).sum()/y_train.shape[0]
    #ann_error_test[k] = np.square(y_train-opt_network(torch.Tensor(X_train)).detach().numpy()).sum()/y_train.shape[0]

    ann_error_train[k] = train_err_vs_lambda[opt_model_index]
    ann_error_test[k] = test_err_vs_lambda[opt_model_index]
    best_hidden_units[k] = n_hidden_units_range[opt_model_index]
    y_est_ann.append(opt_network(torch.Tensor(X_test)).detach().numpy())

    # ----
    # Baseline
    # ----

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Compute the mean of y_train as the baseline prediction
    y_baseline = np.mean(y_train)

    # Predict the same mean value for all test samples
    y_pred_baseline = np.full_like(y_test, y_baseline)

    # Compute and store the MSE
    mse = mean_squared_error(y_test, y_pred_baseline)
    baseline_error_test[k] = mse
    y_est_base.append(y_pred_baseline)


    k += 1

#%%
results = pd.DataFrame({
    "h" : best_hidden_units.flatten(),
    "ann_error_test": ann_error_test.flatten(),

    "lambda": best_lambda.flatten(),
    "lr_error_test": lr_error_test.flatten(),

    "baseline_error_test": baseline_error_test.flatten(),
})
results
#%%
latex_table = r"""\begin{tabular}{c|cc|cc|c}
\toprule
\textbf{Outer fold} & \multicolumn{2}{c|}{\textbf{ANN}} & \multicolumn{2}{c|}{\textbf{Linear regression}} & \textbf{baseline} \\
$i$ & $h_i^*$ & $E_i^{\text{test}}$ & $\lambda_i^*$ & $E_i^{\text{test}}$ & $E_i^{\text{test}}$ \\
\midrule
"""

for i, row in results.iterrows():
    latex_table += f"{i+1} & {row['h']} & {row['ann_error_test']:.3f} & {row['lambda']:.2f} & {row['lr_error_test']:.3f} & {row['baseline_error_test']:.3f} \\\\\n"

latex_table += r"\bottomrule" + "\n" + r"\end{tabular}"
print(latex_table)

#%%
from dtuimldmtools.statistics.statistics import ttest_twomodels
import numpy as np

# Compute the Jeffreys interval
alpha = 0.05
rho = 1/K

# Convert lists to numpy arrays and stack them properly
y_true_array = np.concatenate(y_true).ravel()
y_est_ANN_array = np.concatenate(y_est_ann).ravel()
y_est_LR_array = np.concatenate(y_est_lr).ravel()
y_est_base_array = np.concatenate(y_est_base).ravel()

def run_ttest(y_true, y_A, y_B):
    mean_diff, confidence_interval, p_value =  ttest_twomodels(y_true, y_A, y_B, alpha=alpha)
    return mean_diff, confidence_interval, p_value

# Perform t-tests
mean_diff_ANN_LR, CI_ANN_LR, p_ANN_LR = run_ttest(y_true_array, y_est_ANN_array, y_est_LR_array)
mean_diff_ANN_base, CI_ANN_base, p_ANN_base = run_ttest(y_true_array, y_est_ANN_array, y_est_base_array)
mean_diff_LR_base, CI_LR_base, p_LR_base = run_ttest(y_true_array, y_est_LR_array, y_est_base_array)

results_df = pd.DataFrame({
    "Comparison": ["ANN vs LR", "ANN vs Baseline", "LR vs Baseline"],
    "Mean Difference": [mean_diff_ANN_LR, mean_diff_ANN_base, mean_diff_LR_base],
    "Confidence Interval Min": [CI_ANN_LR[0], CI_ANN_base[0], CI_LR_base[0]],
    "Confidence Interval Max": [CI_ANN_LR[1], CI_ANN_base[1], CI_LR_base[1]],
    "p-value": [p_ANN_LR, p_ANN_base, p_LR_base]
})
#%%
print(results_df.to_latex())
#%%
import matplotlib.pyplot as plt
import pandas as pd


# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(results_df["Comparison"], results_df["Mean Difference"],
             yerr=[results_df["Mean Difference"] - results_df["Confidence Interval Min"], results_df["Confidence Interval Max"] - results_df["Mean Difference"]],
             fmt='o', capsize=5, capthick=2, color='teal', ecolor='gray')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title("Mean Differences with 95% Confidence Intervals")
plt.ylabel("Mean Difference")
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate p-values
for i, row in results_df.iterrows():
    plt.text(i, row["Mean Difference"] + 0.002, f"p = {row['p-value']:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

#%%
import pandas as pd
import numpy as np
from scipy.stats import zscore

url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"

# Load the SAheart dataset
df = pd.read_csv(url, index_col='row.names')

# Convert binary text data to numbered categories
df['famhist'] = pd.Categorical(df['famhist']).codes

# Extract the target attribute, and remove it from the training data
y = np.asarray(np.asmatrix(df["typea"].values).T).squeeze()
df = df.drop(columns=["typea"])

# Attribute names
attributeNames = list(map(lambda x: x.capitalize(), df.columns.tolist()))

# Convert the training data to a numpy array
X = df.to_numpy()
N, M = X.shape

X = zscore(X, ddof=1)  # Mean = 0, Std = 1
y = zscore(y, ddof=1)  # Mean = 0, Std = 1

#%%
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from dtuimldmtools import rlr_validate


# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
# CV = model_selection.KFold(K, shuffle=False)

lambda_count = 50

# Values of lambda
lambdas = np.logspace(-2, 6, lambda_count)

# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )
    #loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = (
        np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test[k] = (
        np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    )
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        plt.figure(k, figsize=(14, 8))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        plt.xlabel("Regularization factor")
        plt.ylabel("Mean Coefficient Values")
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        plt.subplot(1, 2, 2)
        plt.title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
        plt.loglog(
            lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
        )
        plt.xlabel("Regularization factor")
        plt.ylabel("Squared error (crossvalidation)")
        plt.legend(["Train error", "Validation error"])
        plt.grid()

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1
plt.savefig('regression_lambda.png')
plt.show()
# Display results
print("Linear regression without feature selection:")
print("- Training error: {0}".format(Error_train.mean()))
print("- Test error:     {0}".format(Error_test.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()
    )
)
print("Regularized linear regression:")
print("- Training error (mse): {0}".format(Error_train_rlr.mean()))
print("- Test error (mse):     {0}".format(Error_test_rlr.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train_rlr.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test_rlr.sum())
        / Error_test_nofeatures.sum()
    )
)

print("Weights in last fold:")
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, -1], 2)))

#%%

import numpy as np


def rlr_validate_custom(X, y, lambdas, cvf, CV):
    """Validate regularized linear regression model using 'cvf'-fold cross validation.
    Find the optimal lambda (minimizing validation error) from 'lambdas' list.
    The loss function computed as mean squared error on validation set (MSE).
    Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
    average weight values for all lambdas, MSE train&validation errors for all lambdas.
    The cross validation splits are standardized based on the mean and standard
    deviation of the training set when estimating the regularization strength.

    Parameters:
    X       training data set
    y       vector of values
    lambdas vector of lambda values to be validated
    cvf     number of crossvalidation folds

    Returns:
    opt_val_err         validation error for optimum lambda
    opt_lambda          value of optimal lambda
    mean_w_vs_lambda    weights as function of lambda (matrix)
    train_err_vs_lambda train error as function of lambda (vector)
    test_err_vs_lambda  test error as function of lambda (vector)
    """
    M = X.shape[1]
    w = np.empty((M, cvf, len(lambdas)))
    train_error = np.empty((cvf, len(lambdas)))
    test_error = np.empty((cvf, len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train - X_train @ w[:, f, l].T, 2).mean(
                axis=0
            )
            test_error[f, l] = np.power(y_test - X_test @ w[:, f, l].T, 2).mean(axis=0)

        f = f + 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

    return (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    )
