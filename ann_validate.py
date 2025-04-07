import numpy as np
import torch
from sklearn import model_selection

from dtuimldmtools import (
    train_neural_net,
)


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
    for train_index, test_index in CV.split(X, y):
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
