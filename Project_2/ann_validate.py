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
