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
