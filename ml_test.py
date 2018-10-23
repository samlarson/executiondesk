# SOURCE: https://github.com/dennybritz/nn-from-scratch/blob/master/nn_from_scratch.py
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from scipy import stats


class ML:
    def simple_nn(self, df):
        X, y = df.iloc[:, 2:8], df.iloc[:, 8:9]
        X, y = df.iloc[:, 2:8], df.iloc[:, 8:9]
        X, y = X.values, y.values
        y = y.ravel()

        clf = sklearn.linear_model.BayesianRidge()
        z = clf.fit(X, y)
        print(z)
        build_model(nn_hdim)

    def bayes_ridge(self, df):
        X, y = df.iloc[:, 2:8], df.iloc[:, 8:9]
        X, y = df.iloc[:, 2:8], df.iloc[:, 8:9]
        X, y = X.values, y.values
        y = y.ravel()
        # Create weights with a precision lambda_ of 4.
        lambda_ = 4.
        w = np.zeros(6)
        # Only keep 10 weights of interest
        relevant_features = np.random.randint(0, 6, 10)
        for i in relevant_features:
            w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
        # Create noise with a precision alpha of 50.
        alpha_ = 50.
        noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=82)

        # #############################################################################
        # Fit the Bayesian Ridge Regression and an OLS for comparison
        clf = sklearn.linear_model.BayesianRidge(compute_score=True)
        clf.fit(X, y)

        ols = sklearn.linear_model.LinearRegression()
        ols.fit(X, y)

        # #############################################################################
        # Plot true weights, estimated weights, histogram of the weights, and
        # predictions with standard deviations
        lw = 2
        plt.figure(figsize=(6, 5))
        plt.title("Weights of the model")
        plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
                 label="Bayesian Ridge estimate")
        plt.plot(w, color='gold', linewidth=lw, label="Ground truth")
        plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")
        plt.xlabel("Features")
        plt.ylabel("Values of the weights")
        plt.legend(loc="best", prop=dict(size=12))

        plt.figure(figsize=(6, 5))
        plt.title("Histogram of the weights")
        plt.hist(clf.coef_, bins=6, color='gold', log=True,
                 edgecolor='black')
        plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
                    color='navy', label="Relevant features")
        plt.ylabel("Features")
        plt.xlabel("Values of the weights")
        plt.legend(loc="upper left")

        plt.figure(figsize=(6, 5))
        plt.title("Marginal log-likelihood")
        plt.plot(clf.scores_, color='navy', linewidth=lw)
        plt.ylabel("Score")
        plt.xlabel("Iterations")

        # Plotting some predictions for polynomial regression
        def f(x, noise_amount):
            y = np.sqrt(x) * np.sin(x)
            noise = np.random.normal(0, 1, len(x))
            return y + noise_amount * noise

        degree = 10
        X = np.linspace(0, 10, 100)
        y = f(X, noise_amount=0.1)
        clf_poly = sklearn.linear_model.BayesianRidge()
        clf_poly.fit(np.vander(X, degree), y)

        X_plot = np.linspace(0, 11, 25)
        y_plot = f(X_plot, noise_amount=0)
        y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
        plt.figure(figsize=(6, 5))
        plt.errorbar(X_plot, y_mean, y_std, color='navy',
                     label="Polynomial Bayesian Ridge Regression", linewidth=lw)
        plt.plot(X_plot, y_plot, color='gold', linewidth=lw,
                 label="Ground Truth")
        plt.ylabel("Output y")
        plt.xlabel("Feature X")
        plt.legend(loc="lower left")
        plt.show()


matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

np.random.seed(3)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# Plot the decision boundary
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")

num_examples = len(X)  # training set size
nn_input_dim = 2  # input layer dimensionality
nn_output_dim = 2  # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon = 0.01  # learning rate for gradient descent
reg_lambda = 0.01  # regularization strength


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model)))

    return model


# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")

plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
plt.show()

