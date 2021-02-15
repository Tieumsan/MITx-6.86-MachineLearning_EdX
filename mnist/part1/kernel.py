import numpy as np

### Functions for you to fill in ###


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    kernel_matrix = (np.inner(X, Y) + c) ** p
    return kernel_matrix


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    XT_X = np.mat([np.matmul(row, row.T) for row in X]).T
    YT_Y = np.mat([np.matmul(row, row.T) for row in Y]).T

    XTX_matrix = np.repeat(XT_X, Y.shape[0], axis=1)
    YTY_matrix = np.repeat(YT_Y, X.shape[0], axis=1).T

    distance_matrix = np.asarray(XTX_matrix + YTY_matrix - 2 * (np.matmul(X, Y.T)), dtype='float64')
    kernel_matrix = np.exp(-gamma * distance_matrix)
    return kernel_matrix
