#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_iter = m / batch;
    for (int i = 0; i < num_iter; i++) {
        size_t batch_size = batch;
        float *X_iter = new float[batch_size * n];   // batch_size * n
        unsigned char *y_iter = new unsigned char[batch_size];   // batch_size
        for (int j = 0; j < batch_size; j++) {
            for (int t = 0; t < n; t++) {
                X_iter[j * n + t] = X[(batch_size * i + j) * n + t];
            }
            y_iter[j] = y[batch_size * i + j];
        }
        // batch_size * k, Z[u][v] = sum(X[u][t] * theta[t][v])
        // X[u][t] = X[u * n + t], theta[t][v] = theta[t * k + v]
        float *Z_iter = new float[batch_size * k];
        for (int u = 0; u < batch_size; u++) {
            for (int v = 0; v < k; v++) {
                Z_iter[u * k + v] = 0;
                for (int t = 0; t < n; t++) {
                    Z_iter[u * k + v] += X_iter[u * n + t] * theta[t * k + v];
                }
                Z_iter[u * k + v] = std::exp(Z_iter[u * k + v]);
            }
        }
        for (int j = 0; j < batch_size; j++) {
            float norm_sum = 0;
            for (int t = 0; t < k; t++) {
                norm_sum += Z_iter[j * k + t];
            }
            for (int t = 0; t < k; t++) {
                Z_iter[j * k + t] /= norm_sum; 
            }
            Z_iter[j * k + (int)y_iter[j]] -= 1;
        }
        // X_iter^T * Z_iter -> (n * batch_size) * (batch_size * k)
        float *D_iter = new float[n * k];
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < k; v++) {
                D_iter[u * k + v] = 0;
                for (int t = 0; t < batch_size; t++) {
                    D_iter[u * k + v] += X_iter[t * n + u] * Z_iter[t * k + v];
                }
                D_iter[u * k + v] /= batch_size;
            }
        }
        for (int j = 0; j < n * k; j++) theta[j] -= lr * D_iter[j];
        delete[] y_iter;
        delete[] X_iter;
        delete[] Z_iter;
        delete[] D_iter;
    }
    if (m % batch != 0) {
        size_t batch_size = m % batch;
        float *X_iter = new float[batch_size * n];   // batch_size * n
        unsigned char *y_iter = new unsigned char[batch_size];   // batch_size
        for (int j = 0; j < batch_size; j++) {
            for (int t = 0; t < n; t++) {
                X_iter[j * n + t] = X[(batch_size * num_iter + j) * n + t];
            }
            y_iter[j] = y[batch_size * num_iter + j];
        }
        // batch_size * k, Z[u][v] = sum(X[u][t] * theta[t][v])
        // X[u][t] = X[u * n + t], theta[t][v] = theta[t * k + v]
        float *Z_iter = new float[batch_size * k];
        for (int u = 0; u < batch_size; u++) {
            for (int v = 0; v < k; v++) {
                Z_iter[u * k + v] = 0;
                for (int t = 0; t < n; t++) {
                    Z_iter[u * k + v] += X_iter[u * n + t] * theta[t * k + v];
                }
                Z_iter[u * k + v] = std::exp(Z_iter[u * k + v]);
            }
        }
        for (int j = 0; j < batch_size; j++) {
            float norm_sum = 0;
            for (int t = 0; t < k; t++) {
                norm_sum += Z_iter[j * k + t];
            }
            for (int t = 0; t < k; t++) {
                Z_iter[j * k + t] /= norm_sum; 
            }
            Z_iter[j * k + (int)y_iter[j]] -= 1;
        }
        // X_iter^T * Z_iter -> (n * batch_size) * (batch_size * k)
        float *D_iter = new float[n * k];
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < k; v++) {
                D_iter[u * k + v] = 0;
                for (int t = 0; t < batch_size; t++) {
                    D_iter[u * k + v] += X_iter[t * n + u] * Z_iter[t * k + v];
                }
                D_iter[u * k + v] /= batch_size;
            }
        }
        for (int j = 0; j < n * k; j++) theta[j] -= lr * D_iter[j];
        delete[] y_iter;
        delete[] X_iter;
        delete[] Z_iter;
        delete[] D_iter;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
