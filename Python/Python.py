import numpy as np
import cupy as cp
import time

def numpy_random_matrix(dim, preallocated_matrix):
    preallocated_matrix[...] = np.random.rand(dim, dim).astype(np.float32)

def numpy_random_vector(dim, preallocated_vector):
    preallocated_vector[...] = np.random.rand(dim).astype(np.float32)

def cupy_random_matrix(dim, preallocated_matrix):
    preallocated_matrix[...] = cp.random.rand(dim, dim, dtype=cp.float32)

def cupy_random_vector(dim, preallocated_vector):
    preallocated_vector[...] = cp.random.rand(dim, dtype=cp.float32)

def numpy_dot(matrix, vector):
    return np.dot(matrix, vector)

def cupy_dot(matrix, vector):
    return cp.dot(matrix, vector)

def warmup_gpu():
    dim = 100
    matrix = cp.empty((dim, dim), dtype=cp.float32)
    vector = cp.empty(dim, dtype=cp.float32)
    cupy_random_matrix(dim, matrix)
    cupy_random_vector(dim, vector)
    result = cupy_dot(matrix, vector)

def compare_performance(dim):
    print("\nComparing performance with data transfer time for dimension N = {}\n".format(dim))

    np_matrix = np.empty((dim, dim), dtype=np.float32)
    np_vector = np.empty(dim, dtype=np.float32)

    start = time.time()
    numpy_random_matrix(dim, np_matrix)
    numpy_random_vector(dim, np_vector)
    numpy_result = numpy_dot(np_matrix, np_vector)
    numpy_time = time.time() - start

    cp_matrix = cp.empty((dim, dim), dtype=cp.float32)
    cp_vector = cp.empty(dim, dtype=cp.float32)

    start = time.time()
    cupy_random_matrix(dim, cp_matrix)
    cupy_random_vector(dim, cp_vector)
    cupy_result = cupy_dot(cp_matrix, cp_vector)
    cupy_time = time.time() - start

    start = time.time()
    np_cupy_result = cp.asnumpy(cupy_result)
    cupy_transfer_time = time.time() - start

    print("NumPy (CPU) Time: {:.6f} seconds".format(numpy_time))
    print("CuPy (GPU) Time: {:.6f} seconds".format(cupy_time))
    print("CuPy (GPU) Time (including data transfer): {:.6f} seconds".format(cupy_time + cupy_transfer_time))

    speedup1 = numpy_time / (cupy_time + 1e-8)
    print("GPU is {:.2f} times faster than CPU for dimension N = {}".format(speedup1, dim))
    speedup2 = numpy_time / ((cupy_time + cupy_transfer_time) + 1e-8)
    print("GPU is {:.2f} times faster than CPU (including data transfer) for dimension N = {}".format(speedup2, dim))

# Calentamiento de GPU
warmup_gpu()

# Test con diferentes valores de N
compare_performance(2500)
compare_performance(5000)
compare_performance(10000)
compare_performance(20000)
compare_performance(30000)
