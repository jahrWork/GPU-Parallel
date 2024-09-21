import numpy as np
import cupy as cp
import time
import os

os.environ["OMP_NUM_THREADS"] = "12" 


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
    print("\nWarming up the GPU\n")
    dim = 1000
    matrix = cp.empty((dim, dim), dtype=cp.float32)
    vector = cp.empty(dim, dtype=cp.float32)
    cupy_random_matrix(dim, matrix)
    cupy_random_vector(dim, vector)
    result = cupy_dot(matrix, vector)




def parallel_dot(matrix, vector, iterations):
    results = Parallel(n_jobs=-1)(delayed(np.dot)(matrix, vector) for i in range(iterations))
    return results

def test_cpu(dim):
    print("\nComparing performance for dimension N = {}\n".format(dim))
    
    # CPU benchmark
    np_matrix = np.empty((dim, dim), dtype=np.float32)
    np_vector = np.empty(dim, dtype=np.float32)
    numpy_random_matrix(dim, np_matrix)
    numpy_random_vector(dim, np_vector)
    
    start = time.time()
    for i in range(100000):
         numpy_result = i * numpy_dot(np_matrix, np_vector)
    numpy_time = time.time() - start
    print("NumPy (CPU) Time: {:.6f} seconds".format(numpy_time))

    

   


def test_gpu(dim):
    print("\nComparing performance for dimension N = {}\n".format(dim))
    
    # Warmup
    warmup_gpu()
    

    # GPU benchmark
    cp_matrix = cp.empty((dim, dim), dtype=cp.float32)
    cp_vector = cp.empty(dim, dtype=cp.float32)
    cupy_random_matrix(dim, cp_matrix)
    cupy_random_vector(dim, cp_vector)
    
    start = time.time()
    for i in range(10000):
        cupy_result = i * cupy_dot(cp_matrix, cp_vector)
    cupy_time = time.time() - start
    print("CuPy (GPU) Time: {:.6f} seconds".format(cupy_time))



    





