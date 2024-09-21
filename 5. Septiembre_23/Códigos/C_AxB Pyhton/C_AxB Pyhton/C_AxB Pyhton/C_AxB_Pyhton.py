# #CÓDIGO SIN ACCESO A MEMORIA

#import csv
#import cupy as cp
#import numpy as np

#def ab_multiply(C, A, B, TIMES):
#    for _ in range(TIMES):
#        C[:] = cp.dot(A, B)  # gemm on GPU

#def perform_computation_and_record_time(N, TIMES):
#    print(N)
#    A = cp.random.rand(N, N).astype(cp.float32)
#    B = cp.random.rand(N, N).astype(cp.float32)
#    C = cp.empty((N, N))
    
#    start_event = cp.cuda.Event()
#    end_event = cp.cuda.Event()
#    start_event.record()
#    ab_multiply(C, A, B, TIMES)
#    end_event.record()
#    end_event.synchronize()

#    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
#    computation_times.append(elapsed_time)
#    N_list.append(N)
   

#N_ops = 2 * 10000**3

## Warm-up
#N = 5000
#TIMES = 2
#A = cp.random.rand(N, N).astype(cp.float32)
#B = cp.random.rand(N, N).astype(cp.float32)
#C = cp.empty((N, N))
#ab_multiply(C, A, B, TIMES)

#computation_times = []
#N_list = []

### TIMES(N) loop
##for N in range(250, 2499, 25):
##    TIMES = int(np.ceil(N_ops / (2 * N**3)))
##    perform_computation_and_record_time(N, TIMES)

## N(TIMES) loop
#for TIMES in range(5, 0, -1):
#    N = round((N_ops / (2.0 * TIMES)) ** (1./3))
#    perform_computation_and_record_time(N, TIMES)



## Write results to CSV
#with open(
#    "Resultados/results_octubre_gpu_nomem_last5.csv", mode="w", newline="", encoding="utf-8"
#) as file:
#    writer = csv.writer(file)
#    writer.writerows(zip(N_list, computation_times))





# CÓDIGO CON ACCESO A MEMORIA
import csv
import cupy as cp
import numpy as np

def ab_multiply(C, A, B, TIMES):
    for _ in range(TIMES):
        C[:] = cp.dot(A, B)  # gemm on GPU
        C_cpu = cp.asnumpy(C) # Transferencia de C de la GPU a la memoria principal

    
    

def perform_computation_and_record_time(N, TIMES):
    print(N)
    A = cp.random.rand(N, N).astype(cp.float32)
    B = cp.random.rand(N, N).astype(cp.float32)
    C = cp.empty((N, N))
    
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    ab_multiply(C, A, B, TIMES)
    end_event.record()
    end_event.synchronize()

    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
    computation_times.append(elapsed_time)
    N_list.append(N)

N_ops = 2 * 10000**3

# Warm-up
N = 5000
TIMES = 2
A = cp.random.rand(N, N).astype(cp.float32)
B = cp.random.rand(N, N).astype(cp.float32)
C = cp.empty((N, N))
ab_multiply(C, A, B, TIMES)

computation_times = []
N_list = []

## TIMES(N) loop
#for N in range(250, 2499, 25):
#    TIMES = int(np.ceil(N_ops / (2 * N**3)))
#    perform_computation_and_record_time(N, TIMES)

# N(TIMES) loop
for TIMES in range(5, 0, -1):
    N = round((N_ops / (2.0 * TIMES)) ** (1./3))
    perform_computation_and_record_time(N, TIMES)

# Write results to CSV
with open(
    "Resultados/results_octubre_gpu_mem_last5.csv", mode="w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerows(zip(N_list, computation_times))
