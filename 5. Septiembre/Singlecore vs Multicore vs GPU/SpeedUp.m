close all;


% Cargar los datos de los archivos CSV corregidos
data_mkl_single = csvread('final_corrected_octubre_mkl_single.csv', 1, 0); 
data_mkl_multi = csvread('final_corrected_octubre_mkl_multi.csv', 1, 0); 

data_blis_single = csvread('octubre_blis_single.csv', 1, 0); 
data_blis_multi = csvread('octubre_blis_multi.csv', 1, 0); 

data_aocl_single = csvread('octubre_aocl_single.csv', 1, 0);
data_aocl_multi = csvread('octubre_aocl_multi.csv', 1, 0);

data_gpu_mem = csvread('results_octubre_gpu_mem.csv', 1, 0);
data_gpu_mem5 = csvread('results_octubre_gpu_mem_last5.csv', 1, 0);
data_gpu_nomem = csvread('results_octubre_gpu_nomem.csv', 1, 0);
data_gpu_nomem5 = csvread('results_octubre_gpu_nomem_last5.csv', 1, 0);


% Extraer la dimensión de la matriz (N) y el tiempo de ejecución
N_mkl_single = data_mkl_single(:, 1);
time_mkl_single = data_mkl_single(:, 3);
N_mkl_multi = data_mkl_multi(:, 1);
time_mkl_multi = data_mkl_multi(:, 3);

N_blis_single = data_blis_single(:, 1);
time_blis_single = data_blis_single(:, 3);
N_blis_multi = data_blis_multi(:, 1);
time_blis_multi = data_blis_multi(:, 3);

N_aocl_single = data_aocl_single(:, 1);
time_aocl_single = data_aocl_single(:, 3);
N_aocl_multi = data_aocl_multi(:, 1);
time_aocl_multi = data_aocl_multi(:, 3);

N_gpu_mem = data_gpu_mem(:, 1);
time_gpu_mem = data_gpu_mem(:, 2) * 1E-3;

N_gpu_nomem = data_gpu_nomem(:, 1);
time_gpu_nomem = data_gpu_nomem(:, 2) * 1E-3;

N_gpu_mem5 = data_gpu_mem5(:, 1);
time_gpu_mem5 = data_gpu_mem5(:, 2) * 1E-3;

N_gpu_nomem5 = data_gpu_nomem5(:, 1);
time_gpu_nomem5 = data_gpu_nomem5(:, 2) * 1E-3;

% Calcular Speed Up para cada caso
speedup_mkl = time_mkl_single ./ time_mkl_multi;
speedup_blis = time_blis_single ./ time_blis_multi;
speedup_aocl = time_aocl_single ./ time_aocl_multi;

% Gráficas de tiempos absolutos para MKL, BLIS y AOCL
figure;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
plot(N_mkl_single, time_mkl_single, '-o', 'Color', 'b', 'DisplayName', 'MKL Single Core');
hold on;
plot(N_mkl_multi, time_mkl_multi, '-o', 'Color', 'b', 'DisplayName', 'MKL Multi Core');
plot(N_blis_single, time_blis_single, '-s', 'Color', 'r', 'DisplayName', 'BLIS Single Core');
plot(N_blis_multi, time_blis_multi, '-s', 'Color', 'r', 'DisplayName', 'BLIS Multi Core');
plot(N_aocl_single, time_aocl_single, '-d', 'Color', 'g', 'DisplayName', 'AOCL Single Core');
plot(N_aocl_multi, time_aocl_multi, '-d', 'Color', 'g', 'DisplayName', 'AOCL Multi Core');
xlabel('Matrix Dimension (N)');
ylabel('Execution Time (s)');
title('Execution Time for MKL, BLIS and AOCL');
lgd1 = legend;
lgd1.FontSize = 14;
grid on;
ylim([0 30]);
hold off;

% Gráficas de Speed Up
figure;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
plot(N_mkl_single, speedup_mkl, '-o', 'Color', 'b', 'DisplayName', 'MKL Speed Up');
hold on;
plot(N_blis_single, speedup_blis, '-s', 'Color', 'r', 'DisplayName', 'BLIS Speed Up');
plot(N_aocl_single, speedup_aocl, '-d', 'Color', 'g', 'DisplayName', 'AOCL Speed Up');
xlabel('Matrix Dimension (N)');
ylabel('Speed Up');
title('Speed Up (Single Core vs Multi Core)');
lgd2 = legend;
lgd2.FontSize = 14;
grid on;
hold off;

% Gráfica comparativa final
figure;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
semilogy(N_mkl_multi, time_mkl_multi, '-o', 'Color', 'b', 'DisplayName', 'MKL Multi Core');
hold on;
semilogy(N_blis_multi, time_blis_multi, '-s', 'Color', 'r', 'DisplayName', 'BLIS Multi Core');
semilogy(N_aocl_multi, time_aocl_multi, '-d', 'Color', 'g', 'DisplayName', 'AOCL Multi Core');
semilogy(N_gpu_mem, time_gpu_mem, '-^', 'Color', 'm', 'DisplayName', 'CuPy Mem');
semilogy(N_gpu_nomem, time_gpu_nomem, '-^', 'Color', 'c', 'DisplayName', 'CuPy NoMem');
semilogy(N_gpu_mem5, time_gpu_mem5, '-^', 'Color', 'm', 'DisplayName', 'CuPy Mem Last 5');
semilogy(N_gpu_nomem5, time_gpu_nomem5, '-^', 'Color', 'c', 'DisplayName', 'CuPy NoMem Last 5');
xlabel('Matrix Dimension (N)');
ylabel('Execution Time (s)');
title('Execution Time Comparison');
lgd3 = legend;
lgd3.FontSize = 14;
grid on;
hold off;
