
% Test con diferentes valores de N y potencia
compare_perf(2500, 2);
compare_perf(2500, 3);
compare_perf(5000, 2);
compare_perf(10000, 2);
compare_perf(20000, 2);
compare_perf(30000, 2);

function compare_perf(dim, power)
    fprintf('\nComparing performance with data transfer time for dimension N = %d, power = %d\n', dim, power);

    % Generar matriz aleatoria en CPU
    np_matrix = single(rand(dim, dim));

    % Calcular la potencia enésima en CPU
    tic;
    for i = 1:100
        numpy_result = mpower(np_matrix, power);
    end
    numpy_time = toc;

    % Generar matriz aleatoria en GPU
    cp_matrix = gpuArray(single(rand(dim, dim)));

    % Calcular la potencia enésima en GPU
    tic;
    for i = 1:100
        cupy_result = mpower(cp_matrix, power);
    end
    cupy_time = toc;

    % Transferir el resultado de GPU a CPU
    tic;
    np_cupy_result = gather(cupy_result);
    cupy_transfer_time = toc;

    fprintf('NumPy (CPU) Time: %.6f seconds\n', numpy_time);
    fprintf('CuPy (GPU) Time: %.6f seconds\n', cupy_time);
    fprintf('CuPy (GPU) Time (including data transfer): %.6f seconds\n', cupy_time + cupy_transfer_time);

    speedup1 = numpy_time / cupy_time;
    fprintf('GPU is %.2f times faster than CPU for dimension N = %d, power = %d\n', speedup1, dim, power);
    speedup2 = numpy_time / (cupy_time + cupy_transfer_time);
    fprintf('GPU is %.2f times faster than CPU (including data transfer) for dimension N = %d, power = %d\n', speedup2, dim, power);
end