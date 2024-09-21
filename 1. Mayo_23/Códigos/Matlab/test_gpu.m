%% BENCHMARK MATLAB

% Warmup
warmup_gpu();

% Test con diferentes valores de N
test(5000);
test(10000);



function warmup_gpu()
    disp(" Warming up the GPU ");
    dim = 1000;
    matrix = gpuArray.rand(dim, dim, 'single');
    vector = gpuArray.rand(dim, 1, 'single');
    result = matrix * vector;
end

function test(dim)
    disp(" Comparing performance for dimension N = " + num2str(dim));

    % GPU Time
    cp_matrix = gpuArray(rand(dim, dim, 'single'));
    cp_vector = gpuArray(rand(dim, 1, 'single'));
    tic;
    for i = 1:10000
        cupy_result = i * cp_matrix * cp_vector;
    end
    cupy_time = toc;

    disp("CuPy (GPU) Time: " + num2str(cupy_time) + " seconds");
end


