%% BENCHMARK MATLAB CPU MULTICORE

% Test con diferentes valores de N
test(5000);
test(10000);

function test(dim)
    disp(" Comparing performance for dimension N = " + num2str(dim));

    % CPU Time
    np_matrix = rand(dim, dim, 'single');
    np_vector = rand(dim, 1, 'single');
    tic;
    parfor i = 1:10000
        numpy_result = i * np_matrix * np_vector;
    end
    numpy_time = toc;

    disp("NumPy (CPU, multi-core) Time: " + num2str(numpy_time) + " seconds");

end   
