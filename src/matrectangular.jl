function rectangular_matrix_initialization(rows, cols)
    A = rand(Float32, rows, cols)
    B = rand(Float32, cols, rows)
    return A, B
  end
  
  function time_matrix_multilication_rectangular(rows, cols, N_cores, matinit, matmul)
    Time = zeros(length(rows))
    Theoretical_time = 1e9 / (4e9 * 512/32 * N_cores) 
    
    for (i, n) in enumerate(rows)
        A, B = matinit(rows[i], cols[i])    
        m = length(B)
        
        dt = matmul(A, B)
        
        Time[i] = 1e9 * dt / (2 * n * m)
        
        println("Rows=", rows[i], " Cols=", cols[i], " Time per operation =", Time[i], " nsec")
        println("Rows=", rows[i], " Cols=", cols[i], " Theoretical time per operation =", Theoretical_time, " nsec")
    end
    
    return Time, Theoretical_time
  end