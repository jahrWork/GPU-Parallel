
# Main
function show_main_menu()
    while true
        println("Select an option:")
        println("1. CPU Performance Codes")
        println("2. GPU Performance Codes")
        println("3. Physics Problems Solved")
        println("4. Exit")
        print("Enter your choice: ")

        choice = readline()

        if choice == "1"
            show_cpu_codes()
        elseif choice == "2"
            show_gpu_codes()
        elseif choice == "3"
            show_physics_codes()
        elseif choice == "4"
            println("Exiting...")
            return
        else
            println("Invalid choice, please try again.")
        end
    end
end

# CPU
function show_cpu_codes()
    while true
        println("CPU Performance Codes:")
        println("1. Matrix Multiplication Functions Comparison")
        println("2. Matrix multiplication, Matrix-Vector multiplication, Vector multiplication")
        println("3. Back to main menu")
        print("Enter your choice: ")

        choice = readline()

        if choice == "1"
            println("Running Matrix Multiplication Functions Comparison...")
            include("cpu/dot_func_comparison-v2.jl")
            break
        elseif choice == "2"
            println("Running Sorting Algorithm Performance...")
            include("cpu/BLAS_levels.jl")
            break
        elseif choice == "3"
            return
        else
            println("Invalid choice, please try again.")
        end
    end
end

# GPU
function show_gpu_codes()
    while true
        println("GPU Performance Codes:")
        println("1. GPU Code Matrix Multiplication")
        println("2. GPU Code Matrix Multiplication Times")
        println("3. GPU Code Matrix Multiplication Functions Comparison GFlops")
        println("4. Back to main menu")
        print("Enter your choice: ")

        choice = readline()

        if choice == "1"
            println("Running Matrix Multiplication...")
            include("gpu/matrix_mult.jl")
            break
        elseif choice == "2"
            println("Running Matrix Multiplication Times...")
            include("gpu/matrix_mult_times.jl")
            break
        elseif choice == "3"
            println("Running Matrix Multiplication Functions Comparison GFlops...")
            include("gpu/GFLOPS_GPU.jl")
            break
        elseif choice == "4"
            return
        else
            println("Invalid choice, please try again.")
        end
    end
end

# Physics
function show_physics_codes()
    while true
        println("Physics Problems Solved:")
        println("1. Heat Equation")
        println("2. Adv-Dif Equation")
        println("3. Back to main menu")
        print("Enter your choice: ")

        choice = readline()

        if choice == "1"
            println("Running Heat Equation...")
            #include("")
            break
        elseif choice == "2"
            println("Running Adv-Dif Equation...")
            #include("")
            break
        elseif choice == "3"
            return
        else
            println("Invalid choice, please try again.")
        end
    end
end

show_main_menu()
