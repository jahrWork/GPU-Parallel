<h1><ins>GPU Parallel Processing Documentation - Research Project 22-23 & 23-24</ins></h1>

> [!NOTE]
> - 🗂️ This repository gathers the documentation and work carried out by me (Pedro Rodríguez Jiménez) during the 2022/23 academic year, as well as the continuation of the work by his colleagues during the 2023/24 academic year.
> 
> - 🎯 Objective: To explore and optimize numerical calculation codes using GPU parallel processing, with a comparison against traditional CPU-based solutions. The project aimed to evaluate different strategies and techniques to enhance computational efficiency using GPUs over CPUs.

This `README.md` provides guidance on navigating the repository and understanding the content, even if you are new to parallel processing or GPU optimization.

<br>

### <ins>Requirements</ins>

To replicate or run the code in this project, ensure you have installed the necessary dependencies for each programming language:
- **Python**: Version 3.12.2 or newer, with `NumPy`, `CuPy`, and other related libraries.
- **Fortran**: With Intel MKL and OpenMP for parallel processing.
- **C**: Using BLIS/AOCL for optimized linear algebra operations.
- **Matlab**: For comparison benchmarks.

Specific instructions for setting up these environments can be found in their respective folders within this repository.

<br>

### <ins>Project Structure</ins>

The repository is organized as follows:
- **`/Python`**: Contains the benchmark codes and examples using Python with NumPy and CuPy for GPU parallel processing.
- **`/Fortran`**: Includes codes with Intel MKL and OpenMP for parallel computations.
- **`/C`**: Implementation using BLIS and AOCL libraries for optimized matrix multiplication.
- **`/Matlab`**: Matlab benchmarks and comparisons against GPU and CPU implementations.
- **`/Documentation`**: Detailed research findings, explanations of techniques, and performance comparisons.

Additionally:
- **`/Month_23`**: Contains my contributions from the 2022/23 academic year in chronological order.
- **`/Septiembre_24`**: Includes the work carried out by his colleagues in the 2023/24 academic year.

The second folder is still being compiled as some work by colleagues is not fully completed.

<br>

### <ins>Getting Started</ins>

Clone this repository by using the following command:
```bash
git clone https://github.com/jahrWork/GPU-Parallel.git
```

Refer to the documentation within each folder for setup instructions, or use Visual Studio Code, Spyder, or another IDE of your choice.

<br>

### <ins>Final Results & Analysis</ins>
The project includes a comprehensive analysis of:

Execution times across different languages and platforms (Python, Fortran, C, Matlab).

Speed-up comparisons between single-core and multi-core CPU implementations.

GPU performance evaluations using the CuPy library.



### <ins>Contributions</ins>
Contributions from all team members across various branches of the project are consolidated in this repository. Feel free to explore and learn from the diverse approaches taken.