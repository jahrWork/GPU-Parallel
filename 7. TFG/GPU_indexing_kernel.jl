using LinearAlgebra, CUDA
using CUDA.CUBLAS

include("system_info.jl")

# Este archivo es solo para entender cómo se indexa en un kernel de CUDA. No estoy seguro aún.
# Voy a empezar a hacer pruebas con caso mucho más sencillos. Pero aquí hay un buen resumen de los tipos de indexación. 

# Kernel que asigna elementos de R (2D) a C (3D), no es realmente útil, porque no estoy haciendo
# Nt veces la operacón R = A * B, si no que la hago una vez y la copio Nt veces en una tercera dimensión.
# Pero es un buen ejemplo para entender cómo manejar la indexación. He estado todo el día con ello y como
# decía Javier, mejor empezar por algo más sencillo. He intentado explicarlo y he dejado alguna imagen
# o explicación para entender cómo se calculan algunos índices.

# Cada thread asignará un elemento de R en la correspondiente posición 3D de C.
function kernel(R, C)

    # Cálculo del índice global lineal del thread.
    idx = (blockIdx().x - 1)*blockDim().x + threadIdx().x 
    # Ver array_thread_block.png para entender cómo se calcula idx
    # En CUDA, un grid (ver grid.png) se organiza en blocks y cada block en threads.
    # Normalmente, el índice global se calcula como:
    # global_idx = blockIdx.x * blockDim.x + threadIdx.x (pero esto es en zero-based indexing)
    # Restamos 1 a blockIdx().x para usar indexación base-1 de Julia.

    # Si esta C (3D) tiene dimensiones m x p x Nt, entonces:
    total_elems = size(C,1)*size(C,2)*size(C,3)
    # total_elems = m * p * Nt

    # Verificamos si este hilo corresponde a un índice dentro del rango.
    # Puede suceder que algunos hilos no tengan trabajo si el total de hilos lanzados
    # es mayor que total_elems.
    if idx <= total_elems
        m = size(C,1) # Número de filas
        p = size(C,2) # Número de columnas

        # Convertimos el índice lineal (idx) en índices 3D (row, col, k).
        # Como la indexación en Julia empieza en 1, primero convertimos a base 0:
        # idx - 1 se utiliza para facilitar el uso de modulo/rem (%) y división entera (÷).
        # No lo conocía, dejo link de la explicación de qué es cada cosa exactamente.
        # https://es.mathworks.com/matlabcentral/answers/403870-difference-between-mod-and-rem-functions
        
        # Se puede ver cómo sería en zero-based indexing en:
        # https://stackoverflow.com/questions/11821899/how-to-get-row-and-column-from-index
        # row: Cada m elementos cambiamos de columna.
        # (idx - 1) % m da un valor entre 0 y m-1 que indica la fila.
        # Sumamos 1 para volver a base-1.
        row = (idx - 1) % m + 1

        # col: Cada m elementos completamos una columna y pasamos a la siguiente.
        # ((idx - 1) ÷ m) nos dice cuántas columnas completas hemos "saltado".
        # Luego, tomamos ese valor modulo p para quedar en el rango [0, p-1].
        # Sumamos 1 para volver a base-1.
        col = ((idx - 1) ÷ m) % p + 1

        # k: Cada bloque de m*p elementos corresponde a una "capa" en la tercera dimensión.
        # ((idx - 1) ÷ (m*p)) nos dice cuántas "capas" hemos pasado.
        # Sumamos 1 para volver a base-1.
        k   = ((idx - 1) ÷ (m*p)) + 1

        # Asignamos el valor correspondiente de R a la posición 3D de C.
        C[row, col, k] = R[row, col]
    end
    return nothing
end