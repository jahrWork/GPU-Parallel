# Recopilación

## 0. Migración a Julia

## 1. Comparación del dot product
En el código "1/matmul.jl" se compara la funcion de dotproduct de Julia con otras manuales. Se refleja la forma de acceso a memoria del lenguaje

## 2. Benchmark, tiempo teórico de CPU
En esta carpeta hay bastantes códigos, todos con el objetivo de hacer tender las pruebas a los valores teóricos. Hay cierta variabilidad de unos códigos a otros, pero poco a poco se va obteniendo el código que luego se recoge en el documento oficial (dot product vs theoretical time). También incluye comparación entre multicore y singlecore.

## 2.2 Ahora, benchmark en GPU
En esta parte tengo algunos códigos, pero no puedo comprobar si funcionan porque mi ordenador no tiene GPU dedicada. La mayoría de estos códigos los tienen Álvaro y Santiago, ellos te podrán pasar unos cuantos. En este tema básicamente hacemos lo mismo que para la CPU, con diferencia de que en vez de tener una expresión teórica tenemos unos valores teóricos de una página web, que tomamos como referencia.

## 3. Si en matriz x matriz funciona, en matriz x vector debería funcionar, ¿no?

Después de comprobar que se tiende a los valores teóricos esperados, se trata de hacer lo mismo con productos de matrices por vectores. Además de los códigos hay capturas, algunas que incluso comparan matmul con "matvecmul" y se ve como efectivamente hay una considerable diferencia en GFLOPS.

De nuevo, los códigos más limpios son los que están en la carpeta "code", dentro de la carpeta que corresponde al documento de *CPU performance*. La explicación de por qué no se llega al limite teórico multiplicando matriz x vector está tambien en el documento de *CPU performance*

## 4. Maquillar los resultados usando como problema a resolver Heat-Eq y Adv-Dif

Este apartado no introduce ningún resultado resaltable nuevo, solo presenta los resultados de una forma más visual y más atractiva, ya que deja ver que, tanto Matriz x Matriz como Matriz x Vector dan el mismo resultado, con la diferencia de que Matriz x Matriz lo hace 40 veces más rápido. En esta carpeta dejo bastantes códigos, algunas subcarpetas incluyen también capturas. Los códigos finales de esta carpeta deben estar en el documento de Santiago.

## Recomendación

En los documentos de cada uno están los códigos que hemos considerado más relevantes. Aquí incluyo todo lo que he encontrado, y mucho de ello no aporta más a los códigos que se incluyen en los documentos. Aún así pueden ayudar a comprender algo mejor la ruta y el porqué de las formas de alguno de los códigos, espero que te ayude!

