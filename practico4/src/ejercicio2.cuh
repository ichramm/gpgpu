/*!
 * \file ejercicio2.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO2_CUH__
#define EJERCICIO2_CUH__

#include "practico4.h"

#if 0
a) Extienda el ejercicio 4 del práctico 3 para que calcule el siguiente stencil:
    Doout[x,y] = ( (-8)Din[x,y] + \
                   Din[x+1,y] + Din[x+2,y] + Din[x-1,y] + Din[x-2,y] +\
                   Din[x,y+1] + Din[x,y+2] + Din[x,y-1] + Din[x,y-2] ) / h^2

b) Utilice la memoria compartida para reutilizar los datos cargados por hilos vecinos. La configuración de
la grilla de hilos debe asociar un hilo a cada elemento de la grilla (matrices din y dout). La región de
memoria compartida a usar debe contemplar todos los elementos accedidos por cada bloque. La carga de
la memoria compartida debe realizarse con el mayor paralelismo posible.

c) Compare el desempeño de los kernels correspondientes a las partes a) y b) para grillas de tamaño 4096 2
y 81922. No es necesario reservar memoria en CPU ni transferir las matrices. Pueden generarse en GPU
mediante un kernel reservando previamente la memoria necesaria con cudaMalloc.
#endif



void ejercicio2() {

}

#endif // EJERCICIO2_CUH__
