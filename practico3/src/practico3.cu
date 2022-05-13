/*!
 * \file practico3.cu
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#include "practico3.h"

#include "ejercicio1.cuh"
#include "ejercicio2.cuh"
#include "ejercicio3.cuh"

int main(int argc, char *argv[]) {
    const char *part = nullptr;
    if (argc > 1 && strcmp(argv[1], "-h") == 0) {
        printf("Usage: %s [part]\n\t ; part = 1|2|3a|3b|3c|3cs|4\n", argv[0]);
        return 0;
    }
	part = argv[1];

    // GTX 1650 prints 7.5 and 1024
    print_device_capabilities();

#define SOLVE_EXERCISE(name, desc, code) do {\
    if (!part || strcmp(part, name) == 0) { \
        printf("---------- Begin %s:\n", desc); \
        code; \
        printf("---------- End %s ----------\n", desc); \
    } \
} while(false)

    SOLVE_EXERCISE("1", "Ejercicio 1", ejercicio1());
    SOLVE_EXERCISE("2", "Ejercicio 2", ejercicio2());
    SOLVE_EXERCISE("3a", "Ejercicio 3, parte a", ejercicio3a());
    SOLVE_EXERCISE("3b", "Ejercicio 3, parte b", ejercicio3b());
    SOLVE_EXERCISE("3c", "Ejercicio 3, parte c", ejercicio3c());
    SOLVE_EXERCISE("3cs", "Ejercicio 3, parte c (shared memory)", ejercicio3cs());


    return 0;
}
