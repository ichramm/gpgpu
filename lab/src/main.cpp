/*!
 * \file main.cpp
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#include "lab.hpp"

int main(int argc, char *argv[]) {
    const char *part = nullptr;

    if (argc > 1 && strcmp(argv[1], "-h") == 0) {
        printf("Usage: %s [part] [file.txt]\n\t ; part = 1|2|3|4\n"
                "\t file.txt es necesario para el ejercicio 1\n",
                argv[0]);
        return 0;
    }

    if (argc > 1) {
        part = argv[1];
    }

    std::srand(35141);

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
    SOLVE_EXERCISE("3", "Ejercicio 3", ejercicio3());

    return 0;
}
