/*!
 * \file practico4.cu
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

    std::srand(35141);

    // GTX 1650 prints 7.5 and 1024
    print_device_capabilities();

#if 0


    blMat A;

    gen_matriz_bloques (&A, 10, 10);

    print_matriz_bloques ( &A );
    print_matriz_bloques_en_COO ( &A );

    csrMat A_csr;

    bloques_a_CSR ( &A, &A_csr );
    //print_CSR ( &A_csr );
#endif

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
