# GPGPU - Práctico 2

> Juan Ramírez - 4.863.696-4

## Compilar y ejecutar local

Asumiendo que cuda está instalado en la máquina local, basta ejecutar `make`:

```sh
make
```

Se puede usar la variable `NVCCFLAGS` para pasar algún parámetro al compilador, por ejemplo:

```sh
NVCCFLAGS="-ccbin clang-8" make
```

Para ejecutar se puede invocar directamente el binario o utilizar `make`:

```sh
$ ./build/bin/practico2 -h
Usage: ./practico2 secret.txt [part]
         ; part = 1a|1b|1c|2
```

```sh
# ejecuta todos los ejercicios
make run ARGS="secreto.txt"
```

También es posible especificar el ejercicio a ejecutar:

```sh
# ejecuta la parte 1c
make run ARGS="secreto.txt 1c"
```

El comportamiento por defecto es ejecutar todos los ejercicios.

## Compilar y ejecutar en ClusterUY

Para compilar y ejecutar en ClusterUY basta ejecutar el archivo `launch_single.sh` y pasarle solamente un parámetro (opcional) para indicar qué parte ejecutar:

```sh
# ejecuta la parte 1a
sbatch launch_single.sh 1a
```

Nota: El script ya pasa el primer parámetro: `secreto.txt`.
