#ifndef PRACTICO3_H__
#define PRACTICO3_H__

#include "utils.h"

typedef double value_type;

#define PI 3.14159265359
#define L  6.28318530718 // 2*PI

// coordinate grows from left-right and it's displaced by PI to the right
#define COORDVALX(coord, h)             (h * coord - PI)

// Y coordinate grows from top-down is displaced by PI to the bottom
#define COORDVALY(y, h)                 (PI - h * y)

// coordinate can be treated as the X coordinate for this matter
#define COORDVALZ COORDVALX

// Given a (x, y) coordinate in a 2D grid, return the corresponding index as if it were a 1D array
#define COORD2IDX(x, y, xdim) (y * xdim + x)

// Given a (x, y, z) coordinate in a 3D grid, return the corresponding index as if it were a 1D array
#define COORD3IDX(x, y, z, xdim, ydim) (z * ydim * xdim + y * xdim + x)

#ifndef BLOCK_SIZE
// Allowed to be overridden at compile time
#define BLOCK_SIZE                      (128u)
// note: 32 thread per block peformed bad, but more than 128 wasn't much better
// TODO: Show some numbers in the report
#endif

// testing numbers for exercises 1 and 2
constexpr unsigned int fibonacci_numbers[]
        = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
           233, 377, 610, 987, 1597, 2584, 4181};

__host__ __device__ inline value_type f1(value_type x, value_type y)
{
    return cos(x + y);
}

__host__ __device__ inline value_type f2(value_type x, value_type y, value_type z)
{
    return tan(x + y + z);
}

inline std::string val2string(value_type val, int width)
{
    char buffer[12] = {0};
    snprintf(buffer, sizeof(buffer), "% 5.2f", val);
    std::string str = buffer;
    std::string padding(width - str.size(), ' ');
    return padding + str;
}

#endif // PRACTICO3_H__
