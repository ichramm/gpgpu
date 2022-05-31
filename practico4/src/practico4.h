/*!
 * \file practico4.h
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef PRACTICO4_H__
#define PRACTICO4_H__

#include "utils.h"

typedef double value_type;

#define PI 3.14159265359
#define L  6.28318530718 // 2*PI

// coordinate grows from left to right (xval = -PI when xcoord = 0)
#define COORDVALX(xcoord, h)             (h * xcoord - PI)

// Y coordinate grows from top to bottom (yval = PI when ycoord = 0)
#define COORDVALY(ycoord, h)             (PI - h * ycoord)

// coordinate can be treated as the X coordinate for this matter
#define COORDVALZ COORDVALX

// Given a (x, y) coordinate in a 2D grid, return the corresponding index as if it were a 1D array
#define COORD2IDX(x, y, xdim) (y * xdim + x)

// Given a (x, y, z) coordinate in a 3D grid, return the corresponding index as if it were a 1D array
#define COORD3IDX(x, y, z, xdim, ydim) (z * ydim * xdim + y * xdim + x)

#ifndef BLOCK_SIZE
// Allowed to be overridden at compile time
#define BLOCK_SIZE                      (256u)
// note: 32 thread per block peformed bad, but more than 128 wasn't much better
// TODO: Show some numbers in the report
#endif

inline std::string val2string(value_type val, const char *format, int width)
{
    char buffer[12] = {0};
    snprintf(buffer, sizeof(buffer), format, val);
    std::string str = buffer;
    std::string padding(width - str.size(), ' ');
    return padding + str;
}

#endif // PRACTICO4_H__
