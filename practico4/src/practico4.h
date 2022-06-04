/*!
 * \file practico4.h
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef PRACTICO4_H__
#define PRACTICO4_H__

#include "utils.h"

typedef double value_type;

#ifndef BLOCK_SIZE
// Allowed to be overridden at compile time
#define BLOCK_SIZE                      (256u)
// note: 32 thread per block peformed bad, but more than 128 wasn't much better
// TODO: Show some numbers in the report
#endif

#ifndef BENCH_TIMES
#define BENCH_TIMES 100
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
