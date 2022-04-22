/* -*- mode: c++; c-basic-offset: 4 -*- */
/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* Small utilities that are shared by most extension modules. */

#ifndef MPLUTILS_H
#define MPLUTILS_H
#define PY_SSIZE_T_CLEAN

#include <stdint.h>

#ifdef _POSIX_C_SOURCE
#    undef _POSIX_C_SOURCE
#endif
#ifndef _AIX
#ifdef _XOPEN_SOURCE
#    undef _XOPEN_SOURCE
#endif
#endif

// Prevent multiple conflicting definitions of swab from stdlib.h and unistd.h
#if defined(__sun) || defined(sun)
#if defined(_XPG4)
#undef _XPG4
#endif
#if defined(_XPG3)
#undef _XPG3
#endif
#endif

#ifdef HPY
#include "hpy.h"
#else
#include <Python.h>
#endif

inline double mpl_round(double v)
{
    return (double)(int)(v + ((v >= 0.0) ? 0.5 : -0.5));
}

// 'kind' codes for paths.
enum {
    STOP = 0,
    MOVETO = 1,
    LINETO = 2,
    CURVE3 = 3,
    CURVE4 = 4,
    CLOSEPOLY = 0x4f
};

const size_t NUM_VERTICES[] = { 1, 1, 1, 2, 3, 1 };

#endif
