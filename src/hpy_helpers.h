/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#ifndef MPL_HPY_HELPERS_H
#define MPL_HPY_HELPERS_H

/*
 * This header contains helper macros for hpy
 */


#define Arg_ParseTuple(ret, ctx, tuple, fmt, ...)                                   \
    HPy_ssize_t tuple##_nargs = HPy_Length(ctx, tuple);                             \
    HPy *tuple##_args = (HPy *)malloc(tuple##_nargs * sizeof(HPy));                 \
    for (HPy_ssize_t tuple##_i = 0; tuple##_i < tuple##_nargs; ++tuple##_i) {       \
        tuple##_args[tuple##_i] = HPy_GetItem_i(ctx, tuple, tuple##_i);             \
    }                                                                               \
    ret = HPyArg_Parse(ctx, NULL, tuple##_args, tuple##_nargs, fmt, ##__VA_ARGS__);

#define Arg_ParseTupleClose(ctx, tuple)                                         \
    for (HPy_ssize_t tuple##_i = 0; tuple##_i < tuple##_nargs; ++tuple##_i) {   \
        HPy_Close(ctx, tuple##_args[tuple##_i]);                                \
    }                                                                           \
    free(tuple##_args);

#define Arg_ParseTupleAndClose(ret, ctx, tuple, fmt, ...)   \
    Arg_ParseTuple(ret, ctx, tuple, fmt, ##__VA_ARGS__)     \
    Arg_ParseTupleClose(ctx, tuple)

#endif
