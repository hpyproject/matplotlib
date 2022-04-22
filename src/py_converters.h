/* -*- mode: c++; c-basic-offset: 4 -*- */
/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

#ifndef MPL_PY_CONVERTERS_H
#define MPL_PY_CONVERTERS_H

/***************************************************************************
 * This module contains a number of conversion functions from Python types
 * to C++ types.  Most of them meet the Python "converter" signature:
 *
 *    typedef int (*converter_hpy)(HPyContext *, HPy, void *);
 *
 * and thus can be passed as conversion functions to PyArg_ParseTuple
 * and friends.
 */

#include "_backend_agg_basic_types.h"

extern "C" {

#ifdef HPY

#include "hpy.h"
#include "hpy_helpers.h"

typedef int (*converter_hpy)(HPyContext *, HPy, void *);

int convert_from_attr_hpy(HPyContext *ctx, HPy obj, const char *name, converter_hpy func, void *p);
int convert_from_method_hpy(HPyContext *ctx, HPy obj, const char *name, converter_hpy func, void *p);

int convert_double_hpy(HPyContext *ctx, HPy obj, void *p);
int convert_bool_hpy(HPyContext *ctx, HPy obj, void *p);
int convert_cap_hpy(HPyContext *ctx, HPy capobj, void *capp);
int convert_join_hpy(HPyContext *ctx, HPy joinobj, void *joinp);
int convert_rect_hpy(HPyContext *ctx, HPy rectobj, void *rectp);
int convert_rgba_hpy(HPyContext *ctx, HPy rgbaocj, void *rgbap);
int convert_dashes_hpy(HPyContext *ctx, HPy dashobj, void *gcp);
int convert_dashes_vector_hpy(HPyContext *ctx, HPy obj, void *dashesp);
int convert_trans_affine_hpy(HPyContext *ctx, HPy obj, void *transp);
int convert_path_hpy(HPyContext *ctx, HPy obj, void *pathp);
int convert_clippath_hpy(HPyContext *ctx, HPy clippath_tuple, void *clippathp);
int convert_snap_hpy(HPyContext *ctx, HPy obj, void *snapp);
int convert_offset_position_hpy(HPyContext *ctx, HPy obj, void *offsetp);
int convert_sketch_params_hpy(HPyContext *ctx, HPy obj, void *sketchp);
int convert_gcagg_hpy(HPyContext *ctx, HPy pygc, void *gcp);
int convert_points_hpy(HPyContext *ctx, HPy pygc, void *pointsp);
int convert_transforms_hpy(HPyContext *ctx, HPy pygc, void *transp);
int convert_bboxes_hpy(HPyContext *ctx, HPy pygc, void *bboxp);
int convert_colors_hpy(HPyContext *ctx, HPy pygc, void *colorsp);

int convert_face_hpy(HPyContext *ctx, HPy color, GCAgg &gc, agg::rgba *rgba);

#else

#include <Python.h>

typedef int (*converter)(PyObject *, void *);

int convert_from_attr(PyObject *obj, const char *name, converter func, void *p);
int convert_from_method(PyObject *obj, const char *name, converter func, void *p);

int convert_double(PyObject *obj, void *p);
int convert_bool(PyObject *obj, void *p);
int convert_cap(PyObject *capobj, void *capp);
int convert_join(PyObject *joinobj, void *joinp);
int convert_rect(PyObject *rectobj, void *rectp);
int convert_rgba(PyObject *rgbaocj, void *rgbap);
int convert_dashes(PyObject *dashobj, void *gcp);
int convert_dashes_vector(PyObject *obj, void *dashesp);
int convert_trans_affine(PyObject *obj, void *transp);
int convert_path(PyObject *obj, void *pathp);
int convert_clippath(PyObject *clippath_tuple, void *clippathp);
int convert_snap(PyObject *obj, void *snapp);
int convert_offset_position(PyObject *obj, void *offsetp);
int convert_sketch_params(PyObject *obj, void *sketchp);
int convert_gcagg(PyObject *pygc, void *gcp);
int convert_points(PyObject *pygc, void *pointsp);
int convert_transforms(PyObject *pygc, void *transp);
int convert_bboxes(PyObject *pygc, void *bboxp);
int convert_colors(PyObject *pygc, void *colorsp);

int convert_face(PyObject *color, GCAgg &gc, agg::rgba *rgba);

#endif
}

#endif
