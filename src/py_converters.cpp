/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#define NO_IMPORT_ARRAY
#define PY_SSIZE_T_CLEAN
#include "py_converters.h"
#include "numpy_cpp.h"

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_math_stroke.h"

extern "C" {

#ifdef HPY

static int convert_string_enum_hpy(HPyContext *ctx, HPy obj, const char *name, const char **names, int *values, int *result)
{
    HPy bytesobj;
    char *str;

    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        return 1;
    }

    if (HPyUnicode_Check(ctx, obj)) {
        bytesobj = HPyUnicode_AsASCIIString(ctx, obj);
        if (HPy_IsNull(bytesobj)) {
            return 0;
        }
    } else if (HPyBytes_Check(ctx, obj)) {
        bytesobj = HPy_Dup(ctx, obj);
    } else {
        // PyErr_Format(PyExc_TypeError, "%s must be str or bytes", name); TODO: implement HPyErr_Format
        HPyErr_SetString(ctx, ctx->h_TypeError, "must be str or bytes");
        return 0;
    }

    str = HPyBytes_AsString(ctx, bytesobj);
    if (str == NULL) {
        HPy_Close(ctx, bytesobj);
        return 0;
    }

    for ( ; *names != NULL; names++, values++) {
        if (strncmp(str, *names, 64) == 0) {
            *result = *values;
            HPy_Close(ctx, bytesobj);
            return 1;
        }
    }

    // PyErr_Format(PyExc_ValueError, "invalid %s value", name); TODO: implement HPyErr_Format
    HPyErr_SetString(ctx, ctx->h_ValueError, "invalid value");
    HPy_Close(ctx, bytesobj);
    return 0;
}

int convert_from_method_hpy(HPyContext *ctx, HPy obj, const char *name, converter_hpy func, void *p)
{
    if (!HPy_HasAttr_s(ctx, obj, name)) {
        return 1;
    }
    HPy callable = HPy_GetAttr_s(ctx, obj, name);
    HPy value = HPy_CallTupleDict(ctx, callable, HPy_NULL, HPy_NULL);
    HPy_Close(ctx, callable);
    // value = PyObject_CallMethod(obj, name, NULL);
    if (HPy_IsNull(value)) {
        return 0;
    }

    if (!func(ctx, value, p)) {
        HPy_Close(ctx, value);
        return 0;
    }

    HPy_Close(ctx, value);
    return 1;
}

int convert_from_attr_hpy(HPyContext *ctx, HPy obj, const char *name, converter_hpy func, void *p)
{
    HPy value = HPy_GetAttr_s(ctx, obj, name);
    if (HPy_IsNull(value)) {
        if (!HPy_HasAttr_s(ctx, obj, name)) {
            HPyErr_Clear(ctx);
            return 1;
        }
        return 0;
    }

    if (!func(ctx, value, p)) {
        HPy_Close(ctx, value);
        return 0;
    }

    HPy_Close(ctx, value);
    return 1;
}

int convert_double_hpy(HPyContext *ctx, HPy obj, void *p)
{
    double *val = (double *)p;

    *val = HPyFloat_AsDouble(ctx, obj);
    if (HPyErr_Occurred(ctx)) {
        return 0;
    }

    return 1;
}

int convert_bool_hpy(HPyContext *ctx, HPy obj, void *p)
{
    bool *val = (bool *)p;
    switch (HPy_IsTrue(ctx, obj)) {
        case 0: *val = false; break;
        case 1: *val = true; break;
        default: return 0;  // errored.
    }
    return 1;
}

int convert_cap_hpy(HPyContext *ctx, HPy capobj, void *capp)
{
    const char *names[] = {"butt", "round", "projecting", NULL};
    int values[] = {agg::butt_cap, agg::round_cap, agg::square_cap};
    int result = agg::butt_cap;

    if (!convert_string_enum_hpy(ctx, capobj, "capstyle", names, values, &result)) {
        return 0;
    }

    *(agg::line_cap_e *)capp = (agg::line_cap_e)result;
    return 1;
}

int convert_join_hpy(HPyContext *ctx, HPy joinobj, void *joinp)
{
    const char *names[] = {"miter", "round", "bevel", NULL};
    int values[] = {agg::miter_join_revert, agg::round_join, agg::bevel_join};
    int result = agg::miter_join_revert;

    if (!convert_string_enum_hpy(ctx, joinobj, "joinstyle", names, values, &result)) {
        return 0;
    }

    *(agg::line_join_e *)joinp = (agg::line_join_e)result;
    return 1;
}

int convert_rect_hpy(HPyContext *ctx, HPy rectobj, void *rectp)
{
    agg::rect_d *rect = (agg::rect_d *)rectp;

    if (HPy_IsNull(rectobj) || HPy_Is(ctx, rectobj, ctx->h_None)) {
        rect->x1 = 0.0;
        rect->y1 = 0.0;
        rect->x2 = 0.0;
        rect->y2 = 0.0;
    } else {
        PyArrayObject *rect_arr = (PyArrayObject *)PyArray_ContiguousFromAny(
                HPy_AsPyObject(ctx, rectobj), NPY_DOUBLE, 1, 2);
        if (rect_arr == NULL) {
            return 0;
        }

        if (PyArray_NDIM(rect_arr) == 2) {
            if (PyArray_DIM(rect_arr, 0) != 2 ||
                PyArray_DIM(rect_arr, 1) != 2) {
                HPyErr_SetString(ctx, ctx->h_ValueError, "Invalid bounding box");
                Py_DECREF(rect_arr);
                return 0;
            }

        } else {  // PyArray_NDIM(rect_arr) == 1
            if (PyArray_DIM(rect_arr, 0) != 4) {
                HPyErr_SetString(ctx, ctx->h_ValueError, "Invalid bounding box");
                Py_DECREF(rect_arr);
                return 0;
            }
        }

        double *buff = (double *)PyArray_DATA(rect_arr);
        rect->x1 = buff[0];
        rect->y1 = buff[1];
        rect->x2 = buff[2];
        rect->y2 = buff[3];

        Py_DECREF(rect_arr);
    }
    return 1;
}

int convert_rgba_hpy(HPyContext *ctx, HPy rgbaobj, void *rgbap)
{
    agg::rgba *rgba = (agg::rgba *)rgbap;

    if (HPy_IsNull(rgbaobj) || HPy_Is(ctx, rgbaobj, ctx->h_None)) {
        rgba->r = 0.0;
        rgba->g = 0.0;
        rgba->b = 0.0;
        rgba->a = 0.0;
    } else {
        rgba->a = 1.0;
        HPy_ssize_t nargs = HPy_Length(ctx, rgbaobj);
        HPy args[nargs];
        for (HPy_ssize_t i = 0; i < nargs; i++) {
            args[i] = HPy_GetItem_i(ctx, rgbaobj, i);
        }
        int res = HPyArg_Parse(ctx, NULL, args, nargs,
                 "ddd|d:rgba", &(rgba->r), &(rgba->g), &(rgba->b), &(rgba->a));                
        if (!res) {
            return 0;
        }
    }

    return 1;
}

int convert_dashes_hpy(HPyContext *ctx, HPy dashobj, void *dashesp)
{
    if (HPy_IsNull(dashobj) || HPy_Is(ctx, dashobj, ctx->h_None)) {
        return 1;
    }

    Dashes *dashes = (Dashes *)dashesp;

    double dash_offset = 0.0;
    HPy dashes_seq = HPy_NULL;

    int ret;
    Arg_ParseTuple(ret, ctx, dashobj, "dO:dashes", &dash_offset, &dashes_seq)
    dashes_seq = HPy_Dup(ctx, dashes_seq); // copy before closing tuple items
    Arg_ParseTupleClose(ctx, dashobj);

    if (!ret) {
        return 0;
    }

    if (HPy_Is(ctx, dashes_seq, ctx->h_None)) {
        HPy_Close(ctx, dashes_seq);
        return 1;
    }

    // if (!HPySequence_Check(ctx, dashes_seq)) { TODO: HPySequence_Check
    if (!HPyList_Check(ctx, dashes_seq) && !HPyTuple_Check(ctx, dashes_seq)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "Invalid dashes sequence");
        HPy_Close(ctx, dashes_seq);
        return 0;
    }

    HPy_ssize_t nentries = HPy_Length(ctx, dashes_seq);
    // If the dashpattern has odd length, iterate through it twice (in
    // accordance with the pdf/ps/svg specs).
    HPy_ssize_t dash_pattern_length = (nentries % 2) ? 2 * nentries : nentries;

    for (HPy_ssize_t i = 0; i < dash_pattern_length; ++i) {
        double length;
        double skip;

        HPy item = HPy_GetItem_i(ctx, dashes_seq, i % nentries);
        if (HPy_IsNull(item)) {
            HPy_Close(ctx, dashes_seq);
            return 0;
        }
        length = HPyFloat_AsDouble(ctx, item);
        if (HPyErr_Occurred(ctx)) {
            HPy_Close(ctx, item);
            HPy_Close(ctx, dashes_seq);
            return 0;
        }
        HPy_Close(ctx, item);

        ++i;

        item = HPy_GetItem_i(ctx, dashes_seq, i % nentries);
        if (HPy_IsNull(item)) {
            HPy_Close(ctx, dashes_seq);
            return 0;
        }
        skip = HPyFloat_AsDouble(ctx, item);
        if (HPyErr_Occurred(ctx)) {
            HPy_Close(ctx, item);
            HPy_Close(ctx, dashes_seq);
            return 0;
        }
        HPy_Close(ctx, item);

        dashes->add_dash_pair(length, skip);
    }
    HPy_Close(ctx, dashes_seq);

    dashes->set_dash_offset(dash_offset);

    return 1;
}

int convert_dashes_vector_hpy(HPyContext *ctx, HPy obj, void *dashesp)
{
    DashesVector *dashes = (DashesVector *)dashesp;

    // if (!HPySequence_Check(ctx, obj)) { TODO: HPySequence_Check
    if (!HPyList_Check(ctx, obj) && !HPyTuple_Check(ctx, obj)) {
        return 0;
    }

    HPy_ssize_t n = HPy_Length(ctx, obj);

    for (HPy_ssize_t i = 0; i < n; ++i) {
        Dashes subdashes;

        HPy item = HPy_GetItem_i(ctx, obj, i);
        if (HPy_IsNull(item)) {
            return 0;
        }

        if (!convert_dashes_hpy(ctx, item, &subdashes)) {
            HPy_Close(ctx, item);
            return 0;
        }
        HPy_Close(ctx, item);

        dashes->push_back(subdashes);
    }

    return 1;
}

int convert_trans_affine_hpy(HPyContext *ctx, HPy obj, void *transp)
{
    agg::trans_affine *trans = (agg::trans_affine *)transp;

    /** If None assume identity transform. */
    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        return 1;
    }

    PyArrayObject *array = (PyArrayObject *)PyArray_ContiguousFromAny(HPy_AsPyObject(ctx, obj), NPY_DOUBLE, 2, 2);
    if (array == NULL) {
        return 0;
    }

    if (PyArray_DIM(array, 0) == 3 && PyArray_DIM(array, 1) == 3) {
        double *buffer = (double *)PyArray_DATA(array);
        trans->sx = buffer[0];
        trans->shx = buffer[1];
        trans->tx = buffer[2];

        trans->shy = buffer[3];
        trans->sy = buffer[4];
        trans->ty = buffer[5];

        Py_DECREF(array);
        return 1;
    }

    Py_DECREF(array);
    HPyErr_SetString(ctx, ctx->h_ValueError, "Invalid affine transformation matrix");
    return 0;
}


int convert_path_hpy(HPyContext *ctx, HPy obj, void *pathp)
{
    py::PathIterator *path = (py::PathIterator *)pathp;

    HPy vertices_obj = HPy_NULL;
    HPy codes_obj = HPy_NULL;
    HPy should_simplify_obj = HPy_NULL;
    HPy simplify_threshold_obj = HPy_NULL;
    bool should_simplify;
    double simplify_threshold;

    int status = 0;

    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        return 1;
    }

    vertices_obj = HPy_GetAttr_s(ctx, obj, "vertices");
    if (HPy_IsNull(vertices_obj)) {
        goto exit;
    }

    codes_obj = HPy_GetAttr_s(ctx, obj, "codes");
    if (HPy_IsNull(codes_obj)) {
        goto exit;
    }

    should_simplify_obj = HPy_GetAttr_s(ctx, obj, "should_simplify");
    if (HPy_IsNull(should_simplify_obj)) {
        goto exit;
    }
    switch (HPy_IsTrue(ctx, should_simplify_obj)) {
        case 0: should_simplify = 0; break;
        case 1: should_simplify = 1; break;
        default: goto exit;  // errored.
    }

    simplify_threshold_obj = HPy_GetAttr_s(ctx, obj, "simplify_threshold");
    if (HPy_IsNull(simplify_threshold_obj)) {
        goto exit;
    }
    simplify_threshold = HPyFloat_AsDouble(ctx, simplify_threshold_obj);
    if (HPyErr_Occurred(ctx)) {
        goto exit;
    }

    if (!path->set(
            HPy_AsPyObject(ctx, vertices_obj), // PyArrayObject (NumPy)
            HPy_AsPyObject(ctx, codes_obj), // PyArrayObject (NumPy)
            should_simplify, simplify_threshold)) {
        goto exit;
    }

    status = 1;

exit:
    HPy_Close(ctx, vertices_obj);
    HPy_Close(ctx, codes_obj);
    HPy_Close(ctx, should_simplify_obj);
    HPy_Close(ctx, simplify_threshold_obj);

    return status;
}

int convert_clippath_hpy(HPyContext *ctx, HPy clippath_tuple, void *clippathp)
{
    HPy h_path = HPy_NULL, h_trans = HPy_NULL;
    ClipPath *clippath = (ClipPath *)clippathp;
    py::PathIterator path;
    agg::trans_affine trans;

    int res = 1;
    if (!HPy_IsNull(clippath_tuple) && !HPy_Is(ctx, clippath_tuple, ctx->h_None)) {
        Arg_ParseTuple(res, ctx, clippath_tuple, "OO:clippath",
                              &h_path,
                              &h_trans);
        if (res && (!convert_path_hpy(ctx, h_path, &clippath->path)
                    || !convert_trans_affine_hpy(ctx, h_trans, &clippath->trans))) {
            if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "clippath"); // TODO
            res = 0;
        }
        Arg_ParseTupleClose(ctx, clippath_tuple)
    }

    return res;
}

int convert_snap_hpy(HPyContext *ctx, HPy obj, void *snapp)
{
    e_snap_mode *snap = (e_snap_mode *)snapp;
    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        *snap = SNAP_AUTO;
    } else {
        switch (HPy_IsTrue(ctx, obj)) {
            case 0: *snap = SNAP_FALSE; break;
            case 1: *snap = SNAP_TRUE; break;
            default: return 0;  // errored.
        }
    }
    return 1;
}

int convert_sketch_params_hpy(HPyContext *ctx, HPy obj, void *sketchp)
{
    SketchParams *sketch = (SketchParams *)sketchp;

    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        sketch->scale = 0.0;
    } else {
        int res;
        Arg_ParseTupleAndClose(res, ctx, obj, "ddd:sketch_params",
                                 &sketch->scale,
                                 &sketch->length,
                                 &sketch->randomness);
        if (!res) {
            return 0;
        }
    }

    return 1;
}

int convert_gcagg_hpy(HPyContext *ctx, HPy pygc, void *gcp)
{
    GCAgg *gc = (GCAgg *)gcp;

    if (!(convert_from_attr_hpy(ctx, pygc, "_linewidth", &convert_double_hpy, &gc->linewidth) &&
          convert_from_attr_hpy(ctx, pygc, "_alpha", &convert_double_hpy, &gc->alpha) &&
          convert_from_attr_hpy(ctx, pygc, "_forced_alpha", &convert_bool_hpy, &gc->forced_alpha) &&
          convert_from_attr_hpy(ctx, pygc, "_rgb", &convert_rgba_hpy, &gc->color) &&
          convert_from_attr_hpy(ctx, pygc, "_antialiased", &convert_bool_hpy, &gc->isaa) &&
          convert_from_attr_hpy(ctx, pygc, "_capstyle", &convert_cap_hpy, &gc->cap) &&
          convert_from_attr_hpy(ctx, pygc, "_joinstyle", &convert_join_hpy, &gc->join) &&
          convert_from_method_hpy(ctx, pygc, "get_dashes", &convert_dashes_hpy, &gc->dashes) &&
          convert_from_attr_hpy(ctx, pygc, "_cliprect", &convert_rect_hpy, &gc->cliprect) &&
          convert_from_method_hpy(ctx, pygc, "get_clip_path", &convert_clippath_hpy, &gc->clippath) &&
          convert_from_method_hpy(ctx, pygc, "get_snap", &convert_snap_hpy, &gc->snap_mode) &&
          convert_from_method_hpy(ctx, pygc, "get_hatch_path", &convert_path_hpy, &gc->hatchpath) &&
          convert_from_method_hpy(ctx, pygc, "get_hatch_color", &convert_rgba_hpy, &gc->hatch_color) &&
          convert_from_method_hpy(ctx, pygc, "get_hatch_linewidth", &convert_double_hpy, &gc->hatch_linewidth) &&
          convert_from_method_hpy(ctx, pygc, "get_sketch_params", &convert_sketch_params_hpy, &gc->sketch))) {
        return 0;
    }

    return 1;
}

int convert_offset_position_hpy(HPyContext *ctx, HPy obj, void *offsetp)
{
    e_offset_position *offset = (e_offset_position *)offsetp;
    const char *names[] = {"data", NULL};
    int values[] = {OFFSET_POSITION_DATA};
    int result = (int)OFFSET_POSITION_FIGURE;

    if (!convert_string_enum_hpy(ctx, obj, "offset_position", names, values, &result)) {
        HPyErr_Clear(ctx);
    }

    *offset = (e_offset_position)result;

    return 1;
}

int convert_face_hpy(HPyContext *ctx, HPy color, GCAgg &gc, agg::rgba *rgba)
{
    if (!convert_rgba_hpy(ctx, color, rgba)) {
        return 0;
    }

    if (!HPy_IsNull(color) && !HPy_Is(ctx, color, ctx->h_None)) {
        if (gc.forced_alpha || HPy_Length(ctx, color) == 3) {
            rgba->a = gc.alpha;
        }
    }

    return 1;
}

int convert_points_hpy(HPyContext *ctx, HPy obj, void *pointsp)
{
    numpy::array_view<double, 2> *points = (numpy::array_view<double, 2> *)pointsp;

    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        return 1;
    }

    points->set(HPy_AsPyObject(ctx, obj));

    if (points->size() == 0) {
        return 1;
    }

    if (points->dim(1) != 2) {
        // PyErr_Format(PyExc_ValueError,
        //              "Points must be Nx2 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT,
        //              points->dim(0), points->dim(1)); TODO: implement HPyErr_Format
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "Points must be Nx2 array");
        return 0;
    }

    return 1;
}

int convert_transforms_hpy(HPyContext *ctx, HPy obj, void *transp)
{
    numpy::array_view<double, 3> *trans = (numpy::array_view<double, 3> *)transp;

    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        return 1;
    }

    trans->set(HPy_AsPyObject(ctx, obj));

    if (trans->size() == 0) {
        return 1;
    }

    if (trans->dim(1) != 3 || trans->dim(2) != 3) {
        // PyErr_Format(PyExc_ValueError,
        //              "Transforms must be Nx3x3 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT "x%" NPY_INTP_FMT,
        //              trans->dim(0), trans->dim(1), trans->dim(2)); TODO: implement HPyErr_Format
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "Transforms must be Nx3x3 array");
        return 0;
    }

    return 1;
}

int convert_bboxes_hpy(HPyContext *ctx, HPy obj, void *bboxp)
{
    numpy::array_view<double, 3> *bbox = (numpy::array_view<double, 3> *)bboxp;

    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        return 1;
    }

    bbox->set(HPy_AsPyObject(ctx, obj));

    if (bbox->size() == 0) {
        return 1;
    }

    if (bbox->dim(1) != 2 || bbox->dim(2) != 2) {
        // PyErr_Format(PyExc_ValueError,
        //              "Bbox array must be Nx2x2 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT "x%" NPY_INTP_FMT,
        //              bbox->dim(0), bbox->dim(1), bbox->dim(2)); TODO: implement HPyErr_Format
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "Bbox array must be Nx2x2 array");
        return 0;
    }

    return 1;
}

int convert_colors_hpy(HPyContext *ctx, HPy obj, void *colorsp)
{
    numpy::array_view<double, 2> *colors = (numpy::array_view<double, 2> *)colorsp;

    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        return 1;
    }

    colors->set(HPy_AsPyObject(ctx, obj));

    if (colors->size() == 0) {
        return 1;
    }

    if (colors->dim(1) != 4) {
        // PyErr_Format(PyExc_ValueError,
        //              "Colors array must be Nx4 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT,
        //              colors->dim(0), colors->dim(1)); TODO: implement HPyErr_Format
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "Colors array must be Nx4 array");
        return 0;
    }

    return 1;
}

#else

static int convert_string_enum(PyObject *obj, const char *name, const char **names, int *values, int *result)
{
    PyObject *bytesobj;
    char *str;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    if (PyUnicode_Check(obj)) {
        bytesobj = PyUnicode_AsASCIIString(obj);
        if (bytesobj == NULL) {
            return 0;
        }
    } else if (PyBytes_Check(obj)) {
        Py_INCREF(obj);
        bytesobj = obj;
    } else {
        PyErr_Format(PyExc_TypeError, "%s must be str or bytes", name);
        return 0;
    }

    str = PyBytes_AsString(bytesobj);
    if (str == NULL) {
        Py_DECREF(bytesobj);
        return 0;
    }

    for ( ; *names != NULL; names++, values++) {
        if (strncmp(str, *names, 64) == 0) {
            *result = *values;
            Py_DECREF(bytesobj);
            return 1;
        }
    }

    PyErr_Format(PyExc_ValueError, "invalid %s value", name);
    Py_DECREF(bytesobj);
    return 0;
}

int convert_from_method(PyObject *obj, const char *name, converter func, void *p)
{
    PyObject *value;

    value = PyObject_CallMethod(obj, name, NULL);
    if (value == NULL) {
        if (!PyObject_HasAttrString(obj, name)) {
            PyErr_Clear();
            return 1;
        }
        return 0;
    }

    if (!func(value, p)) {
        Py_DECREF(value);
        return 0;
    }

    Py_DECREF(value);
    return 1;
}

int convert_from_attr(PyObject *obj, const char *name, converter func, void *p)
{
    PyObject *value;

    value = PyObject_GetAttrString(obj, name);
    if (value == NULL) {
        if (!PyObject_HasAttrString(obj, name)) {
            PyErr_Clear();
            return 1;
        }
        return 0;
    }

    if (!func(value, p)) {
        Py_DECREF(value);
        return 0;
    }

    Py_DECREF(value);
    return 1;
}

int convert_double(PyObject *obj, void *p)
{
    double *val = (double *)p;

    *val = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        return 0;
    }

    return 1;
}

int convert_bool(PyObject *obj, void *p)
{
    bool *val = (bool *)p;
    switch (PyObject_IsTrue(obj)) {
        case 0: *val = false; break;
        case 1: *val = true; break;
        default: return 0;  // errored.
    }
    return 1;
}

int convert_cap(PyObject *capobj, void *capp)
{
    const char *names[] = {"butt", "round", "projecting", NULL};
    int values[] = {agg::butt_cap, agg::round_cap, agg::square_cap};
    int result = agg::butt_cap;

    if (!convert_string_enum(capobj, "capstyle", names, values, &result)) {
        return 0;
    }

    *(agg::line_cap_e *)capp = (agg::line_cap_e)result;
    return 1;
}

int convert_join(PyObject *joinobj, void *joinp)
{
    const char *names[] = {"miter", "round", "bevel", NULL};
    int values[] = {agg::miter_join_revert, agg::round_join, agg::bevel_join};
    int result = agg::miter_join_revert;

    if (!convert_string_enum(joinobj, "joinstyle", names, values, &result)) {
        return 0;
    }

    *(agg::line_join_e *)joinp = (agg::line_join_e)result;
    return 1;
}

int convert_rect(PyObject *rectobj, void *rectp)
{
    agg::rect_d *rect = (agg::rect_d *)rectp;

    if (rectobj == NULL || rectobj == Py_None) {
        rect->x1 = 0.0;
        rect->y1 = 0.0;
        rect->x2 = 0.0;
        rect->y2 = 0.0;
    } else {
        PyArrayObject *rect_arr = (PyArrayObject *)PyArray_ContiguousFromAny(
                rectobj, NPY_DOUBLE, 1, 2);
        if (rect_arr == NULL) {
            return 0;
        }

        if (PyArray_NDIM(rect_arr) == 2) {
            if (PyArray_DIM(rect_arr, 0) != 2 ||
                PyArray_DIM(rect_arr, 1) != 2) {
                PyErr_SetString(PyExc_ValueError, "Invalid bounding box");
                Py_DECREF(rect_arr);
                return 0;
            }

        } else {  // PyArray_NDIM(rect_arr) == 1
            if (PyArray_DIM(rect_arr, 0) != 4) {
                PyErr_SetString(PyExc_ValueError, "Invalid bounding box");
                Py_DECREF(rect_arr);
                return 0;
            }
        }

        double *buff = (double *)PyArray_DATA(rect_arr);
        rect->x1 = buff[0];
        rect->y1 = buff[1];
        rect->x2 = buff[2];
        rect->y2 = buff[3];

        Py_DECREF(rect_arr);
    }
    return 1;
}

int convert_rgba(PyObject *rgbaobj, void *rgbap)
{
    agg::rgba *rgba = (agg::rgba *)rgbap;

    if (rgbaobj == NULL || rgbaobj == Py_None) {
        rgba->r = 0.0;
        rgba->g = 0.0;
        rgba->b = 0.0;
        rgba->a = 0.0;
    } else {
        rgba->a = 1.0;
        if (!PyArg_ParseTuple(
                 rgbaobj, "ddd|d:rgba", &(rgba->r), &(rgba->g), &(rgba->b), &(rgba->a))) {
            return 0;
        }
    }

    return 1;
}

int convert_dashes(PyObject *dashobj, void *dashesp)
{
    Dashes *dashes = (Dashes *)dashesp;

    if (dashobj == NULL && dashobj == Py_None) {
        return 1;
    }

    PyObject *dash_offset_obj = NULL;
    double dash_offset = 0.0;
    PyObject *dashes_seq = NULL;

    if (!PyArg_ParseTuple(dashobj, "OO:dashes", &dash_offset_obj, &dashes_seq)) {
        return 0;
    }

    if (dash_offset_obj != Py_None) {
        dash_offset = PyFloat_AsDouble(dash_offset_obj);
        if (PyErr_Occurred()) {
            return 0;
        }
    } else {
        if (PyErr_WarnEx(PyExc_FutureWarning,
                         "Passing the dash offset as None is deprecated since "
                         "Matplotlib 3.3 and will be removed in Matplotlib 3.5; "
                         "pass it as zero instead.",
                         1)) {
            return 0;
        }
    }

    if (dashes_seq == Py_None) {
        return 1;
    }

    if (!PySequence_Check(dashes_seq)) {
        PyErr_SetString(PyExc_TypeError, "Invalid dashes sequence");
        return 0;
    }

    Py_ssize_t nentries = PySequence_Size(dashes_seq);
    // If the dashpattern has odd length, iterate through it twice (in
    // accordance with the pdf/ps/svg specs).
    Py_ssize_t dash_pattern_length = (nentries % 2) ? 2 * nentries : nentries;

    for (Py_ssize_t i = 0; i < dash_pattern_length; ++i) {
        PyObject *item;
        double length;
        double skip;

        item = PySequence_GetItem(dashes_seq, i % nentries);
        if (item == NULL) {
            return 0;
        }
        length = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(item);
            return 0;
        }
        Py_DECREF(item);

        ++i;

        item = PySequence_GetItem(dashes_seq, i % nentries);
        if (item == NULL) {
            return 0;
        }
        skip = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(item);
            return 0;
        }
        Py_DECREF(item);

        dashes->add_dash_pair(length, skip);
    }

    dashes->set_dash_offset(dash_offset);

    return 1;
}

int convert_dashes_vector(PyObject *obj, void *dashesp)
{
    DashesVector *dashes = (DashesVector *)dashesp;

    if (!PySequence_Check(obj)) {
        return 0;
    }

    Py_ssize_t n = PySequence_Size(obj);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item;
        Dashes subdashes;

        item = PySequence_GetItem(obj, i);
        if (item == NULL) {
            return 0;
        }

        if (!convert_dashes(item, &subdashes)) {
            Py_DECREF(item);
            return 0;
        }
        Py_DECREF(item);

        dashes->push_back(subdashes);
    }

    return 1;
}

int convert_trans_affine(PyObject *obj, void *transp)
{
    agg::trans_affine *trans = (agg::trans_affine *)transp;

    /** If None assume identity transform. */
    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    PyArrayObject *array = (PyArrayObject *)PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 2, 2);
    if (array == NULL) {
        return 0;
    }

    if (PyArray_DIM(array, 0) == 3 && PyArray_DIM(array, 1) == 3) {
        double *buffer = (double *)PyArray_DATA(array);
        trans->sx = buffer[0];
        trans->shx = buffer[1];
        trans->tx = buffer[2];

        trans->shy = buffer[3];
        trans->sy = buffer[4];
        trans->ty = buffer[5];

        Py_DECREF(array);
        return 1;
    }

    Py_DECREF(array);
    PyErr_SetString(PyExc_ValueError, "Invalid affine transformation matrix");
    return 0;
}

int convert_path(PyObject *obj, void *pathp)
{
    py::PathIterator *path = (py::PathIterator *)pathp;

    PyObject *vertices_obj = NULL;
    PyObject *codes_obj = NULL;
    PyObject *should_simplify_obj = NULL;
    PyObject *simplify_threshold_obj = NULL;
    bool should_simplify;
    double simplify_threshold;

    int status = 0;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    vertices_obj = PyObject_GetAttrString(obj, "vertices");
    if (vertices_obj == NULL) {
        goto exit;
    }

    codes_obj = PyObject_GetAttrString(obj, "codes");
    if (codes_obj == NULL) {
        goto exit;
    }

    should_simplify_obj = PyObject_GetAttrString(obj, "should_simplify");
    if (should_simplify_obj == NULL) {
        goto exit;
    }
    switch (PyObject_IsTrue(should_simplify_obj)) {
        case 0: should_simplify = 0; break;
        case 1: should_simplify = 1; break;
        default: goto exit;  // errored.
    }

    simplify_threshold_obj = PyObject_GetAttrString(obj, "simplify_threshold");
    if (simplify_threshold_obj == NULL) {
        goto exit;
    }
    simplify_threshold = PyFloat_AsDouble(simplify_threshold_obj);
    if (PyErr_Occurred()) {
        goto exit;
    }

    if (!path->set(vertices_obj, codes_obj, should_simplify, simplify_threshold)) {
        goto exit;
    }

    status = 1;

exit:
    Py_XDECREF(vertices_obj);
    Py_XDECREF(codes_obj);
    Py_XDECREF(should_simplify_obj);
    Py_XDECREF(simplify_threshold_obj);

    return status;
}

int convert_clippath(PyObject *clippath_tuple, void *clippathp)
{
    ClipPath *clippath = (ClipPath *)clippathp;
    py::PathIterator path;
    agg::trans_affine trans;

    if (clippath_tuple != NULL && clippath_tuple != Py_None) {
        if (!PyArg_ParseTuple(clippath_tuple,
                              "O&O&:clippath",
                              &convert_path,
                              &clippath->path,
                              &convert_trans_affine,
                              &clippath->trans)) {
            return 0;
        }
    }

    return 1;
}

int convert_snap(PyObject *obj, void *snapp)
{
    e_snap_mode *snap = (e_snap_mode *)snapp;
    if (obj == NULL || obj == Py_None) {
        *snap = SNAP_AUTO;
    } else {
        switch (PyObject_IsTrue(obj)) {
            case 0: *snap = SNAP_FALSE; break;
            case 1: *snap = SNAP_TRUE; break;
            default: return 0;  // errored.
        }
    }
    return 1;
}

int convert_sketch_params(PyObject *obj, void *sketchp)
{
    SketchParams *sketch = (SketchParams *)sketchp;

    if (obj == NULL || obj == Py_None) {
        sketch->scale = 0.0;
    } else if (!PyArg_ParseTuple(obj,
                                 "ddd:sketch_params",
                                 &sketch->scale,
                                 &sketch->length,
                                 &sketch->randomness)) {
        return 0;
    }

    return 1;
}

int convert_gcagg(PyObject *pygc, void *gcp)
{
    GCAgg *gc = (GCAgg *)gcp;

    if (!(convert_from_attr(pygc, "_linewidth", &convert_double, &gc->linewidth) &&
          convert_from_attr(pygc, "_alpha", &convert_double, &gc->alpha) &&
          convert_from_attr(pygc, "_forced_alpha", &convert_bool, &gc->forced_alpha) &&
          convert_from_attr(pygc, "_rgb", &convert_rgba, &gc->color) &&
          convert_from_attr(pygc, "_antialiased", &convert_bool, &gc->isaa) &&
          convert_from_attr(pygc, "_capstyle", &convert_cap, &gc->cap) &&
          convert_from_attr(pygc, "_joinstyle", &convert_join, &gc->join) &&
          convert_from_method(pygc, "get_dashes", &convert_dashes, &gc->dashes) &&
          convert_from_attr(pygc, "_cliprect", &convert_rect, &gc->cliprect) &&
          convert_from_method(pygc, "get_clip_path", &convert_clippath, &gc->clippath) &&
          convert_from_method(pygc, "get_snap", &convert_snap, &gc->snap_mode) &&
          convert_from_method(pygc, "get_hatch_path", &convert_path, &gc->hatchpath) &&
          convert_from_method(pygc, "get_hatch_color", &convert_rgba, &gc->hatch_color) &&
          convert_from_method(pygc, "get_hatch_linewidth", &convert_double, &gc->hatch_linewidth) &&
          convert_from_method(pygc, "get_sketch_params", &convert_sketch_params, &gc->sketch))) {
        return 0;
    }

    return 1;
}

int convert_offset_position(PyObject *obj, void *offsetp)
{
    e_offset_position *offset = (e_offset_position *)offsetp;
    const char *names[] = {"data", NULL};
    int values[] = {OFFSET_POSITION_DATA};
    int result = (int)OFFSET_POSITION_FIGURE;

    if (!convert_string_enum(obj, "offset_position", names, values, &result)) {
        PyErr_Clear();
    }

    *offset = (e_offset_position)result;

    return 1;
}

int convert_face(PyObject *color, GCAgg &gc, agg::rgba *rgba)
{
    if (!convert_rgba(color, rgba)) {
        return 0;
    }

    if (color != NULL && color != Py_None) {
        if (gc.forced_alpha || PySequence_Size(color) == 3) {
            rgba->a = gc.alpha;
        }
    }

    return 1;
}

int convert_points(PyObject *obj, void *pointsp)
{
    numpy::array_view<double, 2> *points = (numpy::array_view<double, 2> *)pointsp;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    points->set(obj);

    if (points->size() == 0) {
        return 1;
    }

    if (points->dim(1) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "Points must be Nx2 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT,
                     points->dim(0), points->dim(1));
        return 0;
    }

    return 1;
}

int convert_transforms(PyObject *obj, void *transp)
{
    numpy::array_view<double, 3> *trans = (numpy::array_view<double, 3> *)transp;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    trans->set(obj);

    if (trans->size() == 0) {
        return 1;
    }

    if (trans->dim(1) != 3 || trans->dim(2) != 3) {
        PyErr_Format(PyExc_ValueError,
                     "Transforms must be Nx3x3 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT "x%" NPY_INTP_FMT,
                     trans->dim(0), trans->dim(1), trans->dim(2));
        return 0;
    }

    return 1;
}

int convert_bboxes(PyObject *obj, void *bboxp)
{
    numpy::array_view<double, 3> *bbox = (numpy::array_view<double, 3> *)bboxp;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    bbox->set(obj);

    if (bbox->size() == 0) {
        return 1;
    }

    if (bbox->dim(1) != 2 || bbox->dim(2) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "Bbox array must be Nx2x2 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT "x%" NPY_INTP_FMT,
                     bbox->dim(0), bbox->dim(1), bbox->dim(2));
        return 0;
    }

    return 1;
}

int convert_colors(PyObject *obj, void *colorsp)
{
    numpy::array_view<double, 2> *colors = (numpy::array_view<double, 2> *)colorsp;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    colors->set(obj);

    if (colors->size() == 0) {
        return 1;
    }

    if (colors->dim(1) != 4) {
        PyErr_Format(PyExc_ValueError,
                     "Colors array must be Nx4 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT,
                     colors->dim(0), colors->dim(1));
        return 0;
    }

    return 1;
}
#endif
}
