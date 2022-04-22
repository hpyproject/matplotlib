/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include "numpy_cpp.h"

#include "_path.h"

#include "py_converters.h"
#include "py_adaptors.h"

HPy convert_polygon_vector(HPyContext *ctx, std::vector<Polygon> &polygons)
{
    HPyListBuilder pyresult = HPyListBuilder_New(ctx, polygons.size());

    for (size_t i = 0; i < polygons.size(); ++i) {
        Polygon poly = polygons[i];
        npy_intp dims[2];
        dims[1] = 2;

        dims[0] = (npy_intp)poly.size();

        numpy::array_view<double, 2> subresult(dims);
        memcpy(subresult.data(), &poly[0], sizeof(double) * poly.size() * 2);

        HPyListBuilder_Set(ctx, pyresult, i, HPy_FromPyObject(ctx, subresult.pyobj()));
    }

    return HPyListBuilder_Build(ctx, pyresult);
}

static const char *Py_point_in_path__doc__ =
    "point_in_path(x, y, radius, path, trans)\n"
    "--\n\n";

static HPy Py_point_in_path(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_path = HPy_NULL, h_trans = HPy_NULL;
    double x, y, r;
    py::PathIterator path;
    agg::trans_affine trans;
    bool result;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "dddOO:point_in_path",
                          &x,
                          &y,
                          &r,
                          &h_path,
                          &h_trans)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "point_in_path"); // TODO
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "point_in_path", (result = point_in_path(x, y, r, path, trans)));

    if (result) {
        return HPy_Dup(ctx, ctx->h_True);
    } else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}

const char *Py_points_in_path__doc__ =
    "points_in_path(points, radius, path, trans)\n"
    "--\n\n";

static HPy Py_points_in_path(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_points = HPy_NULL;
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    numpy::array_view<const double, 2> points;
    double r;
    py::PathIterator path;
    agg::trans_affine trans;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OdOO:points_in_path",
                          &h_points,
                          &r,
                          &h_path,
                          &h_trans)) {
        return HPy_NULL;
    }

    if (!convert_points_hpy(ctx, h_points, &points)
                || !convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "points_in_path"); // TODO
        return HPy_NULL;
    }

    npy_intp dims[] = { (npy_intp)points.size() };
    numpy::array_view<uint8_t, 1> results(dims);

    CALL_CPP_HPY(ctx, "points_in_path", (points_in_path(points, r, path, trans, results)));

    return HPy_FromPyObject(ctx, results.pyobj());
}

const char *Py_point_on_path__doc__ =
    "point_on_path(x, y, radius, path, trans)\n"
    "--\n\n";

static HPy Py_point_on_path(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    double x, y, r;
    py::PathIterator path;
    agg::trans_affine trans;
    bool result;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "dddOO:point_on_path",
                          &x,
                          &y,
                          &r,
                          &h_path,
                          &h_trans)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "point_on_path"); // TODO
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "point_on_path", (result = point_on_path(x, y, r, path, trans)));

    if (result) {
        return HPy_Dup(ctx, ctx->h_True);
    } else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}

const char *Py_points_on_path__doc__ =
    "points_on_path(points, radius, path, trans)\n"
    "--\n\n";

static HPy Py_points_on_path(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_points = HPy_NULL;
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    numpy::array_view<const double, 2> points;
    double r;
    py::PathIterator path;
    agg::trans_affine trans;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OdOO:points_on_path",
                          &h_points,
                          &r,
                          &h_path,
                          &h_trans)) {
        return HPy_NULL;
    }

    if (!convert_points_hpy(ctx, h_points, &points)
                || !convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "points_on_path"); // TODO
        return HPy_NULL;
    }

    npy_intp dims[] = { (npy_intp)points.size() };
    numpy::array_view<uint8_t, 1> results(dims);

    CALL_CPP_HPY(ctx, "points_on_path", (points_on_path(points, r, path, trans, results)));

    return HPy_FromPyObject(ctx, results.pyobj());
}

const char *Py_get_path_extents__doc__ =
    "get_path_extents(path, trans)\n"
    "--\n\n";

static HPy Py_get_path_extents(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    py::PathIterator path;
    agg::trans_affine trans;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
             "OO:get_path_extents", &h_path, &h_trans)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "get_path_extents"); // TODO
        return HPy_NULL;
    }

    extent_limits e;

    CALL_CPP_HPY(ctx, "get_path_extents", (reset_limits(e)));
    CALL_CPP_HPY(ctx, "get_path_extents", (update_path_extents(path, trans, e)));

    npy_intp dims[] = { 2, 2 };
    numpy::array_view<double, 2> extents(dims);
    extents(0, 0) = e.x0;
    extents(0, 1) = e.y0;
    extents(1, 0) = e.x1;
    extents(1, 1) = e.y1;

    return HPy_FromPyObject(ctx, extents.pyobj());
}

const char *Py_update_path_extents__doc__ =
    "update_path_extents(path, trans, rect, minpos, ignore)\n"
    "--\n\n";

static HPy Py_update_path_extents(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    HPy h_rect = HPy_NULL;
    HPy h_minpos = HPy_NULL;
    py::PathIterator path;
    agg::trans_affine trans;
    agg::rect_d rect;
    numpy::array_view<double, 1> minpos;
    int ignore;
    int changed;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OOOOi:update_path_extents",
                          &h_path,
                          &h_trans,
                          &h_rect,
                          &h_minpos,
                          &ignore)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)
                || !convert_rect_hpy(ctx, h_rect, &rect)
                || !minpos.converter(HPy_AsPyObject(ctx, h_minpos), &minpos)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "update_path_extents"); // TODO
        return HPy_NULL;
    }

    if (minpos.dim(0) != 2) {
        // PyErr_Format(PyExc_ValueError,
        //              "minpos must be of length 2, got %" NPY_INTP_FMT,
        //              minpos.dim(0));
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "minpos must be of length 2");
        return HPy_NULL;
    }

    extent_limits e;

    if (ignore) {
        CALL_CPP_HPY(ctx, "update_path_extents", reset_limits(e));
    } else {
        if (rect.x1 > rect.x2) {
            e.x0 = std::numeric_limits<double>::infinity();
            e.x1 = -std::numeric_limits<double>::infinity();
        } else {
            e.x0 = rect.x1;
            e.x1 = rect.x2;
        }
        if (rect.y1 > rect.y2) {
            e.y0 = std::numeric_limits<double>::infinity();
            e.y1 = -std::numeric_limits<double>::infinity();
        } else {
            e.y0 = rect.y1;
            e.y1 = rect.y2;
        }
        e.xm = minpos(0);
        e.ym = minpos(1);
    }

    CALL_CPP_HPY(ctx, "update_path_extents", (update_path_extents(path, trans, e)));

    changed = (e.x0 != rect.x1 || e.y0 != rect.y1 || e.x1 != rect.x2 || e.y1 != rect.y2 ||
               e.xm != minpos(0) || e.ym != minpos(1));

    npy_intp extentsdims[] = { 2, 2 };
    numpy::array_view<double, 2> outextents(extentsdims);
    outextents(0, 0) = e.x0;
    outextents(0, 1) = e.y0;
    outextents(1, 0) = e.x1;
    outextents(1, 1) = e.y1;

    npy_intp minposdims[] = { 2 };
    numpy::array_view<double, 1> outminpos(minposdims);
    outminpos(0) = e.xm;
    outminpos(1) = e.ym;

    return HPy_BuildValue(ctx, "OOi", 
                                HPy_FromPyObject(ctx, outextents.pyobj()), 
                                HPy_FromPyObject(ctx, outminpos.pyobj()), changed);
}

const char *Py_get_path_collection_extents__doc__ =
    "get_path_collection_extents("
    "master_transform, paths, transforms, offsets, offset_transform)\n"
    "--\n\n";

static HPy Py_get_path_collection_extents(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_master_transform = HPy_NULL;
    HPy h_paths = HPy_NULL;
    HPy h_transforms = HPy_NULL;
    HPy h_offsets = HPy_NULL;
    HPy h_offset_trans = HPy_NULL;
    agg::trans_affine master_transform;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    extent_limits e;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OOOOO:get_path_collection_extents",
                          &h_master_transform,
                          &h_paths,
                          &h_transforms,
                          &h_offsets,
                          &h_offset_trans)) {
        return HPy_NULL;
    }

    if (!convert_trans_affine_hpy(ctx, h_master_transform, &master_transform)
                || !convert_transforms_hpy(ctx, h_transforms, &transforms)
                || !convert_points_hpy(ctx, h_offsets, &offsets)
                || !convert_trans_affine_hpy(ctx, h_offset_trans, &offset_trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "get_path_collection_extents"); // TODO
        return HPy_NULL;
    }


    try
    {
        py::PathGenerator paths(ctx, h_paths);

        CALL_CPP_HPY(ctx, "get_path_collection_extents",
                 (get_path_collection_extents(
                     master_transform, paths, transforms, offsets, offset_trans, e)));
    }
    catch (const py::exception &)
    {
        return HPy_NULL;
    }

    npy_intp dims[] = { 2, 2 };
    numpy::array_view<double, 2> extents(dims);
    extents(0, 0) = e.x0;
    extents(0, 1) = e.y0;
    extents(1, 0) = e.x1;
    extents(1, 1) = e.y1;

    npy_intp minposdims[] = { 2 };
    numpy::array_view<double, 1> minpos(minposdims);
    minpos(0) = e.xm;
    minpos(1) = e.ym;

    return HPy_BuildValue(ctx, "OO", 
                                HPy_FromPyObject(ctx, extents.pyobj()), 
                                HPy_FromPyObject(ctx, minpos.pyobj()));
}

const char *Py_point_in_path_collection__doc__ =
    "point_in_path_collection("
    "x, y, radius, master_transform, paths, transforms, offsets, "
    "offset_trans, filled, offset_position)\n"
    "--\n\n";

static HPy Py_point_in_path_collection(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_master_transform = HPy_NULL;
    HPy h_paths = HPy_NULL;
    HPy h_transforms = HPy_NULL;
    HPy h_offsets = HPy_NULL;
    HPy h_offset_trans = HPy_NULL;
    HPy h_filled = HPy_NULL;
    double x, y, radius;
    agg::trans_affine master_transform;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    bool filled;
    e_offset_position offset_position;
    std::vector<int> result;
    HPy h_offset_position = HPy_NULL;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "dddOOOOOOO:point_in_path_collection",
                          &x,
                          &y,
                          &radius,
                          &h_master_transform,
                          &h_paths,
                          &h_transforms,
                          &h_offsets,
                          &h_offset_trans,
                          &h_filled,
                          &h_offset_position)) {
        return HPy_NULL;
    }

    if (!convert_trans_affine_hpy(ctx, h_master_transform, &master_transform)
                || !convert_transforms_hpy(ctx, h_transforms, &transforms)
                || !convert_points_hpy(ctx, h_offsets, &offsets)
                || !convert_trans_affine_hpy(ctx, h_offset_trans, &offset_trans)
                || !convert_bool_hpy(ctx, h_filled, &filled)
                || !convert_offset_position_hpy(ctx, h_offset_position, &offset_position)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "point_in_path_collection"); // TODO
        return HPy_NULL;
    }


    try
    {
        py::PathGenerator paths(ctx, h_paths);

        CALL_CPP_HPY(ctx, "point_in_path_collection",
                 (point_in_path_collection(x,
                                           y,
                                           radius,
                                           master_transform,
                                           paths,
                                           transforms,
                                           offsets,
                                           offset_trans,
                                           filled,
                                           offset_position,
                                           result)));
    }
    catch (const py::exception &)
    {
        return HPy_NULL;
    }

    npy_intp dims[] = {(npy_intp)result.size() };
    numpy::array_view<int, 1> pyresult(dims);
    if (result.size() > 0) {
        memcpy(pyresult.data(), &result[0], result.size() * sizeof(int));
    }
    return HPy_FromPyObject(ctx, pyresult.pyobj());
}

const char *Py_path_in_path__doc__ =
    "path_in_path(path_a, trans_a, path_b, trans_b)\n"
    "--\n\n";

static HPy Py_path_in_path(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_a = HPy_NULL;
    HPy h_atrans = HPy_NULL;
    HPy h_b = HPy_NULL;
    HPy h_btrans = HPy_NULL;
    py::PathIterator a;
    agg::trans_affine atrans;
    py::PathIterator b;
    agg::trans_affine btrans;
    bool result;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OOOO:path_in_path",
                          &h_a,
                          &h_atrans,
                          &h_b,
                          &h_btrans)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_a, &a)
                || !convert_trans_affine_hpy(ctx, h_atrans, &atrans)
                || !convert_path_hpy(ctx, h_b, &b)
                || !convert_trans_affine_hpy(ctx, h_btrans, &btrans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "path_in_path"); // TODO
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "path_in_path", (result = path_in_path(a, atrans, b, btrans)));

    if (result) {
        return HPy_Dup(ctx, ctx->h_True);
    } else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}

const char *Py_clip_path_to_rect__doc__ =
    "clip_path_to_rect(path, rect, inside)\n"
    "--\n\n";

static HPy Py_clip_path_to_rect(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_path = HPy_NULL;
    HPy h_rect = HPy_NULL;
    HPy h_inside = HPy_NULL;
    py::PathIterator path;
    agg::rect_d rect;
    bool inside;
    std::vector<Polygon> result;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OOO:clip_path_to_rect",
                          &h_path,
                          &h_rect,
                          &h_inside)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || !convert_rect_hpy(ctx, h_rect, &rect)
                || !convert_bool_hpy(ctx, h_inside, &inside)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "clip_path_to_rect"); // TODO
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "clip_path_to_rect", (clip_path_to_rect(path, rect, inside, result)));

    return convert_polygon_vector(ctx, result);
}

const char *Py_affine_transform__doc__ =
    "affine_transform(points, trans)\n"
    "--\n\n";

static HPy Py_affine_transform(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy vertices_obj = HPy_NULL;
    HPy h_trans = HPy_NULL;
    agg::trans_affine trans;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OO:affine_transform",
                          &vertices_obj,
                          &h_trans)) {
        return HPy_NULL;
    }

    if (!convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "affine_transform"); // TODO
        return HPy_NULL;
    }

    PyArrayObject* vertices_arr = (PyArrayObject *)PyArray_ContiguousFromAny(HPy_AsPyObject(ctx, vertices_obj), NPY_DOUBLE, 1, 2);
    if (vertices_arr == NULL) {
        return HPy_NULL;
    }

    if (PyArray_NDIM(vertices_arr) == 2) {
        numpy::array_view<double, 2> vertices(vertices_arr);
        Py_DECREF(vertices_arr);

        npy_intp dims[] = { (npy_intp)vertices.size(), 2 };
        numpy::array_view<double, 2> result(dims);
        CALL_CPP_HPY(ctx, "affine_transform", (affine_transform_2d(vertices, trans, result)));
        return HPy_FromPyObject(ctx, result.pyobj());
    } else { // PyArray_NDIM(vertices_arr) == 1
        numpy::array_view<double, 1> vertices(vertices_arr);
        Py_DECREF(vertices_arr);

        npy_intp dims[] = { (npy_intp)vertices.size() };
        numpy::array_view<double, 1> result(dims);
        CALL_CPP_HPY(ctx, "affine_transform", (affine_transform_1d(vertices, trans, result)));
        return HPy_FromPyObject(ctx, result.pyobj());
    }
}

const char *Py_count_bboxes_overlapping_bbox__doc__ =
    "count_bboxes_overlapping_bbox(bbox, bboxes)\n"
    "--\n\n";

static HPy Py_count_bboxes_overlapping_bbox(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_bbox = HPy_NULL;
    HPy h_bboxes = HPy_NULL;
    agg::rect_d bbox;
    numpy::array_view<const double, 3> bboxes;
    int result;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OO:count_bboxes_overlapping_bbox",
                          &h_bbox,
                          &h_bboxes)) {
        return HPy_NULL;
    }

    if (!convert_rect_hpy(ctx, h_bbox, &bbox)
                || !convert_bboxes_hpy(ctx, h_bboxes, &bboxes)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "count_bboxes_overlapping_bbox"); // TODO
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "count_bboxes_overlapping_bbox",
             (result = count_bboxes_overlapping_bbox(bbox, bboxes)));

    return HPyLong_FromLong(ctx, result);
}

const char *Py_path_intersects_path__doc__ =
    "path_intersects_path(path1, path2, filled=False)\n"
    "--\n\n";

static HPy Py_path_intersects_path(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    HPy h_p1 = HPy_NULL;
    HPy h_p2 = HPy_NULL;
    py::PathIterator p1;
    py::PathIterator p2;
    agg::trans_affine t1;
    agg::trans_affine t2;
    int filled = 0;
    const char *names[] = { "p1", "p2", "filled", NULL };
    bool result;

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
                                     kwds,
                                     "OOi:path_intersects_path",
                                     (const char **)names,
                                     &h_p1,
                                     &h_p2,
                                     &filled)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_p1, &p1)
                || !convert_path_hpy(ctx, h_p2, &p2)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "path_intersects_path"); // TODO
        goto error;
    }

    CALL_CPP_HPY(ctx, "path_intersects_path", (result = path_intersects_path(p1, p2)));
    if (filled) {
        if (!result) {
            CALL_CPP_HPY(ctx, "path_intersects_path",
                     (result = path_in_path(p1, t1, p2, t2)));
        }
        if (!result) {
            CALL_CPP_HPY(ctx, "path_intersects_path",
                     (result = path_in_path(p2, t1, p1, t2)));
        }
    }

    HPyTracker_Close(ctx, ht);
    if (result) {
        return HPy_Dup(ctx, ctx->h_True);
    } else {
        return HPy_Dup(ctx, ctx->h_False);
    }
error:
    HPyTracker_Close(ctx, ht);
    return HPy_NULL;
}

const char *Py_path_intersects_rectangle__doc__ =
    "path_intersects_rectangle("
    "path, rect_x1, rect_y1, rect_x2, rect_y2, filled=False)\n"
    "--\n\n";

static HPy Py_path_intersects_rectangle(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    HPy h_path = HPy_NULL;
    HPy h_filled = HPy_NULL;
    py::PathIterator path;
    double rect_x1, rect_y1, rect_x2, rect_y2;
    bool filled = false;
    const char *names[] = { "path", "rect_x1", "rect_y1", "rect_x2", "rect_y2", "filled", NULL };
    bool result;

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
                                     kwds,
                                     "Odddd|O:path_intersects_rectangle",
                                     (const char **)names,
                                     &h_path,
                                     &rect_x1,
                                     &rect_y1,
                                     &rect_x2,
                                     &rect_y2,
                                     &h_filled)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || (!HPy_IsNull(h_filled) && !convert_bool_hpy(ctx, h_filled, &filled))) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "path_intersects_rectangle"); // TODO
        goto error;
    }

    CALL_CPP_HPY(ctx, "path_intersects_rectangle", (result = path_intersects_rectangle(path, rect_x1, rect_y1, rect_x2, rect_y2, filled)));

    HPyTracker_Close(ctx, ht);
    if (result) {
        return HPy_Dup(ctx, ctx->h_True);
    } else {
        return HPy_Dup(ctx, ctx->h_False);
    }
error:
    HPyTracker_Close(ctx, ht);
    return HPy_NULL;
}

const char *Py_convert_path_to_polygons__doc__ =
    "convert_path_to_polygons(path, trans, width=0, height=0)\n"
    "--\n\n";

static HPy Py_convert_path_to_polygons(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    py::PathIterator path;
    agg::trans_affine trans;
    double width = 0.0, height = 0.0;
    int closed_only = 1;
    std::vector<Polygon> result;
    const char *names[] = { "path", "transform", "width", "height", "closed_only", NULL };

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
                                     kwds,
                                     "OO|ddi:convert_path_to_polygons",
                                     (const char **)names,
                                     &h_path,
                                     &h_trans,
                                     &width,
                                     &height,
                                     &closed_only)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "convert_path_to_polygons"); // TODO
        goto error;
    }

    CALL_CPP_HPY(ctx, "convert_path_to_polygons",
             (convert_path_to_polygons(path, trans, width, height, closed_only, result)));

    HPyTracker_Close(ctx, ht);
    return convert_polygon_vector(ctx, result);
error:
    HPyTracker_Close(ctx, ht);
    return HPy_NULL;
}

const char *Py_cleanup_path__doc__ =
    "cleanup_path("
    "path, trans, remove_nans, clip_rect, snap_mode, stroke_width, simplify, "
    "return_curves, sketch)\n"
    "--\n\n";

static HPy Py_cleanup_path(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    HPy h_remove_nans = HPy_NULL;
    HPy h_clip_rect = HPy_NULL;
    HPy h_snap_mode = HPy_NULL;
    HPy h_return_curves = HPy_NULL;
    HPy h_sketch = HPy_NULL;
    py::PathIterator path;
    agg::trans_affine trans;
    bool remove_nans;
    agg::rect_d clip_rect;
    e_snap_mode snap_mode;
    double stroke_width;
    HPy simplifyobj;
    bool simplify = false;
    bool return_curves;
    SketchParams sketch;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OOOOOdOOO:cleanup_path",
                          &h_path,
                          &h_trans,
                          &h_remove_nans,
                          &h_clip_rect,
                          &h_snap_mode,
                          &stroke_width,
                          &simplifyobj,
                          &h_return_curves,
                          &h_sketch)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)
                || !convert_bool_hpy(ctx, h_remove_nans, &remove_nans)
                || !convert_rect_hpy(ctx, h_clip_rect, &clip_rect)
                || !convert_snap_hpy(ctx, h_snap_mode, &snap_mode)
                || !convert_bool_hpy(ctx, h_return_curves, &return_curves)
                || !convert_sketch_params_hpy(ctx, h_sketch, &sketch)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "cleanup_path"); // TODO
        return HPy_NULL;
    }

    if (HPy_Is(ctx, simplifyobj, ctx->h_None)) {
        simplify = path.should_simplify();
    } else {
        switch (HPy_IsTrue(ctx, simplifyobj)) {
            case 0: simplify = false; break;
            case 1: simplify = true; break;
            default: return HPy_NULL;  // errored.
        }
    }

    bool do_clip = (clip_rect.x1 < clip_rect.x2 && clip_rect.y1 < clip_rect.y2);

    std::vector<double> vertices;
    std::vector<npy_uint8> codes;

    CALL_CPP_HPY(ctx, "cleanup_path",
             (cleanup_path(path,
                           trans,
                           remove_nans,
                           do_clip,
                           clip_rect,
                           snap_mode,
                           stroke_width,
                           simplify,
                           return_curves,
                           sketch,
                           vertices,
                           codes)));

    size_t length = codes.size();

    npy_intp vertices_dims[] = {(npy_intp)length, 2 };
    numpy::array_view<double, 2> pyvertices(vertices_dims);

    npy_intp codes_dims[] = {(npy_intp)length };
    numpy::array_view<unsigned char, 1> pycodes(codes_dims);

    memcpy(pyvertices.data(), &vertices[0], sizeof(double) * 2 * length);
    memcpy(pycodes.data(), &codes[0], sizeof(unsigned char) * length);

    return HPy_BuildValue(ctx, "OO", 
                            HPy_FromPyObject(ctx, pyvertices.pyobj()), 
                            HPy_FromPyObject(ctx, pycodes.pyobj()));
}

const char *Py_convert_to_string__doc__ =
    "convert_to_string("
    "path, trans, clip_rect, simplify, sketch, precision, codes, postfix)\n"
    "--\n\n"
    "Convert *path* to a bytestring.\n"
    "\n"
    "The first five parameters (up to *sketch*) are interpreted as in \n"
    "`.cleanup_path`.  The following ones are detailed below.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "path : Path\n"
    "trans : Transform or None\n"
    "clip_rect : sequence of 4 floats, or None\n"
    "simplify : bool\n"
    "sketch : tuple of 3 floats, or None\n"
    "precision : int\n"
    "    The precision used to \"%.*f\"-format the values.  Trailing zeros\n"
    "    and decimal points are always removed.  (precision=-1 is a special \n"
    "    case used to implement ttconv-back-compatible conversion.)\n"
    "codes : sequence of 5 bytestrings\n"
    "    The bytes representation of each opcode (MOVETO, LINETO, CURVE3,\n"
    "    CURVE4, CLOSEPOLY), in that order.  If the bytes for CURVE3 is\n"
    "    empty, quad segments are automatically converted to cubic ones\n"
    "    (this is used by backends such as pdf and ps, which do not support\n"
    "    quads).\n"
    "postfix : bool\n"
    "    Whether the opcode comes after the values (True) or before (False).\n"
    ;

static HPy Py_convert_to_string(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    HPy h_cliprect = HPy_NULL;
    HPy h_sketch = HPy_NULL;
    HPy h_postfix = HPy_NULL;
    HPy h_codes = HPy_NULL;
    py::PathIterator path;
    agg::trans_affine trans;
    agg::rect_d cliprect;
    HPy simplifyobj;
    bool simplify = false;
    SketchParams sketch;
    int precision;
    char *codes[5];
    bool postfix;
    std::string buffer;
    bool status;


    if (!HPyArg_Parse(ctx, NULL, args, nargs, 
                          "OOOOOiOO:convert_to_string",
                          &h_path,
                          &h_trans,
                          &h_cliprect,
                          &simplifyobj,
                          &h_sketch,
                          &precision,
                          &h_codes,
                          &h_postfix)) {
        return HPy_NULL;
    }

    if (!convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)
                || !convert_rect_hpy(ctx, h_cliprect, &cliprect)
                || !convert_sketch_params_hpy(ctx, h_sketch, &sketch)
                || !convert_bool_hpy(ctx, h_postfix, &postfix)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "convert_to_string"); // TODO
        return HPy_NULL;
    }

    if (!HPyTuple_Check(ctx, h_codes) && !HPyList_Check(ctx, h_codes)) { // (yyyyy) not supported
        HPyErr_SetString(ctx, ctx->h_TypeError, "convert_to_string");
        return HPy_NULL;
    }

    HPy_ssize_t codes_len = HPy_Length(ctx, h_codes);
    if (codes_len != 5) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "convert_to_string");
        return HPy_NULL;
    }

    for (HPy_ssize_t i=0; i < codes_len; i++) {
        HPy item = HPy_GetItem_i(ctx, h_codes, i);
        if (!HPyBytes_Check(ctx, item)) {
            HPyErr_SetString(ctx, ctx->h_TypeError, "convert_to_string");
            return HPy_NULL;
        }
        codes[i] = HPyBytes_AsString(ctx, item);
        HPy_Close(ctx, item);
    }

    if (HPy_Is(ctx, simplifyobj, ctx->h_None)) {
        simplify = path.should_simplify();
    } else {
        switch (HPy_IsTrue(ctx, simplifyobj)) {
            case 0: simplify = false; break;
            case 1: simplify = true; break;
            default: return HPy_NULL;  // errored.
        }
    }

    CALL_CPP_HPY(ctx, "convert_to_string",
             (status = convert_to_string(
                 path, trans, cliprect, simplify, sketch,
                 precision, codes, postfix, buffer)));

    if (!status) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Malformed path codes");
        return HPy_NULL;
    }

    return HPyBytes_FromStringAndSize(ctx, buffer.c_str(), buffer.size());
}


const char *Py_is_sorted__doc__ =
    "is_sorted(array)\n"
    "--\n\n"
    "Return whether the 1D *array* is monotonically increasing, ignoring NaNs.\n";

static HPy Py_is_sorted(HPyContext *ctx, HPy self, HPy obj)
{
    npy_intp size;
    bool result;

    PyArrayObject *array = (PyArrayObject *)PyArray_FromAny(
        HPy_AsPyObject(ctx, obj), NULL, 1, 1, 0, NULL);

    if (array == NULL) {
        return HPy_NULL;
    }

    size = PyArray_DIM(array, 0);

    if (size < 2) {
        Py_DECREF(array);
        return HPy_Dup(ctx, ctx->h_True);
    }

    /* Handle just the most common types here, otherwise coerce to
    double */
    switch(PyArray_TYPE(array)) {
    case NPY_INT:
        {
            _is_sorted_int<npy_int> is_sorted;
            result = is_sorted(array);
        }
        break;

    case NPY_LONG:
        {
            _is_sorted_int<npy_long> is_sorted;
            result = is_sorted(array);
        }
        break;

    case NPY_LONGLONG:
        {
            _is_sorted_int<npy_longlong> is_sorted;
            result = is_sorted(array);
        }
        break;

    case NPY_FLOAT:
        {
            _is_sorted<npy_float> is_sorted;
            result = is_sorted(array);
        }
        break;

    case NPY_DOUBLE:
        {
            _is_sorted<npy_double> is_sorted;
            result = is_sorted(array);
        }
        break;

    default:
        {
            Py_DECREF(array);
            array = (PyArrayObject *)PyArray_FromObject(HPy_AsPyObject(ctx, obj), NPY_DOUBLE, 1, 1);

            if (array == NULL) {
                return HPy_NULL;
            }

            _is_sorted<npy_double> is_sorted;
            result = is_sorted(array);
        }
    }

    Py_DECREF(array);

    if (result) {
        return HPy_Dup(ctx, ctx->h_True);
    } else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}


HPyDef_METH(Py_point_in_path_def, "point_in_path", Py_point_in_path, HPyFunc_VARARGS, .doc = Py_point_in_path__doc__);
HPyDef_METH(Py_points_in_path_def, "points_in_path", Py_points_in_path, HPyFunc_VARARGS, .doc = Py_points_in_path__doc__);
HPyDef_METH(Py_point_on_path_def, "point_on_path", Py_point_on_path, HPyFunc_VARARGS, .doc = Py_point_on_path__doc__);
HPyDef_METH(Py_points_on_path_def, "points_on_path", Py_points_on_path, HPyFunc_VARARGS, .doc = Py_points_on_path__doc__);
HPyDef_METH(Py_get_path_extents_def, "get_path_extents", Py_get_path_extents, HPyFunc_VARARGS, .doc = Py_get_path_extents__doc__);
HPyDef_METH(Py_update_path_extents_def, "update_path_extents", Py_update_path_extents, HPyFunc_VARARGS, .doc = Py_update_path_extents__doc__);
HPyDef_METH(Py_get_path_collection_extents_def, "get_path_collection_extents", Py_get_path_collection_extents, HPyFunc_VARARGS, .doc = Py_get_path_collection_extents__doc__);
HPyDef_METH(Py_point_in_path_collection_def, "point_in_path_collection", Py_point_in_path_collection, HPyFunc_VARARGS, .doc = Py_point_in_path_collection__doc__);
HPyDef_METH(Py_path_in_path_def, "path_in_path", Py_path_in_path, HPyFunc_VARARGS, .doc = Py_path_in_path__doc__);
HPyDef_METH(Py_clip_path_to_rect_def, "clip_path_to_rect", Py_clip_path_to_rect, HPyFunc_VARARGS, .doc = Py_clip_path_to_rect__doc__);
HPyDef_METH(Py_affine_transform_def, "affine_transform", Py_affine_transform, HPyFunc_VARARGS, .doc = Py_affine_transform__doc__);
HPyDef_METH(Py_count_bboxes_overlapping_bbox_def, "count_bboxes_overlapping_bbox", Py_count_bboxes_overlapping_bbox, HPyFunc_VARARGS, .doc = Py_count_bboxes_overlapping_bbox__doc__);
HPyDef_METH(Py_path_intersects_path_def, "path_intersects_path", Py_path_intersects_path, HPyFunc_KEYWORDS, .doc = Py_path_intersects_path__doc__);
HPyDef_METH(Py_path_intersects_rectangle_def, "path_intersects_rectangle", Py_path_intersects_rectangle, HPyFunc_KEYWORDS, .doc = Py_path_intersects_rectangle__doc__);
HPyDef_METH(Py_convert_path_to_polygons_def, "convert_path_to_polygons", Py_convert_path_to_polygons, HPyFunc_KEYWORDS, .doc = Py_convert_path_to_polygons__doc__);
HPyDef_METH(Py_cleanup_path_def, "cleanup_path", Py_cleanup_path, HPyFunc_VARARGS, .doc = Py_cleanup_path__doc__);
HPyDef_METH(Py_convert_to_string_def, "convert_to_string", Py_convert_to_string, HPyFunc_VARARGS, .doc = Py_convert_to_string__doc__);
HPyDef_METH(Py_is_sorted_def, "is_sorted", Py_is_sorted, HPyFunc_O, .doc = Py_is_sorted__doc__);

static HPyDef *module_defines[] = {
    &Py_point_in_path_def,
    &Py_points_in_path_def,
    &Py_point_on_path_def,
    &Py_points_on_path_def,
    &Py_get_path_extents_def,
    &Py_update_path_extents_def,
    &Py_get_path_collection_extents_def,
    &Py_point_in_path_collection_def,
    &Py_path_in_path_def,
    &Py_clip_path_to_rect_def,
    &Py_affine_transform_def,
    &Py_count_bboxes_overlapping_bbox_def,
    &Py_path_intersects_path_def,
    &Py_path_intersects_rectangle_def,
    &Py_convert_path_to_polygons_def,
    &Py_cleanup_path_def,
    &Py_convert_to_string_def,
    &Py_is_sorted_def,
    NULL
};

static HPyModuleDef moduledef = {
  .name = "_path",
  .doc = NULL,
  .size = 0,
  .defines = module_defines,
};


// Logic is from NumPy's import_array()
static int npy_import_array_hpy(HPyContext *ctx) {
    if (_import_array() < 0) {
        // HPyErr_Print(ctx); TODO
        HPyErr_SetString(ctx, ctx->h_ImportError, "numpy.core.multiarray failed to import"); 
        return 0; 
    }
    return 1;
}

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(default)
HPy_MODINIT(_path)
static HPy init__path_impl(HPyContext *ctx)
{
    if (!npy_import_array_hpy(ctx)) {
        return HPy_NULL;
    }

    return HPyModule_Create(ctx, &moduledef);
}

#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
