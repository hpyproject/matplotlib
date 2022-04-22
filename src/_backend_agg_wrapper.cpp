/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include "mplutils.h"
#include "numpy_cpp.h"
#include "py_converters.h"
#include "_backend_agg.h"
#include "hpy.h"

typedef struct
{
    RendererAgg *x;
    HPy_ssize_t shape[3];
    HPy_ssize_t strides[3];
    HPy_ssize_t suboffsets[3];
} PyRendererAgg;

HPyType_HELPERS(PyRendererAgg)


typedef struct
{
    BufferRegion *x;
    HPy_ssize_t shape[3];
    HPy_ssize_t strides[3];
    HPy_ssize_t suboffsets[3];
} PyBufferRegion;

HPyType_HELPERS(PyBufferRegion)

/**********************************************************************
 * BufferRegion
 * */

static HPy PyBufferRegion_new(HPyContext *ctx, HPy type, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    PyBufferRegion *self;
    HPy h_self = HPy_New(ctx, type, &self);
    self->x = NULL;
    return h_self;
}

static void PyBufferRegion_dealloc(void *obj)
{
    PyBufferRegion* self = (PyBufferRegion*)obj;
    delete self->x;
    //Py_TYPE(self)->tp_free((PyObject *)self);
}

static HPy PyBufferRegion_to_string(HPyContext *ctx, HPy h_self)
{
    PyBufferRegion* self = PyBufferRegion_AsStruct(ctx, h_self);
    return HPyBytes_FromStringAndSize(ctx, (const char *)self->x->get_data(),
                                     self->x->get_height() * self->x->get_stride());
}

/* TODO: This doesn't seem to be used internally.  Remove? */

static HPy PyBufferRegion_set_x(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    int x;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "i:set_x", &x)) {
        return HPy_NULL;
    }
    PyBufferRegion* self = PyBufferRegion_AsStruct(ctx, h_self);
    self->x->get_rect().x1 = x;

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy PyBufferRegion_set_y(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    int y;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "i:set_y", &y)) {
        return HPy_NULL;
    }
    PyBufferRegion* self = PyBufferRegion_AsStruct(ctx, h_self);
    self->x->get_rect().y1 = y;

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy PyBufferRegion_get_extents(HPyContext *ctx, HPy h_self)
{
    PyBufferRegion* self = PyBufferRegion_AsStruct(ctx, h_self);
    agg::rect_i rect = self->x->get_rect();

    return HPy_BuildValue(ctx, "IIII", rect.x1, rect.y1, rect.x2, rect.y2);
}

static HPy PyBufferRegion_to_string_argb(HPyContext *ctx, HPy h_self)
{
    PyBufferRegion* self = PyBufferRegion_AsStruct(ctx, h_self);
    HPy bufobj = HPyBytes_FromStringAndSize(ctx, NULL, self->x->get_height() * self->x->get_stride());
    uint8_t *buf = (uint8_t *)HPyBytes_AS_STRING(ctx, bufobj);

    CALL_CPP_CLEANUP_HPY(ctx, "to_string_argb", (self->x->to_string_argb(buf)), HPy_Close(ctx, bufobj));

    return bufobj;
}

static int PyBufferRegion_get_buffer(HPyContext *ctx, HPy h_self, HPy_buffer* buf, int flags)
{
    PyBufferRegion* self = PyBufferRegion_AsStruct(ctx, h_self);
    buf->obj = HPy_Dup(ctx, h_self);
    buf->buf = self->x->get_data();
    buf->len = (HPy_ssize_t)self->x->get_width() * (HPy_ssize_t)self->x->get_height() * 4;
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 3;
    self->shape[0] = self->x->get_height();
    self->shape[1] = self->x->get_width();
    self->shape[2] = 4;
    buf->shape = self->shape;
    self->strides[0] = self->x->get_width() * 4;
    self->strides[1] = 4;
    self->strides[2] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}

HPyDef_SLOT(PyBufferRegion_new_def, PyBufferRegion_new, HPy_tp_new)
HPyDef_SLOT(PyBufferRegion_get_buffer_def, PyBufferRegion_get_buffer, HPy_bf_getbuffer)
HPyDef_SLOT(PyBufferRegion_dealloc_def, PyBufferRegion_dealloc, HPy_tp_destroy)

HPyDef_METH(PyBufferRegion_to_string_def, "to_string", PyBufferRegion_to_string, HPyFunc_NOARGS)
HPyDef_METH(PyBufferRegion_to_string_argb_def, "to_string_argb", PyBufferRegion_to_string_argb, HPyFunc_NOARGS)
HPyDef_METH(PyBufferRegion_set_x_def, "set_x", PyBufferRegion_set_x, HPyFunc_VARARGS)
HPyDef_METH(PyBufferRegion_set_y_def, "set_y", PyBufferRegion_set_y, HPyFunc_VARARGS)
HPyDef_METH(PyBufferRegion_get_extents_def, "get_extents", PyBufferRegion_get_extents, HPyFunc_NOARGS)

HPyDef *PyBufferRegion_defines[] = {
    // slots
    &PyBufferRegion_new_def,
    &PyBufferRegion_get_buffer_def,
    &PyBufferRegion_dealloc_def,

    // methods
    &PyBufferRegion_to_string_def,
    &PyBufferRegion_to_string_argb_def,
    &PyBufferRegion_set_x_def,
    &PyBufferRegion_set_y_def,
    &PyBufferRegion_get_extents_def,
    NULL
};

HPyType_Spec PyBufferRegion_type_spec = {
    .name = "matplotlib.backends._backend_agg.BufferRegion",
    .basicsize = sizeof(PyBufferRegion),
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_BASETYPE,
    .defines = PyBufferRegion_defines,
};


/**********************************************************************
 * RendererAgg
 * */

static HPy PyRendererAgg_new(HPyContext *ctx, HPy type, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    PyRendererAgg *self;
    HPy h_self = HPy_New(ctx, type, &self);
    self->x = NULL;
    return h_self;
}

static int PyRendererAgg_init(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    unsigned int width;
    unsigned int height;
    double dpi;
    int debug = 0;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "IId|i:RendererAgg", &width, &height, &dpi, &debug)) {
        return -1;
    }

    if (dpi <= 0.0) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "dpi must be positive");
        return -1;
    }

    if (width >= 1 << 16 || height >= 1 << 16) {
        // PyErr_Format(
        //     PyExc_ValueError,
        //     "Image size of %dx%d pixels is too large. "
        //     "It must be less than 2^16 in each direction.",
        //     width, height);
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "Image size is too large. "
            "It must be less than 2^16 in each direction.");
        return -1;
    }

    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    CALL_CPP_INIT_HPY(ctx, "RendererAgg", self->x = new RendererAgg(width, height, dpi))

    return 0;
}

static void PyRendererAgg_dealloc(void *obj)
{
    PyRendererAgg* self = (PyRendererAgg*)obj;
    delete self->x;
    //Py_TYPE(self)->tp_free((PyObject *)self);
}

static HPy PyRendererAgg_draw_path(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    HPy h_gc = HPy_NULL;
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    GCAgg gc;
    py::PathIterator path;
    agg::trans_affine trans;
    HPy faceobj = HPy_NULL;
    agg::rgba face;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OOO|O:draw_path",
                          &h_gc,
                          &h_path,
                          &h_trans,
                          &faceobj)) {
        return HPy_NULL;
    }

    if (!convert_gcagg_hpy(ctx, h_gc, &gc)
                || !convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)
                || !convert_face_hpy(ctx, faceobj, gc, &face)) {
            if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_path"); // TODO
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "draw_path", (self->x->draw_path(gc, path, trans, face)));

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy PyRendererAgg_draw_text_image(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_image = HPy_NULL;
    HPy h_gc = HPy_NULL;
    numpy::array_view<agg::int8u, 2> image;
    double x;
    double y;
    double angle;
    GCAgg gc;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OdddO:draw_text_image",
                          &h_image,
                          &x,
                          &y,
                          &angle,
                          &h_gc)) {
        return HPy_NULL;
    }

    if (!image.converter_contiguous(HPy_AsPyObject(ctx, h_image), &image)
            || !convert_gcagg_hpy(ctx, h_gc, &gc)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_text_image"); // TODO
        return HPy_NULL;
    }

    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "draw_text_image", (self->x->draw_text_image(gc, image, x, y, angle)));

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy PyRendererAgg_draw_markers(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_gc = HPy_NULL;
    HPy h_marker_path = HPy_NULL;
    HPy h_marker_path_trans = HPy_NULL;
    HPy h_path = HPy_NULL;
    HPy h_trans = HPy_NULL;
    GCAgg gc;
    py::PathIterator marker_path;
    agg::trans_affine marker_path_trans;
    py::PathIterator path;
    agg::trans_affine trans;
    HPy faceobj = HPy_NULL;
    agg::rgba face;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OOOOO|O:draw_markers",
                          &h_gc,
                          &h_marker_path,
                          &h_marker_path_trans,
                          &h_path,
                          &h_trans,
                          &faceobj)) {
        return HPy_NULL;
    }

    if (!convert_gcagg_hpy(ctx, h_gc, &gc)
                || !convert_path_hpy(ctx, h_marker_path, &marker_path)
                || !convert_trans_affine_hpy(ctx, h_marker_path_trans, &marker_path_trans)
                || !convert_path_hpy(ctx, h_path, &path)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)
                || !convert_face_hpy(ctx, faceobj, gc, &face)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_markers"); // TODO
        return HPy_NULL;
    }

    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "draw_markers",
             (self->x->draw_markers(gc, marker_path, marker_path_trans, path, trans, face)));

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy PyRendererAgg_draw_image(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_gc = HPy_NULL;
    HPy h_image = HPy_NULL;
    GCAgg gc;
    double x;
    double y;
    numpy::array_view<agg::int8u, 3> image;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OddO:draw_image",
                          &h_gc,
                          &x,
                          &y,
                          &h_image)) {
        return HPy_NULL;
    }

    if (!convert_gcagg_hpy(ctx, h_gc, &gc)
                || !image.converter_contiguous(HPy_AsPyObject(ctx, h_image), &image)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_image"); // TODO
        return HPy_NULL;
    }

    x = mpl_round(x);
    y = mpl_round(y);

    gc.alpha = 1.0;
    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "draw_image", (self->x->draw_image(gc, x, y, image)));

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy 
PyRendererAgg_draw_path_collection(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_gc = HPy_NULL;
    HPy h_master_transform = HPy_NULL;
    HPy h_paths = HPy_NULL;
    HPy h_transforms = HPy_NULL;
    HPy h_offsets = HPy_NULL;
    HPy h_offset_trans = HPy_NULL;
    HPy h_facecolors = HPy_NULL;
    HPy h_edgecolors = HPy_NULL;
    HPy h_linewidths = HPy_NULL;
    HPy h_dashes = HPy_NULL;
    HPy h_antialiaseds = HPy_NULL;
    GCAgg gc;
    agg::trans_affine master_transform;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    numpy::array_view<const double, 2> facecolors;
    numpy::array_view<const double, 2> edgecolors;
    numpy::array_view<const double, 1> linewidths;
    DashesVector dashes;
    numpy::array_view<const uint8_t, 1> antialiaseds;
    HPy ignored;
    e_offset_position offset_position;
    HPy h_offset_position = HPy_NULL;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OOOOOOOOOOOOO:draw_path_collection",
                          &h_gc,
                          &h_master_transform,
                          &h_paths,
                          &h_transforms,
                          &h_offsets,
                          &h_offset_trans,
                          &h_facecolors,
                          &h_edgecolors,
                          &h_linewidths,
                          &h_dashes,
                          &h_antialiaseds,
                          &ignored,
                          &h_offset_position)) {
        return HPy_NULL;
    }

    if (!convert_gcagg_hpy(ctx, h_gc, &gc)
                || !convert_trans_affine_hpy(ctx, h_master_transform, &master_transform)
                || !convert_transforms_hpy(ctx, h_transforms, &transforms)
                || !convert_points_hpy(ctx, h_offsets, &offsets)
                || !convert_trans_affine_hpy(ctx, h_offset_trans, &offset_trans)
                || !convert_colors_hpy(ctx, h_facecolors, &facecolors)
                || !convert_colors_hpy(ctx, h_edgecolors, &edgecolors)
                || !linewidths.converter(HPy_AsPyObject(ctx, h_linewidths), &linewidths)
                || !convert_dashes_vector_hpy(ctx, h_dashes, &dashes)
                || !antialiaseds.converter(HPy_AsPyObject(ctx, h_antialiaseds), &antialiaseds)
                || !convert_offset_position_hpy(ctx, h_offset_position, &offset_position)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_path_collection"); // TODO
        return HPy_NULL;
    }

    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    try
    {
        py::PathGenerator path(ctx, h_paths);

        CALL_CPP_HPY(ctx, "draw_path_collection",
                 (self->x->draw_path_collection(gc,
                                                master_transform,
                                                path,
                                                transforms,
                                                offsets,
                                                offset_trans,
                                                facecolors,
                                                edgecolors,
                                                linewidths,
                                                dashes,
                                                antialiaseds,
                                                offset_position)));
    }
    catch (const py::exception &)
    {
        return HPy_NULL;
    }

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy PyRendererAgg_draw_quad_mesh(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_gc = HPy_NULL;
    HPy h_master_transform = HPy_NULL;
    HPy h_coordinates = HPy_NULL;
    HPy h_offsets = HPy_NULL;
    HPy h_offset_trans = HPy_NULL;
    HPy h_facecolors = HPy_NULL;
    HPy h_antialiased = HPy_NULL;
    HPy h_edgecolors = HPy_NULL;
    GCAgg gc;
    agg::trans_affine master_transform;
    unsigned int mesh_width;
    unsigned int mesh_height;
    numpy::array_view<const double, 3> coordinates;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    numpy::array_view<const double, 2> facecolors;
    bool antialiased;
    numpy::array_view<const double, 2> edgecolors;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OOIIOOOOOO:draw_quad_mesh",
                          &h_gc,
                          &h_master_transform,
                          &mesh_width,
                          &mesh_height,
                          &h_coordinates,
                          &h_offsets,
                          &h_offset_trans,
                          &h_facecolors,
                          &h_antialiased,
                          &h_edgecolors)) {
        return HPy_NULL;
    }

    if (!convert_gcagg_hpy(ctx, h_gc, &gc)
                || !convert_trans_affine_hpy(ctx, h_master_transform, &master_transform)
                || !coordinates.converter(HPy_AsPyObject(ctx, h_coordinates), &coordinates)
                || !convert_points_hpy(ctx, h_offsets, &offsets)
                || !convert_trans_affine_hpy(ctx, h_offset_trans, &offset_trans)
                || !convert_colors_hpy(ctx, h_facecolors, &facecolors)
                || !convert_bool_hpy(ctx, h_antialiased, &antialiased)
                || !convert_colors_hpy(ctx, h_edgecolors, &edgecolors)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_quad_mesh"); // TODO
        return HPy_NULL;
    }

    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "draw_quad_mesh",
             (self->x->draw_quad_mesh(gc,
                                      master_transform,
                                      mesh_width,
                                      mesh_height,
                                      coordinates,
                                      offsets,
                                      offset_trans,
                                      facecolors,
                                      antialiased,
                                      edgecolors)));

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy 
PyRendererAgg_draw_gouraud_triangle(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_gc = HPy_NULL;
    HPy h_points = HPy_NULL;
    HPy h_colors = HPy_NULL;
    HPy h_trans = HPy_NULL;
    GCAgg gc;
    numpy::array_view<const double, 2> points;
    numpy::array_view<const double, 2> colors;
    agg::trans_affine trans;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OOOO|O:draw_gouraud_triangle",
                          &h_gc,
                          &h_points,
                          &h_colors,
                          &h_trans)) {
        return HPy_NULL;
    }

    if (!convert_gcagg_hpy(ctx, h_gc, &gc)
                || !points.converter(HPy_AsPyObject(ctx, h_points), &points)
                || !colors.converter(HPy_AsPyObject(ctx, h_colors), &colors)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_gouraud_triangle"); // TODO
        return HPy_NULL;
    }

    if (points.dim(0) != 3 || points.dim(1) != 2) {
        // PyErr_Format(PyExc_ValueError,
        //              "points must be a 3x2 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT,
        //              points.dim(0), points.dim(1)); TODO: HPyErr_Format
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "points must be a 3x2 array");
        return HPy_NULL;
    }

    if (colors.dim(0) != 3 || colors.dim(1) != 4) {
        // PyErr_Format(PyExc_ValueError,
        //              "colors must be a 3x4 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT,
        //              colors.dim(0), colors.dim(1)); TODO: HPyErr_Format
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "colors must be a 3x4 array");
        return HPy_NULL;
    }


    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "draw_gouraud_triangle", (self->x->draw_gouraud_triangle(gc, points, colors, trans)));

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy 
PyRendererAgg_draw_gouraud_triangles(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_gc = HPy_NULL;
    HPy h_points = HPy_NULL;
    HPy h_colors = HPy_NULL;
    HPy h_trans = HPy_NULL;
    GCAgg gc;
    numpy::array_view<const double, 3> points;
    numpy::array_view<const double, 3> colors;
    agg::trans_affine trans;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OOOO|O:draw_gouraud_triangles",
                          &h_gc,
                          &h_points,
                          &h_colors,
                          &h_trans)) {
        return HPy_NULL;
    }

    if (!convert_gcagg_hpy(ctx, h_gc, &gc)
                || !points.converter(HPy_AsPyObject(ctx, h_points), &points)
                || !colors.converter(HPy_AsPyObject(ctx, h_colors), &colors)
                || !convert_trans_affine_hpy(ctx, h_trans, &trans)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_gouraud_triangles"); // TODO
        return HPy_NULL;
    }

    if (points.size() != 0 && (points.dim(1) != 3 || points.dim(2) != 2)) {
        // PyErr_Format(PyExc_ValueError,
        //              "points must be a Nx3x2 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT "x%" NPY_INTP_FMT,
        //              points.dim(0), points.dim(1), points.dim(2));
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "points must be a Nx3x2 array");
        return HPy_NULL;
    }

    if (colors.size() != 0 && (colors.dim(1) != 3 || colors.dim(2) != 4)) {
        // PyErr_Format(PyExc_ValueError,
        //              "colors must be a Nx3x4 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT "x%" NPY_INTP_FMT,
        //              colors.dim(0), colors.dim(1), colors.dim(2));
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "colors must be a Nx3x4 array");
        return HPy_NULL;
    }

    if (points.size() != colors.size()) {
        // PyErr_Format(PyExc_ValueError,
        //              "points and colors arrays must be the same length, got %" NPY_INTP_FMT " and %" NPY_INTP_FMT,
        //              points.dim(0), colors.dim(0));
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "points and colors arrays must be the same length");
        return HPy_NULL;
    }

    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "draw_gouraud_triangles", self->x->draw_gouraud_triangles(gc, points, colors, trans));

    return HPy_Dup(ctx, ctx->h_None);
}

static int PyRendererAgg_get_buffer(HPyContext *ctx, HPy h_self, HPy_buffer* buf, int flags)
{
    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    buf->obj = HPy_Dup(ctx, h_self);
    buf->buf = self->x->pixBuffer;
    buf->len = (HPy_ssize_t)self->x->get_width() * (HPy_ssize_t)self->x->get_height() * 4;
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 3;
    self->shape[0] = self->x->get_height();
    self->shape[1] = self->x->get_width();
    self->shape[2] = 4;
    buf->shape = self->shape;
    self->strides[0] = self->x->get_width() * 4;
    self->strides[1] = 4;
    self->strides[2] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}

static HPy PyRendererAgg_clear(HPyContext *ctx, HPy h_self)
{
    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "clear", self->x->clear());

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy PyRendererAgg_copy_from_bbox(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    agg::rect_d bbox;
    BufferRegion *reg;
    HPy h_regobj;
    HPy m;
    HPy h_bbox = HPy_NULL;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OO:copy_from_bbox", &h_bbox, &m /* _backend_agg module */)) {
                return HPy_NULL;
    }

    if (!convert_rect_hpy(ctx, h_bbox, &bbox)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "copy_from_bbox"); // TODO
        return HPy_NULL;
    }


    CALL_CPP_HPY(ctx, "copy_from_bbox", (reg = self->x->copy_from_bbox(bbox)));

    HPy h_PyBufferRegionType = HPy_GetAttr_s(ctx, m, "BufferRegion");
    h_regobj = PyBufferRegion_new(ctx, h_PyBufferRegionType, NULL, 0, HPy_NULL);
    HPy_Close(ctx, h_PyBufferRegionType);
    PyBufferRegion* regobj = PyBufferRegion_AsStruct(ctx, h_regobj);
    regobj->x = reg;

    return h_regobj;
}

static HPy PyRendererAgg_restore_region(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_regobj;
    HPy m;
    int xx1 = 0, yy1 = 0, xx2 = 0, yy2 = 0, x = 0, y = 0;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OO|iiiiii:restore_region",
                          &h_regobj,
                          &m, /* _backend_agg module */
                          &xx1,
                          &yy1,
                          &xx2,
                          &yy2,
                          &x,
                          &y)) {
        return HPy_NULL;
    }

    HPy h_PyBufferRegionType = HPy_GetAttr_s(ctx, m, "BufferRegion");
    if (!HPy_TypeCheck(ctx, h_regobj, h_PyBufferRegionType)) {
        HPy_Close(ctx, h_PyBufferRegionType);
        HPyErr_SetString(ctx, ctx->h_TypeError, "arg must be BufferRegion"); // TODO
        return HPy_NULL;
    }
    HPy_Close(ctx, h_PyBufferRegionType);

    PyRendererAgg* self = PyRendererAgg_AsStruct(ctx, h_self);
    PyBufferRegion* regobj = PyBufferRegion_AsStruct(ctx, h_regobj);
    if (nargs == 1) {
        CALL_CPP_HPY(ctx, "restore_region", self->x->restore_region(*(regobj->x)));
    } else {
        CALL_CPP_HPY(ctx, "restore_region", self->x->restore_region(*(regobj->x), xx1, yy1, xx2, yy2, x, y));
    }

    return HPy_Dup(ctx, ctx->h_None);
}

HPyDef_SLOT(PyRendererAgg_new_def, PyRendererAgg_new, HPy_tp_new)
HPyDef_SLOT(PyRendererAgg_init_def, PyRendererAgg_init, HPy_tp_init)
HPyDef_SLOT(PyRendererAgg_get_buffer_def, PyRendererAgg_get_buffer, HPy_bf_getbuffer)
HPyDef_SLOT(PyRendererAgg_dealloc_def, PyRendererAgg_dealloc, HPy_tp_destroy)

HPyDef_METH(PyRendererAgg_draw_path_def, "draw_path", PyRendererAgg_draw_path, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_draw_markers_def, "draw_markers", PyRendererAgg_draw_markers, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_draw_text_image_def, "draw_text_image", PyRendererAgg_draw_text_image, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_draw_image_def, "draw_image", PyRendererAgg_draw_image, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_draw_path_collection_def, "draw_path_collection", PyRendererAgg_draw_path_collection, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_draw_quad_mesh_def, "draw_quad_mesh", PyRendererAgg_draw_quad_mesh, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_draw_gouraud_triangle_def, "draw_gouraud_triangle", PyRendererAgg_draw_gouraud_triangle, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_draw_gouraud_triangles_def, "draw_gouraud_triangles", PyRendererAgg_draw_gouraud_triangles, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_clear_def, "clear", PyRendererAgg_clear, HPyFunc_NOARGS)
HPyDef_METH(PyRendererAgg_copy_from_bbox_def, "copy_from_bbox", PyRendererAgg_copy_from_bbox, HPyFunc_VARARGS)
HPyDef_METH(PyRendererAgg_restore_region_def, "restore_region", PyRendererAgg_restore_region, HPyFunc_VARARGS)


HPyDef *PyRendererAgg_defines[] = {
    // slots
    &PyRendererAgg_new_def,
    &PyRendererAgg_init_def,
    &PyRendererAgg_get_buffer_def,
    &PyRendererAgg_dealloc_def,
    
    // methods
    &PyRendererAgg_draw_path_def,
    &PyRendererAgg_draw_markers_def,
    &PyRendererAgg_draw_text_image_def,
    &PyRendererAgg_draw_image_def,
    &PyRendererAgg_draw_path_collection_def,
    &PyRendererAgg_draw_quad_mesh_def,
    &PyRendererAgg_draw_gouraud_triangle_def,
    &PyRendererAgg_draw_gouraud_triangles_def,
    &PyRendererAgg_clear_def,
    &PyRendererAgg_copy_from_bbox_def,
    &PyRendererAgg_restore_region_def,
    NULL
};

HPyType_Spec PyRendererAgg_type_spec = {
    .name = "matplotlib.backends._backend_agg.RendererAgg",
    .basicsize = sizeof(PyRendererAgg),
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_BASETYPE,
    .defines = PyRendererAgg_defines,
};

static HPyModuleDef moduledef = {
    .name = "_backend_agg",
    .doc = NULL,
    .size = 0,
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

HPy_MODINIT(_backend_agg)
static HPy init__backend_agg_impl(HPyContext *ctx)
{
    if (!npy_import_array_hpy(ctx)) {
        return HPy_NULL;
    }
    HPy m = HPyModule_Create(ctx, &moduledef);
    if (HPy_IsNull(m)) {
        return HPy_NULL;
    }

    if (!HPyHelpers_AddType(ctx, m, "RendererAgg", &PyRendererAgg_type_spec, NULL)) {
        HPy_Close(ctx, m);
        return HPy_NULL;
    }

    if (!HPyHelpers_AddType(ctx, m, "BufferRegion", &PyBufferRegion_type_spec, NULL)) {
        HPy_Close(ctx, m);
        return HPy_NULL;
    }

    return m;
}

#pragma GCC visibility pop

#ifdef __cplusplus
}
#endif
