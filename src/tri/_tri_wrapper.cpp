/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include "_tri.h"
#include "../mplutils.h"
#include "../py_exceptions.h"
#include "../hpy_utils.h"

/* Triangulation */

typedef struct
{
    Triangulation* ptr;
} PyTriangulation;

HPyType_HELPERS(PyTriangulation)

static HPyGlobal PyTriangulationType;


HPyDef_SLOT(PyTriangulation_new, HPy_tp_new)
static HPy PyTriangulation_new_impl(HPyContext *ctx, HPy type, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyTriangulation* self;
    HPy h_self = HPy_New(ctx, type, &self);
    self->ptr = NULL;
    return h_self;
}

static const char PyTriangulation_init__doc__[] =
    "Triangulation(x, y, triangles, mask, edges, neighbors)\n"
    "--\n\n"
    "Create a new C++ Triangulation object\n"
    "This should not be called directly, instead use the python class\n"
    "matplotlib.tri.Triangulation instead.\n";

HPyDef_SLOT(PyTriangulation_init, HPy_tp_init)
static int PyTriangulation_init_impl(HPyContext *ctx, HPy h_self, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyTriangulation* self = PyTriangulation_AsStruct(ctx, h_self);
    HPy h_x, h_y, h_triangles, h_mask, h_edges, h_neighbors;
    Triangulation::CoordinateArray x, y;
    Triangulation::TriangleArray triangles;
    Triangulation::MaskArray mask;
    Triangulation::EdgeArray edges;
    Triangulation::NeighborArray neighbors;
    int correct_triangle_orientations;

    if (!HPyArg_Parse(ctx, NULL, args, nargs,
                          "OOOOOOi",
                          &h_x,
                          &h_y,
                          &h_triangles,
                          &h_mask,
                          &h_edges,
                          &h_neighbors,
                          &correct_triangle_orientations)) {
        return -1;
    }

    if (!x.converter(HPy_AsPyObject(ctx, h_x), &x) ||
            !y.converter(HPy_AsPyObject(ctx, h_y), &y) ||
            !triangles.converter(HPy_AsPyObject(ctx, h_triangles), &triangles) ||
            !mask.converter(HPy_AsPyObject(ctx, h_mask), &mask) ||
            !edges.converter(HPy_AsPyObject(ctx, h_edges), &edges) ||
            !neighbors.converter(HPy_AsPyObject(ctx, h_neighbors), &neighbors)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, ""); // TODO
        return -1;
    }

    // x and y.
    if (x.empty() || y.empty() || x.dim(0) != y.dim(0)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "x and y must be 1D arrays of the same length");
        return -1;
    }

    // triangles.
    if (triangles.empty() || triangles.dim(1) != 3) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "triangles must be a 2D array of shape (?,3)");
        return -1;
    }

    // Optional mask.
    if (!mask.empty() && mask.dim(0) != triangles.dim(0)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "mask must be a 1D array with the same length as the triangles array");
        return -1;
    }

    // Optional edges.
    if (!edges.empty() && edges.dim(1) != 2) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "edges must be a 2D array with shape (?,2)");
        return -1;
    }

    // Optional neighbors.
    if (!neighbors.empty() && (neighbors.dim(0) != triangles.dim(0) ||
                               neighbors.dim(1) != triangles.dim(1))) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "neighbors must be a 2D array with the same shape as the triangles array");
        return -1;
    }

    CALL_CPP_INIT_HPY(ctx, "Triangulation",
                  (self->ptr = new Triangulation(x, y, triangles, mask,
                                                 edges, neighbors,
                                                 correct_triangle_orientations)));
    return 0;
}

HPyDef_SLOT(PyTriangulation_dealloc, HPy_tp_destroy)
static void PyTriangulation_dealloc_impl(void *obj)
{
    PyTriangulation* self = (PyTriangulation*)obj;
    delete self->ptr;
    // Py_TYPE(self)->tp_free((PyObject*)self);
}

static const char PyTriangulation_calculate_plane_coefficients__doc__[] =
    "calculate_plane_coefficients(z, plane_coefficients)\n"
    "--\n\n"
    "Calculate plane equation coefficients for all unmasked triangles";

HPyDef_METH(PyTriangulation_calculate_plane_coefficients, "calculate_plane_coefficients", HPyFunc_KEYWORDS,
            .doc = PyTriangulation_calculate_plane_coefficients__doc__)
static HPy PyTriangulation_calculate_plane_coefficients_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyTriangulation* self = PyTriangulation_AsStruct(ctx, h_self);
    HPy h_z;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O:calculate_plane_coefficients", &h_z)) {
        return HPy_NULL;
    }
    Triangulation::CoordinateArray z;
    if (!z.converter(HPy_AsPyObject(ctx, h_z), &z)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "calculate_plane_coefficients"); // TODO
        return HPy_NULL;
    }

    if (z.empty() || z.dim(0) != self->ptr->get_npoints()) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "z array must have same length as triangulation x and y arrays");
        return HPy_NULL;
    }

    Triangulation::TwoCoordinateArray result;
    CALL_CPP_HPY(ctx, "calculate_plane_coefficients",
             (result = self->ptr->calculate_plane_coefficients(z)));
    return HPy_FromPyObject(ctx, result.pyobj());
}

static const char PyTriangulation_get_edges__doc__[] =
    "get_neighbors()\n"
    "--\n\n"
    "Return neighbors array";

HPyDef_METH(PyTriangulation_get_edges, "get_edges", HPyFunc_NOARGS,
            .doc = PyTriangulation_get_edges__doc__)
static HPy PyTriangulation_get_edges_impl(HPyContext *ctx, HPy h_self)
{
    PyTriangulation* self = PyTriangulation_AsStruct(ctx, h_self);
    Triangulation::EdgeArray* result;
    CALL_CPP_HPY(ctx, "get_edges", (result = &self->ptr->get_edges()));

    if (result->empty()) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    else
        return HPy_FromPyObject(ctx, result->pyobj());
}

HPyDef_METH(PyTriangulation_get_neighbors, "get_neighbors", HPyFunc_NOARGS,
            .doc = "get_neighbors()\n"
                   "--\n\n"
                   "Return neighbors array")
static HPy PyTriangulation_get_neighbors_impl(HPyContext *ctx, HPy h_self)
{
    PyTriangulation* self = PyTriangulation_AsStruct(ctx, h_self);
    Triangulation::NeighborArray* result;
    CALL_CPP_HPY(ctx, "get_neighbors", (result = &self->ptr->get_neighbors()));

    if (result->empty()) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    else
        return HPy_FromPyObject(ctx, result->pyobj());
}

static const char PyTriangulation_set_mask__doc__[] =
    "set_mask(mask)\n"
    "--\n\n"
    "Set or clear the mask array.";

HPyDef_METH(PyTriangulation_set_mask, "set_mask", HPyFunc_KEYWORDS,
            .doc = PyTriangulation_set_mask__doc__)
static HPy PyTriangulation_set_mask_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyTriangulation* self = PyTriangulation_AsStruct(ctx, h_self);
    HPy h_mask;
    Triangulation::MaskArray mask;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O:set_mask", &h_mask)) {
        return HPy_NULL;
    }

    if (!mask.converter(HPy_AsPyObject(ctx, h_mask), &mask)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "set_mask"); // TODO
        return HPy_NULL;
    }

    if (!mask.empty() && mask.dim(0) != self->ptr->get_ntri()) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "mask must be a 1D array with the same length as the triangles array");
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "set_mask", (self->ptr->set_mask(mask)));
    return HPy_Dup(ctx, ctx->h_None);
}

static HPyDef *Triangulation_defines[] = {
    // slots
    &PyTriangulation_new,
    &PyTriangulation_init,
    &PyTriangulation_dealloc,
    
    // methods
    &PyTriangulation_calculate_plane_coefficients,
    &PyTriangulation_get_edges,
    &PyTriangulation_get_neighbors,
    &PyTriangulation_set_mask,
    NULL
};

static HPyType_Spec Triangulation_type_spec = {
    .name = "matplotlib._tri_hpy.Triangulation",
    .basicsize = sizeof(PyTriangulation),
    .flags = HPy_TPFLAGS_DEFAULT,
    .defines = Triangulation_defines,
    .doc = PyTriangulation_init__doc__,
};

/* TriContourGenerator */

typedef struct
{
    TriContourGenerator* ptr;
    PyTriangulation* py_triangulation;
} PyTriContourGenerator;

HPyType_HELPERS(PyTriContourGenerator)

static HPyGlobal PyTriContourGeneratorType;

HPyDef_SLOT(PyTriContourGenerator_new, HPy_tp_new)
static HPy PyTriContourGenerator_new_impl(HPyContext *ctx, HPy type, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyTriContourGenerator* self;
    HPy h_self = HPy_New(ctx, type, &self);
    self->ptr = NULL;
    self->py_triangulation = NULL;
    return h_self;
}

static const char PyTriContourGenerator_init__doc__[] =
    "TriContourGenerator(triangulation, z)\n"
    "--\n\n"
    "Create a new C++ TriContourGenerator object\n"
    "This should not be called directly, instead use the functions\n"
    "matplotlib.axes.tricontour and tricontourf instead.\n";

HPyDef_SLOT(PyTriContourGenerator_init, HPy_tp_init)
static int PyTriContourGenerator_init_impl(HPyContext *ctx, HPy h_self, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyTriContourGenerator* self = PyTriContourGenerator_AsStruct(ctx, h_self);

    HPy triangulation_arg, h_z;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OO",
                          &triangulation_arg,
                          &h_z)) {
        return -1;
    }

    TriContourGenerator::CoordinateArray z;
    if (!HPy_TypeCheck_g(ctx, triangulation_arg, PyTriangulationType)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "arg must be Triangulation"); // TODO
        return -1;
    }
    if (!z.converter(HPy_AsPyObject(ctx, h_z), &z)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, ""); // TODO
        return -1;
    }
    HPy_Dup(ctx, triangulation_arg);

    PyTriangulation* py_triangulation = PyTriangulation_AsStruct(ctx, triangulation_arg);
    self->py_triangulation = py_triangulation;
    Triangulation& triangulation = *(py_triangulation->ptr);

    if (z.empty() || z.dim(0) != triangulation.get_npoints()) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "z must be a 1D array with the same length as the x and y arrays");
        return -1;
    }

    CALL_CPP_INIT_HPY(ctx, "TriContourGenerator",
                  (self->ptr = new TriContourGenerator(triangulation, z)));
    return 0;
}

HPyDef_SLOT(PyTriContourGenerator_dealloc, HPy_tp_destroy)
static void PyTriContourGenerator_dealloc_impl(void *obj)
{
    PyTriContourGenerator* self = (PyTriContourGenerator*)obj;
    delete self->ptr;
    // HPy_Close(ctx, self->py_triangulation);
}

static const char PyTriContourGenerator_create_contour__doc__[] =
    "create_contour(level)\n"
    "\n"
    "Create and return a non-filled contour.";

HPyDef_METH(PyTriContourGenerator_create_contour, "create_contour", HPyFunc_KEYWORDS,
            .doc = PyTriContourGenerator_create_contour__doc__)
static HPy PyTriContourGenerator_create_contour_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyTriContourGenerator* self = PyTriContourGenerator_AsStruct(ctx, h_self);
    double level;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "d:create_contour", &level)) {
        return HPy_NULL;
    }

    HPy result;
    CALL_CPP_HPY(ctx, "create_contour", (result = self->ptr->create_contour(ctx, level)));
    return result;
}

static const char PyTriContourGenerator_create_filled_contour__doc__[] =
    "create_filled_contour(lower_level, upper_level)\n"
    "\n"
    "Create and return a filled contour";

HPyDef_METH(PyTriContourGenerator_create_filled_contour, "create_filled_contour", HPyFunc_KEYWORDS,
            .doc = PyTriContourGenerator_create_filled_contour__doc__)
static HPy PyTriContourGenerator_create_filled_contour_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyTriContourGenerator* self = PyTriContourGenerator_AsStruct(ctx, h_self);
    double lower_level, upper_level;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "dd:create_filled_contour",
                          &lower_level, &upper_level)) {
        return HPy_NULL;
    }

    if (lower_level >= upper_level)
    {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "filled contour levels must be increasing");
        return HPy_NULL;
    }

    HPy result;
    CALL_CPP_HPY(ctx, "create_filled_contour",
             (result = self->ptr->create_filled_contour(ctx, lower_level,
                                                        upper_level)));
    return result;
}

static HPyDef *TriContourGenerator_defines[] = {
    // slots
    &PyTriContourGenerator_new,
    &PyTriContourGenerator_init,
    &PyTriContourGenerator_dealloc,
    
    // methods
    &PyTriContourGenerator_create_contour,
    &PyTriContourGenerator_create_filled_contour,
    NULL
};

static HPyType_Spec TriContourGenerator_type_spec = {
    .name = "matplotlib._tri_hpy.TriContourGenerator",
    .basicsize = sizeof(PyTriContourGenerator),
    .flags = HPy_TPFLAGS_DEFAULT,
    .defines = TriContourGenerator_defines,
    .doc = PyTriContourGenerator_init__doc__,
};


/* TrapezoidMapTriFinder */

typedef struct
{
    TrapezoidMapTriFinder* ptr;
    PyTriangulation* py_triangulation;
} PyTrapezoidMapTriFinder;

HPyType_HELPERS(PyTrapezoidMapTriFinder)

static HPyGlobal PyTrapezoidMapTriFinderType;

HPyDef_SLOT(PyTrapezoidMapTriFinder_new, HPy_tp_new)
static HPy PyTrapezoidMapTriFinder_new_impl(HPyContext *ctx, HPy type, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyTrapezoidMapTriFinder* self;
    HPy h_self = HPy_New(ctx, type, &self);
    self->ptr = NULL;
    self->py_triangulation = NULL;
    return h_self;
}

static const char PyTrapezoidMapTriFinder_init__doc__[] =
    "TrapezoidMapTriFinder(triangulation)\n"
    "--\n\n"
    "Create a new C++ TrapezoidMapTriFinder object\n"
    "This should not be called directly, instead use the python class\n"
    "matplotlib.tri.TrapezoidMapTriFinder instead.\n";

HPyDef_SLOT(PyTrapezoidMapTriFinder_init, HPy_tp_init)
static int PyTrapezoidMapTriFinder_init_impl(HPyContext *ctx, HPy h_self, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyTrapezoidMapTriFinder* self = PyTrapezoidMapTriFinder_AsStruct(ctx, h_self);
    HPy triangulation_arg;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O",
                          &triangulation_arg)) {
        return -1;
    }

    if (!HPy_TypeCheck_g(ctx, triangulation_arg, PyTriangulationType)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "arg must be Triangulation"); // TODO
        return -1;
    }

    PyTriangulation* py_triangulation = PyTriangulation_AsStruct(ctx, triangulation_arg);
    HPy_Dup(ctx, triangulation_arg);
    self->py_triangulation = py_triangulation;
    Triangulation& triangulation = *(py_triangulation->ptr);

    CALL_CPP_INIT_HPY(ctx, "TrapezoidMapTriFinder",
                  (self->ptr = new TrapezoidMapTriFinder(triangulation)));
    return 0;
}

HPyDef_SLOT(PyTrapezoidMapTriFinder_dealloc, HPy_tp_destroy)
static void PyTrapezoidMapTriFinder_dealloc_impl(void *obj)
{
    PyTrapezoidMapTriFinder* self = (PyTrapezoidMapTriFinder*)obj;
    delete self->ptr;
    // HPy_Close(ctx, self->py_triangulation);
}

static const char PyTrapezoidMapTriFinder_find_many__doc__[] =
    "find_many(x, y)\n"
    "\n"
    "Find indices of triangles containing the point coordinates (x, y)";

HPyDef_METH(PyTrapezoidMapTriFinder_find_many, "find_many", HPyFunc_KEYWORDS,
        .doc = PyTrapezoidMapTriFinder_find_many__doc__)
static HPy PyTrapezoidMapTriFinder_find_many_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyTrapezoidMapTriFinder* self = PyTrapezoidMapTriFinder_AsStruct(ctx, h_self);
    HPy h_x, h_y;
    TrapezoidMapTriFinder::CoordinateArray x, y;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OO:find_many",
                          &h_x,
                          &h_y)) {
        return HPy_NULL;
    }
    if (!x.converter(HPy_AsPyObject(ctx, h_x), &x) ||
            !y.converter(HPy_AsPyObject(ctx, h_y), &y)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "find_many"); // TODO
        return HPy_NULL;
    }
    if (x.empty() || y.empty() || x.dim(0) != y.dim(0)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "x and y must be array-like with same shape");
        return HPy_NULL;
    }

    TrapezoidMapTriFinder::TriIndexArray result;
    CALL_CPP_HPY(ctx, "find_many", (result = self->ptr->find_many(x, y)));
    return HPy_FromPyObject(ctx, result.pyobj());
}

static const char PyTrapezoidMapTriFinder_get_tree_stats__doc__[] =
    "get_tree_stats()\n"
    "\n"
    "Return statistics about the tree used by the trapezoid map";

HPyDef_METH(PyTrapezoidMapTriFinder_get_tree_stats, "get_tree_stats", HPyFunc_NOARGS,
            .doc = PyTrapezoidMapTriFinder_get_tree_stats__doc__)
static HPy PyTrapezoidMapTriFinder_get_tree_stats_impl(HPyContext *ctx, HPy h_self)
{
    PyTrapezoidMapTriFinder* self = PyTrapezoidMapTriFinder_AsStruct(ctx, h_self);
    HPy result;
    CALL_CPP_HPY(ctx, "get_tree_stats", (result = self->ptr->get_tree_stats(ctx)));
    return result;
}

static const char PyTrapezoidMapTriFinder_initialize__doc__[] =
    "initialize()\n"
    "\n"
    "Initialize this object, creating the trapezoid map from the triangulation";

HPyDef_METH(PyTrapezoidMapTriFinder_initialize, "initialize", HPyFunc_NOARGS,
            .doc = PyTrapezoidMapTriFinder_initialize__doc__)
static HPy PyTrapezoidMapTriFinder_initialize_impl(HPyContext *ctx, HPy h_self)
{
    PyTrapezoidMapTriFinder* self = PyTrapezoidMapTriFinder_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "initialize", (self->ptr->initialize()));
    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyTrapezoidMapTriFinder_print_tree__doc__[] =
    "print_tree()\n"
    "\n"
    "Print the search tree as text to stdout; useful for debug purposes";

HPyDef_METH(PyTrapezoidMapTriFinder_print_tree, "print_tree", HPyFunc_NOARGS,
            .doc = PyTrapezoidMapTriFinder_print_tree__doc__)
static HPy PyTrapezoidMapTriFinder_print_tree_impl(HPyContext *ctx, HPy h_self)
{
    PyTrapezoidMapTriFinder* self = PyTrapezoidMapTriFinder_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "print_tree", (self->ptr->print_tree()));
    return HPy_Dup(ctx, ctx->h_None);
}



static HPyDef *TrapezoidMapTriFinder_defines[] = {
    // slots
    &PyTrapezoidMapTriFinder_new,
    &PyTrapezoidMapTriFinder_init,
    
    // methods
    &PyTrapezoidMapTriFinder_find_many,
    &PyTrapezoidMapTriFinder_get_tree_stats,
    &PyTrapezoidMapTriFinder_initialize,
    &PyTrapezoidMapTriFinder_print_tree,
    NULL
};

static HPyType_Spec TrapezoidMapTriFinder_type_spec = {
    .name = "matplotlib._tri_hpy.TrapezoidMapTriFinder",
    .basicsize = sizeof(PyTrapezoidMapTriFinder),
    .flags = HPy_TPFLAGS_DEFAULT,
    .defines = TrapezoidMapTriFinder_defines,
    .doc = PyTrapezoidMapTriFinder_init__doc__,
};

/* Module */

// Logic is from NumPy's import_array()
static int npy_import_array_hpy(HPyContext *ctx) {
    if (_import_array() < 0) {
        // HPyErr_Print(ctx); TODO
        HPyErr_SetString(ctx, ctx->h_ImportError, "numpy.core.multiarray failed to import");
        return 0;
    }
    return 1;
}

HPyDef_SLOT(_tri_hpy_exec, HPy_mod_exec)
static int _tri_hpy_exec_impl(HPyContext *ctx, HPy m)
{
    HPy tmp;

    if (!npy_import_array_hpy(ctx)) {
        return 1;
    }

    if (!HPyHelpers_AddType(ctx, m, "Triangulation", &Triangulation_type_spec, NULL)) {
        return 1;
    }

    if (!HPyHelpers_AddType(ctx, m, "TriContourGenerator", &TriContourGenerator_type_spec, NULL)) {
        return 1;
    }

    if (!HPyHelpers_AddType(ctx, m, "TrapezoidMapTriFinder", &TrapezoidMapTriFinder_type_spec, NULL)) {
        return 1;
    }

    tmp = HPy_GetAttr_s(ctx, m, "Triangulation");
    HPyGlobal_Store(ctx, &PyTriangulationType, tmp);
    HPy_Close(ctx, tmp);
    tmp = HPy_GetAttr_s(ctx, m, "TriContourGenerator");
    HPyGlobal_Store(ctx, &PyTriContourGeneratorType, tmp);
    HPy_Close(ctx, tmp);
    tmp = HPy_GetAttr_s(ctx, m, "TrapezoidMapTriFinder");
    HPyGlobal_Store(ctx, &PyTrapezoidMapTriFinderType, tmp);
    HPy_Close(ctx, tmp);

    return 0;
}

static HPyDef *module_defines[] = {
    &_tri_hpy_exec,
    NULL
};

static HPyGlobal *module_globals[] = {
    &PyTriangulationType,
    &PyTriContourGeneratorType,
    &PyTrapezoidMapTriFinderType,
    NULL
};

static HPyModuleDef moduledef = {
    .defines = module_defines,
    .globals = module_globals,
};

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(default)

HPy_MODINIT(_tri_hpy, moduledef)

#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
