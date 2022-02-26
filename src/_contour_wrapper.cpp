#include "_contour.h"
#include "mplutils.h"
#include "py_converters.h"
#include "py_exceptions.h"

/* QuadContourGenerator */

typedef struct
{
    QuadContourGenerator* ptr;
} PyQuadContourGenerator;

HPyType_HELPERS(PyQuadContourGenerator)


static HPy PyQuadContourGenerator_new(HPyContext *ctx, HPy type, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    PyQuadContourGenerator* self;
    HPy h_self = HPy_New(ctx, type, &self);
    self->ptr = NULL;
    return h_self;
}

const char* PyQuadContourGenerator_init__doc__ =
    "QuadContourGenerator(x, y, z, mask, corner_mask, chunk_size)\n"
    "--\n\n"
    "Create a new C++ QuadContourGenerator object\n";

static int PyQuadContourGenerator_init(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    PyQuadContourGenerator* self = PyQuadContourGenerator_AsStruct(ctx, h_self);
    QuadContourGenerator::CoordinateArray x, y, z;
    QuadContourGenerator::MaskArray mask;
    bool corner_mask;
    long chunk_size;
    HPy h_x = HPy_NULL;
    HPy h_y = HPy_NULL;
    HPy h_z = HPy_NULL;
    HPy h_mask = HPy_NULL;
    HPy h_corner_mask = HPy_NULL;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OOOOOl",
                          &h_x,
                          &h_y,
                          &h_z,
                          &h_mask,
                          &h_corner_mask,
                          &chunk_size)) {
        return -1;
    }

    if (!x.converter_contiguous(HPy_AsPyObject(ctx, h_x), &x) ||
            !y.converter_contiguous(HPy_AsPyObject(ctx, h_y), &y) ||
            !z.converter_contiguous(HPy_AsPyObject(ctx, h_z), &z) ||
            !mask.converter_contiguous(HPy_AsPyObject(ctx, h_mask), &mask) ||
            !convert_bool_hpy(ctx, h_corner_mask, &corner_mask)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, ""); // TODO
        return -1;
    }

    if (x.empty() || y.empty() || z.empty() ||
        y.dim(0) != x.dim(0) || z.dim(0) != x.dim(0) ||
        y.dim(1) != x.dim(1) || z.dim(1) != x.dim(1)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "x, y and z must all be 2D arrays with the same dimensions");
        return -1;
    }

    if (z.dim(0) < 2 || z.dim(1) < 2) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "x, y and z must all be at least 2x2 arrays");
        return -1;
    }

    // Mask array is optional, if set must be same size as other arrays.
    if (!mask.empty() && (mask.dim(0) != x.dim(0) || mask.dim(1) != x.dim(1))) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "If mask is set it must be a 2D array with the same dimensions as x.");
        return -1;
    }

    CALL_CPP_INIT_HPY(ctx, "QuadContourGenerator",
                  (self->ptr = new QuadContourGenerator(
                       x, y, z, mask, corner_mask, chunk_size)));
    return 0;
}

static void PyQuadContourGenerator_dealloc(void* obj)
{
    PyQuadContourGenerator* self = (PyQuadContourGenerator*)obj;
    delete self->ptr;
    // Py_TYPE(self)->tp_free((PyObject *)self);
}

const char* PyQuadContourGenerator_create_contour__doc__ =
    "create_contour(level)\n"
    "--\n\n"
    "Create and return a non-filled contour.";

static HPy PyQuadContourGenerator_create_contour(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    PyQuadContourGenerator* self = PyQuadContourGenerator_AsStruct(ctx, h_self);
    double level;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "d:create_contour", &level)) {
        return HPy_NULL;
    }

    HPy result;
    CALL_CPP_HPY(ctx, "create_contour", (result = self->ptr->create_contour(ctx, level)));
    return result;
}

const char* PyQuadContourGenerator_create_filled_contour__doc__ =
    "create_filled_contour(lower_level, upper_level)\n"
    "--\n\n"
    "Create and return a filled contour";

static HPy PyQuadContourGenerator_create_filled_contour(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    PyQuadContourGenerator* self = PyQuadContourGenerator_AsStruct(ctx, h_self);
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

HPyDef_SLOT(PyQuadContourGenerator_new_def, PyQuadContourGenerator_new, HPy_tp_new)
HPyDef_SLOT(PyQuadContourGenerator_init_def, PyQuadContourGenerator_init, HPy_tp_init)
HPyDef_SLOT(PyQuadContourGenerator_dealloc_def, PyQuadContourGenerator_dealloc, HPy_tp_destroy)

HPyDef_METH(PyQuadContourGenerator_create_contour_def, 
                "create_contour", 
                PyQuadContourGenerator_create_contour, 
                HPyFunc_VARARGS, 
                .doc = PyQuadContourGenerator_create_contour__doc__)
HPyDef_METH(PyQuadContourGenerator_create_filled_contour_def, 
                "create_filled_contour", 
                PyQuadContourGenerator_create_filled_contour, 
                HPyFunc_VARARGS, 
                .doc = PyQuadContourGenerator_create_filled_contour__doc__)

HPyDef *PyQuadContourGenerator_defines[] = {
    // slots
    &PyQuadContourGenerator_new_def,
    &PyQuadContourGenerator_init_def,
    &PyQuadContourGenerator_dealloc_def,
    // methods
    &PyQuadContourGenerator_create_contour_def,
    &PyQuadContourGenerator_create_filled_contour_def,
    NULL
};

HPyType_Spec PyQuadContourGenerator_type_spec = {
    .name = "matplotlib.QuadContourGenerator",
    .basicsize = sizeof(PyQuadContourGenerator),
    .flags = HPy_TPFLAGS_DEFAULT,
    .defines = PyQuadContourGenerator_defines,
    .doc = PyQuadContourGenerator_init__doc__,
};

/* Module */

static HPyModuleDef moduledef = {
    .name = "_contour",
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
HPy_MODINIT(_contour)
static HPy init__contour_impl(HPyContext *ctx)
{

    if (!npy_import_array_hpy(ctx)) {
        return HPy_NULL;
    }
    HPy m = HPyModule_Create(ctx, &moduledef);
    if (HPy_IsNull(m)) {
        return HPy_NULL;
    }

    if (!HPyHelpers_AddType(ctx, m, "QuadContourGenerator", &PyQuadContourGenerator_type_spec, NULL)) {
        HPy_Close(ctx, m);
        return HPy_NULL;
    }

    return m;
}

#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
