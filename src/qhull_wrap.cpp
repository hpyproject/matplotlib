/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
/*
 * Wrapper module for libqhull, providing Delaunay triangulation.
 *
 * This module's methods should not be accessed directly.  To obtain a Delaunay
 * triangulation, construct an instance of the matplotlib.tri.Triangulation
 * class without specifying a triangles array.
 */
#define PY_SSIZE_T_CLEAN
#include "hpy.h"
#include "numpy_cpp.h"
#ifdef _MSC_VER
/* The Qhull header does not declare this as extern "C", but only MSVC seems to
 * do name mangling on global variables. We thus need to declare this before
 * the header so that it treats it correctly, and doesn't mangle the name. */
extern "C" {
extern const char qh_version[];
}
#endif
#include "libqhull_r/qhull_ra.h"
#include <cstdio>
#include <vector>


#ifndef MPL_DEVNULL
#error "MPL_DEVNULL must be defined as the OS-equivalent of /dev/null"
#endif

#define STRINGIFY(x) STR(x)
#define STR(x) #x


static const char* qhull_error_msg[6] = {
    "",                     /* 0 = qh_ERRnone */
    "input inconsistency",  /* 1 = qh_ERRinput */
    "singular input data",  /* 2 = qh_ERRsingular */
    "precision error",      /* 3 = qh_ERRprec */
    "insufficient memory",  /* 4 = qh_ERRmem */
    "internal error"};      /* 5 = qh_ERRqhull */


/* Return the indices of the 3 vertices that comprise the specified facet (i.e.
 * triangle). */
static void
get_facet_vertices(qhT* qh, const facetT* facet, int indices[3])
{
    vertexT *vertex, **vertexp;
    FOREACHvertex_(facet->vertices) {
        *indices++ = qh_pointid(qh, vertex->point);
    }
}

/* Return the indices of the 3 triangles that are neighbors of the specified
 * facet (triangle). */
static void
get_facet_neighbours(const facetT* facet, std::vector<int>& tri_indices,
                     int indices[3])
{
    facetT *neighbor, **neighborp;
    FOREACHneighbor_(facet) {
        *indices++ = (neighbor->upperdelaunay ? -1 : tri_indices[neighbor->id]);
    }
}

/* Return true if the specified points arrays contain at least 3 unique points,
 * or false otherwise. */
static bool
at_least_3_unique_points(npy_intp npoints, const double* x, const double* y)
{
    int i;
    const int unique1 = 0;  /* First unique point has index 0. */
    int unique2 = 0;        /* Second unique point index is 0 until set. */

    if (npoints < 3) {
        return false;
    }

    for (i = 1; i < npoints; ++i) {
        if (unique2 == 0) {
            /* Looking for second unique point. */
            if (x[i] != x[unique1] || y[i] != y[unique1]) {
                unique2 = i;
            }
        }
        else {
            /* Looking for third unique point. */
            if ( (x[i] != x[unique1] || y[i] != y[unique1]) &&
                 (x[i] != x[unique2] || y[i] != y[unique2]) ) {
                /* 3 unique points found, with indices 0, unique2 and i. */
                return true;
            }
        }
    }

    /* Run out of points before 3 unique points found. */
    return false;
}

/* Holds on to info from Qhull so that it can be destructed automatically. */
class QhullInfo {
public:
    QhullInfo(FILE *error_file, qhT* qh) {
        this->error_file = error_file;
        this->qh = qh;
    }

    ~QhullInfo() {
        qh_freeqhull(this->qh, !qh_ALL);
        int curlong, totlong;  /* Memory remaining. */
        qh_memfreeshort(this->qh, &curlong, &totlong);
        if (curlong || totlong) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Qhull could not free all allocated memory", 1);
        }

        if (this->error_file != stderr) {
            fclose(error_file);
        }
    }

private:
    FILE* error_file;
    qhT* qh;
};

/* Delaunay implementation method.
 * If hide_qhull_errors is true then qhull error messages are discarded;
 * if it is false then they are written to stderr. */
static HPy
_delaunay_impl(HPyContext *ctx, npy_intp npoints, const double* x, const double* y,
              bool hide_qhull_errors)
{
    qhT qh_qh;                  /* qh variable type and name must be like */
    qhT* qh = &qh_qh;           /* this for Qhull macros to work correctly. */
    facetT* facet;
    int i, ntri, max_facet_id;
    int exitcode;               /* Value returned from qh_new_qhull(). */
    const int ndim = 2;
    double x_mean = 0.0;
    double y_mean = 0.0;

    QHULL_LIB_CHECK

    /* Allocate points. */
    std::vector<coordT> points(npoints * ndim);

    /* Determine mean x, y coordinates. */
    for (i = 0; i < npoints; ++i) {
        x_mean += x[i];
        y_mean += y[i];
    }
    x_mean /= npoints;
    y_mean /= npoints;

    /* Prepare points array to pass to qhull. */
    for (i = 0; i < npoints; ++i) {
        points[2*i  ] = x[i] - x_mean;
        points[2*i+1] = y[i] - y_mean;
    }

    /* qhull expects a FILE* to write errors to. */
    FILE* error_file = NULL;
    if (hide_qhull_errors) {
        /* qhull errors are ignored by writing to OS-equivalent of /dev/null.
         * Rather than have OS-specific code here, instead it is determined by
         * setupext.py and passed in via the macro MPL_DEVNULL. */
        error_file = fopen(STRINGIFY(MPL_DEVNULL), "w");
        if (error_file == NULL) {
            throw std::runtime_error("Could not open devnull");
        }
    }
    else {
        /* qhull errors written to stderr. */
        error_file = stderr;
    }

    /* Perform Delaunay triangulation. */
    QhullInfo info(error_file, qh);
    qh_zero(qh, error_file);
    exitcode = qh_new_qhull(qh, ndim, (int)npoints, points.data(), False,
                            (char*)"qhull d Qt Qbb Qc Qz", NULL, error_file);
    if (exitcode != qh_ERRnone) {
        HPyErr_Format(ctx, ctx->h_RuntimeError,
                      "Error in qhull Delaunay triangulation calculation: %s (exitcode=%d)%s",
                      qhull_error_msg[exitcode], exitcode,
                      hide_qhull_errors ? "; use python verbose option (-v) to see original qhull error." : "");
        return HPy_NULL;
    }

    /* Split facets so that they only have 3 points each. */
    qh_triangulate(qh);

    /* Determine ntri and max_facet_id.
       Note that libqhull uses macros to iterate through collections. */
    ntri = 0;
    FORALLfacets {
        if (!facet->upperdelaunay) {
            ++ntri;
        }
    }

    max_facet_id = qh->facet_id - 1;

    /* Create array to map facet id to triangle index. */
    std::vector<int> tri_indices(max_facet_id+1);

    /* Allocate Python arrays to return. */
    npy_intp dims[2] = {ntri, 3};
    numpy::array_view<int, ndim> triangles(dims);
    int* triangles_ptr = triangles.data();

    numpy::array_view<int, ndim> neighbors(dims);
    int* neighbors_ptr = neighbors.data();

    /* Determine triangles array and set tri_indices array. */
    i = 0;
    FORALLfacets {
        if (!facet->upperdelaunay) {
            int indices[3];
            tri_indices[facet->id] = i++;
            get_facet_vertices(qh, facet, indices);
            *triangles_ptr++ = (facet->toporient ? indices[0] : indices[2]);
            *triangles_ptr++ = indices[1];
            *triangles_ptr++ = (facet->toporient ? indices[2] : indices[0]);
        }
        else {
            tri_indices[facet->id] = -1;
        }
    }

    /* Determine neighbors array. */
    FORALLfacets {
        if (!facet->upperdelaunay) {
            int indices[3];
            get_facet_neighbours(facet, tri_indices, indices);
            *neighbors_ptr++ = (facet->toporient ? indices[2] : indices[0]);
            *neighbors_ptr++ = (facet->toporient ? indices[0] : indices[2]);
            *neighbors_ptr++ = indices[1];
        }
    }

    HPy tuple[] = {
        HPy_FromPyObject(ctx, (cpy_PyObject *)triangles.pyobj()), 
        HPy_FromPyObject(ctx, (cpy_PyObject *)neighbors.pyobj())
    };
    
    HPy h_tuple = HPyTuple_FromArray(ctx, tuple, 2);
    if (HPy_IsNull(h_tuple)) {
        throw std::runtime_error("Failed to create Python tuple");
    }
    return h_tuple;
}

/* Process Python arguments and call Delaunay implementation method. */
HPyDef_METH(delaunay, "delaunay", HPyFunc_VARARGS, .doc = "")
static HPy
delaunay_impl(HPyContext *ctx, HPy self, const HPy *args, size_t nargs)
{
    HPy xarg = HPy_NULL;
    HPy yarg = HPy_NULL;
    numpy::array_view<double, 1> xarray;
    numpy::array_view<double, 1> yarray;
    HPy ret;
    npy_intp npoints;
    const double* x;
    const double* y;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OO", 
                      &xarg, 
                      &yarg)) {
        return HPy_NULL;
    }

    if (!xarray.converter_contiguous(HPy_AsPyObject(ctx, xarg), &xarray) 
            || !yarray.converter_contiguous(HPy_AsPyObject(ctx, yarg), &yarray)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, ""); // TODO
        return HPy_NULL;
    }

    npoints = xarray.dim(0);
    if (npoints != yarray.dim(0)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "x and y must be 1D arrays of the same length");
        return HPy_NULL;
    }

    if (npoints < 3) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "x and y arrays must have a length of at least 3");
        return HPy_NULL;
    }

    x = xarray.data();
    y = yarray.data();

    if (!at_least_3_unique_points(npoints, x, y)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "x and y arrays must consist of at least 3 unique points");
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "qhull.delaunay",
             (ret = _delaunay_impl(ctx, npoints, x, y, Py_VerboseFlag == 0)));

    return ret;
}

/* Return qhull version string for assistance in debugging. */
HPyDef_METH(version, "version", HPyFunc_NOARGS, .doc = "")
static HPy
version_impl(HPyContext *ctx, HPy module)
{
    return HPyBytes_FromString(ctx, qh_version);
}


// Logic is from NumPy's import_array()
static int npy_import_array_hpy(HPyContext *ctx) {
    if (_import_array() < 0) {
        // HPyErr_Print(ctx); TODO
        HPyErr_SetString(ctx, ctx->h_ImportError, "numpy.core.multiarray failed to import"); 
        return 0; 
    }
    return 1;
}

HPyDef_SLOT(_qhull_hpy_exec, HPy_mod_exec)
static int _qhull_hpy_exec_impl(HPyContext *ctx, HPy m)
{
    if (!npy_import_array_hpy(ctx)) {
        return 1;
    }
    return 0;
}

static HPyDef *module_defines[] = {
    &_qhull_hpy_exec,
    &delaunay,
    &version,
    NULL
};

static HPyModuleDef moduledef = {
    .doc = "Computing Delaunay triangulations.\n",
    .defines = module_defines,
};

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(default)

HPy_MODINIT(_qhull_hpy, moduledef)

#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
