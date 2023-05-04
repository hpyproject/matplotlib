/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include "mplutils.h"
#include "_image_resample.h"
#include "numpy_cpp.h"
#include "py_converters.h"
#include "hpy_utils.h"


/**********************************************************************
 * Free functions
 * */

static const char image_resample__doc__[] =
"resample(input_array, output_array, matrix, interpolation=NEAREST, alpha=1.0, norm=False, radius=1)\n\n"

"Resample input_array, blending it in-place into output_array, using an\n"
"affine transformation.\n\n"

"Parameters\n"
"----------\n"
"input_array : 2-d or 3-d Numpy array of float, double or uint8\n"
"    If 2-d, the image is grayscale.  If 3-d, the image must be of size\n"
"    4 in the last dimension and represents RGBA data.\n\n"

"output_array : 2-d or 3-d Numpy array of float, double or uint8\n"
"    The dtype and number of dimensions must match `input_array`.\n\n"

"transform : matplotlib.transforms.Transform instance\n"
"    The transformation from the input array to the output\n"
"    array.\n\n"

"interpolation : int, optional\n"
"    The interpolation method.  Must be one of the following constants\n"
"    defined in this module:\n\n"

"      NEAREST (default), BILINEAR, BICUBIC, SPLINE16, SPLINE36,\n"
"      HANNING, HAMMING, HERMITE, KAISER, QUADRIC, CATROM, GAUSSIAN,\n"
"      BESSEL, MITCHELL, SINC, LANCZOS, BLACKMAN\n\n"

"resample : bool, optional\n"
"    When `True`, use a full resampling method.  When `False`, only\n"
"    resample when the output image is larger than the input image.\n\n"

"alpha : float, optional\n"
"    The level of transparency to apply.  1.0 is completely opaque.\n"
"    0.0 is completely transparent.\n\n"

"norm : bool, optional\n"
"    Whether to norm the interpolation function.  Default is `False`.\n\n"

"radius: float, optional\n"
"    The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.\n"
"    Default is 1.\n";

static PyArrayObject *
_get_transform_mesh(HPyContext *ctx, HPy py_affine, npy_intp *dims)
{
    /* TODO: Could we get away with float, rather than double, arrays here? */

    /* Given a non-affine transform object, create a mesh that maps
    every pixel in the output image to the input image.  This is used
    as a lookup table during the actual resampling. */

    npy_intp out_dims[3];

    out_dims[0] = dims[0] * dims[1];
    out_dims[1] = 2;

    const HPy args[] = {py_affine};
    HPy py_inverse = HPy_CallMethod_s(ctx, "inverted", args, 0, HPy_NULL);
    if (HPy_IsNull(py_inverse)) {
        return NULL;
    }

    numpy::array_view<double, 2> input_mesh(out_dims);
    double *p = (double *)input_mesh.data();

    for (npy_intp y = 0; y < dims[0]; ++y) {
        for (npy_intp x = 0; x < dims[1]; ++x) {
            *p++ = (double)x;
            *p++ = (double)y;
        }
    }

    HPy h_val = HPy_FromPyObject(ctx, input_mesh.pyobj_steal());
    const HPy transform_args[] = {py_inverse, h_val};
    HPy output_mesh = HPy_CallMethod_s(ctx, "transform", transform_args, 2, HPy_NULL);

    HPy_Close(ctx, h_val);
    HPy_Close(ctx, py_inverse);

    if (HPy_IsNull(output_mesh)) {
        return NULL;
    }

    PyArrayObject *output_mesh_array =
        (PyArrayObject *)PyArray_ContiguousFromAny(
            HPy_AsPyObject(ctx, output_mesh), NPY_DOUBLE, 2, 2);

    HPy_Close(ctx, output_mesh);

    if (output_mesh_array == NULL) {
        return NULL;
    }

    return output_mesh_array;
}


template<class T>
static void
resample(HPyContext *ctx, PyArrayObject* input, PyArrayObject* output, resample_params_t params)
{
    HPy_BEGIN_LEAVE_PYTHON(ctx);
    resample(
        (T*)PyArray_DATA(input), PyArray_DIM(input, 1), PyArray_DIM(input, 0),
        (T*)PyArray_DATA(output), PyArray_DIM(output, 1), PyArray_DIM(output, 0),
        params);
    HPy_END_LEAVE_PYTHON(ctx);
}


HPyDef_METH(image_resample, "resample", HPyFunc_KEYWORDS, .doc=image_resample__doc__)
static HPy
image_resample_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    HPy py_input_array = HPy_NULL;
    HPy py_output_array = HPy_NULL;
    HPy py_transform = HPy_NULL;
    resample_params_t params;

    PyArrayObject *input_array = NULL;
    PyArrayObject *output_array = NULL;
    PyArrayObject *transform_mesh_array = NULL;

    HPy h_resample = HPy_NULL;
    HPy h_norm = HPy_NULL;
    params.interpolation = NEAREST;
    params.transform_mesh = NULL;
    params.resample = false;
    params.norm = false;
    params.radius = 1.0;
    params.alpha = 1.0;

    const char *kwlist[] = {
        "input_array", "output_array", "transform", "interpolation",
        "resample", "alpha", "norm", "radius", NULL };

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
            kwnames,
            "OOO|iOdOd:resample", 
            (const char **)kwlist,
            &py_input_array, 
            &py_output_array, 
            &py_transform,
            &params.interpolation, 
            &h_resample,
            &params.alpha, 
            &h_norm, 
            &params.radius)) {
        goto error;
    }

    if ((!HPy_IsNull(h_resample) && !convert_bool_hpy(ctx, h_resample, &params.resample)) || 
            (!HPy_IsNull(h_norm) && !convert_bool_hpy(ctx, h_norm, &params.norm))) {
        goto error;
    }

    if (params.interpolation < 0 || params.interpolation >= _n_interpolation) {
        HPyErr_Format(ctx, ctx->h_ValueError, "invalid interpolation value %d",
                     params.interpolation);
        goto error;
    }

    input_array = (PyArrayObject *)PyArray_FromAny(
        HPy_AsPyObject(ctx, py_input_array), NULL, 2, 3, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (input_array == NULL) {
        goto error;
    }

    output_array = (PyArrayObject *)HPy_AsPyObject(ctx, py_output_array);
    if (!PyArray_Check(output_array)) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "output array must be a NumPy array");
        goto error;
    }
    if (!PyArray_IS_C_CONTIGUOUS(output_array)) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "output array must be C-contiguous");
        goto error;
    }
    if (PyArray_NDIM(output_array) < 2 || PyArray_NDIM(output_array) > 3) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "output array must be 2- or 3-dimensional");
        goto error;
    }

    if (HPy_IsNull(py_transform) || HPy_Is(ctx, py_transform, ctx->h_None)) {
        params.is_affine = true;
    } else {
        HPy py_is_affine = HPy_GetAttr_s(ctx, py_transform, "is_affine");
        if (HPy_IsNull(py_is_affine)) {
            goto error;
        }

        int py_is_affine2 = HPy_IsTrue(ctx, py_is_affine);
        HPy_Close(ctx, py_is_affine);

        if (py_is_affine2 == -1) {
            goto error;
        } else if (py_is_affine2) {
            if (!convert_trans_affine_hpy(ctx, py_transform, &params.affine)) {
                goto error;
            }
            params.is_affine = true;
        } else {
            transform_mesh_array = _get_transform_mesh(ctx,
                py_transform, PyArray_DIMS(output_array));
            if (transform_mesh_array == NULL) {
                goto error;
            }
            params.transform_mesh = (double *)PyArray_DATA(transform_mesh_array);
            params.is_affine = false;
        }
    }

    if (PyArray_NDIM(input_array) != PyArray_NDIM(output_array)) {
        HPyErr_Format(ctx, ctx->h_ValueError,
            "Mismatched number of dimensions. Got %d and %d.",
            PyArray_NDIM(input_array), PyArray_NDIM(output_array));
        goto error;
    }

    if (PyArray_TYPE(input_array) != PyArray_TYPE(output_array)) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Mismatched types");
        goto error;
    }

    if (PyArray_NDIM(input_array) == 3) {
        if (PyArray_DIM(output_array, 2) != 4) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                "Output array must be RGBA");
            goto error;
        }

        if (PyArray_DIM(input_array, 2) == 4) {
            switch (PyArray_TYPE(input_array)) {
            case NPY_UINT8:
            case NPY_INT8:
                resample<agg::rgba8>(ctx, input_array, output_array, params);
                break;
            case NPY_UINT16:
            case NPY_INT16:
                resample<agg::rgba16>(ctx, input_array, output_array, params);
                break;
            case NPY_FLOAT32:
                resample<agg::rgba32>(ctx, input_array, output_array, params);
                break;
            case NPY_FLOAT64:
                resample<agg::rgba64>(ctx, input_array, output_array, params);
                break;
            default:
                HPyErr_SetString(ctx, ctx->h_ValueError,
                    "3-dimensional arrays must be of dtype unsigned byte, "
                    "unsigned short, float32 or float64");
                goto error;
            }
        } else {
            HPyErr_Format(ctx, ctx->h_ValueError,
                "If 3-dimensional, array must be RGBA.  Got %" NPY_INTP_FMT " planes.",
                PyArray_DIM(input_array, 2));
            goto error;
        }
    } else { // NDIM == 2
        switch (PyArray_TYPE(input_array)) {
        case NPY_DOUBLE:
            resample<double>(ctx, input_array, output_array, params);
            break;
        case NPY_FLOAT:
            resample<float>(ctx, input_array, output_array, params);
            break;
        case NPY_UINT8:
        case NPY_INT8:
            resample<unsigned char>(ctx, input_array, output_array, params);
            break;
        case NPY_UINT16:
        case NPY_INT16:
            resample<unsigned short>(ctx, input_array, output_array, params);
            break;
        default:
            HPyErr_SetString(ctx, ctx->h_ValueError, "Unsupported dtype");
            goto error;
        }
    }

    Py_DECREF(input_array);
    Py_XDECREF(transform_mesh_array);
    HPyTracker_Close(ctx, ht);
    return HPy_Dup(ctx, ctx->h_None);

 error:
    Py_XDECREF(input_array);
    Py_XDECREF(transform_mesh_array);
    HPyTracker_Close(ctx, ht);
    return HPy_NULL;
}


int add_dict_int(HPyContext *ctx, HPy dict, const char *key, long val)
{
    HPy valobj = HPyLong_FromLong(ctx, val);
    if (HPy_IsNull(valobj)) {
        return 1;
    }

    if (HPy_SetAttr_s(ctx, dict, key, valobj)) {
        HPy_Close(ctx, valobj);
        return 1;
    }

    HPy_Close(ctx, valobj);
    return 0;
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

HPyDef_SLOT(_image_hpy_exec, HPy_mod_exec)
static int _image_hpy_exec_impl(HPyContext *ctx, HPy m)
{
    if (!npy_import_array_hpy(ctx)) {
        return 1;
    }

    if (add_dict_int(ctx, m, "NEAREST", NEAREST) ||
        add_dict_int(ctx, m, "BILINEAR", BILINEAR) ||
        add_dict_int(ctx, m, "BICUBIC", BICUBIC) ||
        add_dict_int(ctx, m, "SPLINE16", SPLINE16) ||
        add_dict_int(ctx, m, "SPLINE36", SPLINE36) ||
        add_dict_int(ctx, m, "HANNING", HANNING) ||
        add_dict_int(ctx, m, "HAMMING", HAMMING) ||
        add_dict_int(ctx, m, "HERMITE", HERMITE) ||
        add_dict_int(ctx, m, "KAISER", KAISER) ||
        add_dict_int(ctx, m, "QUADRIC", QUADRIC) ||
        add_dict_int(ctx, m, "CATROM", CATROM) ||
        add_dict_int(ctx, m, "GAUSSIAN", GAUSSIAN) ||
        add_dict_int(ctx, m, "BESSEL", BESSEL) ||
        add_dict_int(ctx, m, "MITCHELL", MITCHELL) ||
        add_dict_int(ctx, m, "SINC", SINC) ||
        add_dict_int(ctx, m, "LANCZOS", LANCZOS) ||
        add_dict_int(ctx, m, "BLACKMAN", BLACKMAN) ||
        add_dict_int(ctx, m, "_n_interpolation", _n_interpolation)) {
        return 1;

    }

    return 0;
}

static HPyDef *module_defines[] = {
    &_image_hpy_exec,
    &image_resample,
    NULL
};

static HPyModuleDef moduledef = {
    .doc = NULL,
    .size = 0,
    .defines = module_defines,
};

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(default)

HPy_MODINIT(_image_hpy, moduledef)

#pragma GCC visibility pop

#ifdef __cplusplus
}
#endif
