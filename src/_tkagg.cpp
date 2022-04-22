/* -*- mode: c++; c-basic-offset: 4 -*- */
/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

// Where is PIL?
//
// Many years ago, Matplotlib used to include code from PIL (the Python Imaging
// Library).  Since then, the code has changed a lot - the organizing principle
// and methods of operation are now quite different.  Because our review of
// the codebase showed that all the code that came from PIL was removed or
// rewritten, we have removed the PIL licensing information.  If you want PIL,
// you can get it at https://python-pillow.org/

#define PY_SSIZE_T_CLEAN
#include "hpy.h"
#include "hpy_helpers.h"

#ifdef _WIN32
#define WIN32_DLL
#endif
#ifdef __CYGWIN__
/*
 * Unfortunately cygwin's libdl inherits restrictions from the underlying
 * Windows OS, at least currently. Therefore, a symbol may be loaded from a
 * module by dlsym() only if it is really located in the given modile,
 * dependencies are not included. So we have to use native WinAPI on Cygwin
 * also.
 */
#define WIN32_DLL
#endif

#ifdef WIN32_DLL
#include <windows.h>
#define PSAPI_VERSION 1
#include <psapi.h>  // Must be linked with 'psapi' library
#define dlsym GetProcAddress
#else
#include <dlfcn.h>
#endif

// Include our own excerpts from the Tcl / Tk headers
#include "_tkmini.h"

static int convert_voidptr(HPyContext *ctx, HPy obj, void *p)
{
    void **val = (void **)p;
    *val = HPyLong_AsVoidPtr(ctx, obj);
    return *val != NULL ? 1 : !HPyErr_Occurred(ctx);
}

// Global vars for Tk functions.  We load these symbols from the tkinter
// extension module or loaded Tk libraries at run-time.
static Tk_FindPhoto_t TK_FIND_PHOTO;
static Tk_PhotoPutBlock_NoComposite_t TK_PHOTO_PUT_BLOCK_NO_COMPOSITE;

static HPy mpl_tk_blit(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs)
{
    HPy h_interp = HPy_NULL, h_data_ptr = HPy_NULL;
    Tcl_Interp *interp;
    char const *photo_name;
    int height, width;
    unsigned char *data_ptr;
    int o0, o1, o2, o3;
    int x1, x2, y1, y2;
    Tk_PhotoHandle photo;
    Tk_PhotoImageBlock block;
    HPy tuple1, tuple2, tuple3;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OsOOO:blit",
                          &h_interp, &photo_name,
                          &tuple1,
                          &tuple2,
                          &tuple3)) {
        return HPy_NULL;
    }
    int ret;
    Arg_ParseTuple(ret, ctx, tuple1, "iiO:blit", &height, &width, &h_data_ptr)
    h_data_ptr = HPy_Dup(ctx, h_data_ptr); // copy before closing tuple items
    Arg_ParseTupleClose(ctx, tuple1);
    if (!ret) {
        HPy_Close(ctx, h_data_ptr);
        return HPy_NULL;
    }
    if (!convert_voidptr(ctx, h_interp, &interp) || !convert_voidptr(ctx, h_data_ptr, &data_ptr)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "blit"); // TODO
        HPy_Close(ctx, h_data_ptr);
        return HPy_NULL;
    }
    HPy_Close(ctx, h_data_ptr);

    if (!(photo = TK_FIND_PHOTO(interp, photo_name))) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Failed to extract Tk_PhotoHandle");
        return HPy_NULL;
    }

    Arg_ParseTupleAndClose(ret, ctx, tuple2, "iiii:blit", &o0, &o1, &o2, &o3)
    if (!ret) {
        return HPy_NULL;
    }
    Arg_ParseTupleAndClose(ret, ctx, tuple3, "iiii:blit", &x1, &x2, &y1, &y2)
    if (!ret) {
        return HPy_NULL;
    }
    if (0 > y1 || y1 > y2 || y2 > height || 0 > x1 || x1 > x2 || x2 > width) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Attempting to draw out of bounds");
        goto exit;
    }
    block.pixelPtr = data_ptr + 4 * ((height - y2) * width + x1);
    block.width = x2 - x1;
    block.height = y2 - y1;
    block.pitch = 4 * width;
    block.pixelSize = 4;
    block.offset[0] = o0;
    block.offset[1] = o1;
    block.offset[2] = o2;
    block.offset[3] = o3;
    TK_PHOTO_PUT_BLOCK_NO_COMPOSITE(
        photo, &block, x1, height - y2, x2 - x1, y2 - y1);
exit:
    if (HPyErr_Occurred(ctx)) {
        return HPy_NULL;
    } else {
        return HPy_Dup(ctx, ctx->h_None);
    }
}

HPyDef_METH(mpl_tk_blit_def, "blit", mpl_tk_blit, HPyFunc_VARARGS)

static HPyDef *module_defines[] = {
    &mpl_tk_blit_def,
    NULL
};

// Functions to fill global Tk function pointers by dynamic loading

template <class T>
int load_tk(T lib)
{
    // Try to fill Tk global vars with function pointers.  Return the number of
    // functions found.
    return
        !!(TK_FIND_PHOTO =
            (Tk_FindPhoto_t)dlsym(lib, "Tk_FindPhoto")) +
        !!(TK_PHOTO_PUT_BLOCK_NO_COMPOSITE =
            (Tk_PhotoPutBlock_NoComposite_t)dlsym(lib, "Tk_PhotoPutBlock_NoComposite"));
}

#ifdef WIN32_DLL

/*
 * On Windows, we can't load the tkinter module to get the Tk symbols, because
 * Windows does not load symbols into the library name-space of importing
 * modules. So, knowing that tkinter has already been imported by Python, we
 * scan all modules in the running process for the Tk function names.
 */

void load_tkinter_funcs(HPyContext *ctx)
{
    // Load Tk functions by searching all modules in current process.
    HMODULE hMods[1024];
    HANDLE hProcess;
    DWORD cbNeeded;
    unsigned int i;
    // Returns pseudo-handle that does not need to be closed
    hProcess = GetCurrentProcess();
    // Iterate through modules in this process looking for Tk names.
    if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
        for (i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {
            if (load_tk(hMods[i])) {
                return;
            }
        }
    }
}

#else  // not Windows

/*
 * On Unix, we can get the Tk symbols from the tkinter module, because tkinter
 * uses these symbols, and the symbols are therefore visible in the tkinter
 * dynamic library (module).
 */

void load_tkinter_funcs(HPyContext *ctx)
{
    // Load tkinter global funcs from tkinter compiled module.
    void *main_program = NULL, *tkinter_lib = NULL;
    HPy module = HPy_NULL, py_path = HPy_NULL, py_path_b = HPy_NULL;
    char *path;

    // Try loading from the main program namespace first.
    main_program = dlopen(NULL, RTLD_LAZY);
    if (load_tk(main_program)) {
        goto exit;
    }
    // Clear exception triggered when we didn't find symbols above.
    HPyErr_Clear(ctx);

    // Handle PyPy first, as that import will correctly fail on CPython.
    module = HPyImport_ImportModule(ctx, "_tkinter.tklib_cffi");   // PyPy
    if (HPy_IsNull(module)) {
        HPyErr_Clear(ctx);
        module = HPyImport_ImportModule(ctx,"_tkinter");  // CPython
    }
    if (!(!HPy_IsNull(module) &&
          !HPy_IsNull(py_path = HPy_GetAttr_s(ctx, module, "__file__")) &&
          !HPy_IsNull(py_path_b = HPyUnicode_EncodeFSDefault(ctx, py_path)) &&
          (path = HPyBytes_AsString(ctx, py_path_b)))) {
        goto exit;
    }
    tkinter_lib = dlopen(path, RTLD_LAZY);
    if (!tkinter_lib) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, dlerror());
        goto exit;
    }
    if (load_tk(tkinter_lib)) {
        goto exit;
    }

exit:
    // We don't need to keep a reference open as the main program & tkinter
    // have been imported.  Use a non-short-circuiting "or" to try closing both
    // handles before handling errors.
    if ((main_program && dlclose(main_program))
        | (tkinter_lib && dlclose(tkinter_lib))) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, dlerror());
    }
    HPy_Close(ctx, module);
    HPy_Close(ctx, py_path);
    HPy_Close(ctx, py_path_b);
}
#endif // end not Windows

static HPyModuleDef moduledef = {
  .name = "_tkagg",
  .doc = "",
  .size = -1,
  .defines = module_defines,
};

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(default)
HPy_MODINIT(_tkagg)
static HPy init__tkagg_impl(HPyContext *ctx)
{
    load_tkinter_funcs(ctx);
    if (HPyErr_Occurred(ctx)) {
        return HPy_NULL;
    } else if (!TK_FIND_PHOTO) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to load Tk_FindPhoto");
        return HPy_NULL;
    } else if (!TK_PHOTO_PUT_BLOCK_NO_COMPOSITE) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to load Tk_PhotoPutBlock_NoComposite");
        return HPy_NULL;
    }
    return HPyModule_Create(ctx, &moduledef);
}

#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
