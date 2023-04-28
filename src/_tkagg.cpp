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
#include <string>
#include <windows.h>
#include <commctrl.h>
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
static Tk_PhotoPutBlock_t TK_PHOTO_PUT_BLOCK;
#ifdef WIN32_DLL
// Global vars for Tcl functions.  We load these symbols from the tkinter
// extension module or loaded Tcl libraries at run-time.
static Tcl_SetVar_t TCL_SETVAR;
#endif

HPyDef_METH(mpl_tk_blit, "blit", HPyFunc_VARARGS)
static HPy mpl_tk_blit_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    HPy h_interp = HPy_NULL, h_data_ptr = HPy_NULL;
    Tcl_Interp *interp;
    char const *photo_name;
    int height, width;
    unsigned char *data_ptr;
    int comp_rule;
    int put_retval;
    int o0, o1, o2, o3;
    int x1, x2, y1, y2;
    Tk_PhotoHandle photo;
    Tk_PhotoImageBlock block;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "Os(iiO)i(iiii)(iiii):blit",
                          &h_interp, &photo_name,
                          &height, &width, &h_data_ptr,
                          &comp_rule,
                          &o0, &o1, &o2, &o3,
                          &x1, &x2, &y1, &y2)) {
        goto exit;
    }

    if (!convert_voidptr(ctx, h_interp, &interp) || !convert_voidptr(ctx, h_interp, &data_ptr)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "blit"); // TODO
        goto exit;
    }
    
    if (!(photo = TK_FIND_PHOTO(interp, photo_name))) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Failed to extract Tk_PhotoHandle");
        goto exit;
    }
    if (0 > y1 || y1 > y2 || y2 > height || 0 > x1 || x1 > x2 || x2 > width) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Attempting to draw out of bounds");
        goto exit;
    }
    if (comp_rule != TK_PHOTO_COMPOSITE_OVERLAY && comp_rule != TK_PHOTO_COMPOSITE_SET) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Invalid comp_rule argument");
        goto exit;
    }

    HPy_BEGIN_LEAVE_PYTHON(ctx);
    block.pixelPtr = data_ptr + 4 * ((height - y2) * width + x1);
    block.width = x2 - x1;
    block.height = y2 - y1;
    block.pitch = 4 * width;
    block.pixelSize = 4;
    block.offset[0] = o0;
    block.offset[1] = o1;
    block.offset[2] = o2;
    block.offset[3] = o3;
    put_retval = TK_PHOTO_PUT_BLOCK(
        interp, photo, &block, x1, height - y2, x2 - x1, y2 - y1, comp_rule);
    HPy_END_LEAVE_PYTHON(ctx);
    if (put_retval == TCL_ERROR) {
        return HPyErr_NoMemory(ctx);
    }

exit:
    if (HPyErr_Occurred(ctx)) {
        return HPy_NULL;
    } else {
        return HPy_Dup(ctx, ctx->h_None);
    }
}

#ifdef WIN32_DLL
LRESULT CALLBACK
DpiSubclassProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam,
                UINT_PTR uIdSubclass, DWORD_PTR dwRefData)
{
    switch (uMsg) {
    case WM_DPICHANGED:
        // This function is a subclassed window procedure, and so is run during
        // the Tcl/Tk event loop. Unfortunately, Tkinter has a *second* lock on
        // Tcl threading that is not exposed publicly, but is currently taken
        // while we're in the window procedure. So while we can take the GIL to
        // call Python code, we must not also call *any* Tk code from Python.
        // So stay with Tcl calls in C only.
        {
            // This variable naming must match the name used in
            // lib/matplotlib/backends/_backend_tk.py:FigureManagerTk.
            std::string var_name("window_dpi");
            var_name += std::to_string((unsigned long long)hwnd);

            // X is high word, Y is low word, but they are always equal.
            std::string dpi = std::to_string(LOWORD(wParam));

            Tcl_Interp* interp = (Tcl_Interp*)dwRefData;
            TCL_SETVAR(interp, var_name.c_str(), dpi.c_str(), 0);
        }
        return 0;
    case WM_NCDESTROY:
        RemoveWindowSubclass(hwnd, DpiSubclassProc, uIdSubclass);
        break;
    }

    return DefSubclassProc(hwnd, uMsg, wParam, lParam);
}
#endif

HPyDef_METH(mpl_tk_enable_dpi_awareness, "enable_dpi_awareness", HPyFunc_VARARGS)
static HPy
mpl_tk_enable_dpi_awareness_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    if (nargs != 2) {
        return HPyErr_Format(ctx, ctx->h_TypeError,
                            "enable_dpi_awareness() takes 2 positional "
                            "arguments but %zd were given",
                            nargs);
    }

#ifdef WIN32_DLL
    HWND frame_handle = NULL;
    Tcl_Interp *interp = NULL;

    if (!convert_voidptr(ctx, args[0], &frame_handle)) {
        return HPy_NULL;
    }
    if (!convert_voidptr(ctx, args[1], &interp)) {
        return HPy_NULL;
    }

#ifdef _DPI_AWARENESS_CONTEXTS_
    HMODULE user32 = LoadLibrary("user32.dll");

    typedef DPI_AWARENESS_CONTEXT (WINAPI *GetWindowDpiAwarenessContext_t)(HWND);
    GetWindowDpiAwarenessContext_t GetWindowDpiAwarenessContextPtr =
        (GetWindowDpiAwarenessContext_t)GetProcAddress(
            user32, "GetWindowDpiAwarenessContext");
    if (GetWindowDpiAwarenessContextPtr == NULL) {
        FreeLibrary(user32);
        return HPy_Dup(ctx, ctx->h_False);
    }

    typedef BOOL (WINAPI *AreDpiAwarenessContextsEqual_t)(DPI_AWARENESS_CONTEXT,
                                                          DPI_AWARENESS_CONTEXT);
    AreDpiAwarenessContextsEqual_t AreDpiAwarenessContextsEqualPtr =
        (AreDpiAwarenessContextsEqual_t)GetProcAddress(
            user32, "AreDpiAwarenessContextsEqual");
    if (AreDpiAwarenessContextsEqualPtr == NULL) {
        FreeLibrary(user32);
        return HPy_Dup(ctx, ctx->h_False);
    }

    DPI_AWARENESS_CONTEXT ctx = GetWindowDpiAwarenessContextPtr(frame_handle);
    bool per_monitor = (
        AreDpiAwarenessContextsEqualPtr(
            ctx, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) ||
        AreDpiAwarenessContextsEqualPtr(
            ctx, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE));

    if (per_monitor) {
        // Per monitor aware means we need to handle WM_DPICHANGED by wrapping
        // the Window Procedure, and the Python side needs to trace the Tk
        // window_dpi variable stored on interp.
        SetWindowSubclass(frame_handle, DpiSubclassProc, 0, (DWORD_PTR)interp);
    }
    FreeLibrary(user32);
    return HPyBool_FromLong(ctx, per_monitor);
#endif
#endif

    return HPy_Dup(ctx, ctx->h_None);
}


// Functions to fill global Tcl/Tk function pointers by dynamic loading.

template <class T>
int load_tk(T lib)
{
    // Try to fill Tk global vars with function pointers. Return the number of
    // functions found.
    return
        !!(TK_FIND_PHOTO =
            (Tk_FindPhoto_t)dlsym(lib, "Tk_FindPhoto")) +
        !!(TK_PHOTO_PUT_BLOCK =
            (Tk_PhotoPutBlock_t)dlsym(lib, "Tk_PhotoPutBlock"));
}

#ifdef WIN32_DLL

template <class T>
int load_tcl(T lib)
{
    // Try to fill Tcl global vars with function pointers. Return the number of
    // functions found.
    return
        !!(TCL_SETVAR = (Tcl_SetVar_t)dlsym(lib, "Tcl_SetVar"));
}

/* On Windows, we can't load the tkinter module to get the Tcl/Tk symbols,
 * because Windows does not load symbols into the library name-space of
 * importing modules. So, knowing that tkinter has already been imported by
 * Python, we scan all modules in the running process for the Tcl/Tk function
 * names.
 */

void load_tkinter_funcs(HPyContext *ctx)
{
    // Load Tcl/Tk functions by searching all modules in current process.
    HMODULE hMods[1024];
    HANDLE hProcess;
    DWORD cbNeeded;
    unsigned int i;
    bool tcl_ok = false, tk_ok = false;
    // Returns pseudo-handle that does not need to be closed
    hProcess = GetCurrentProcess();
    // Iterate through modules in this process looking for Tcl/Tk names.
    if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
        for (i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {
            if (!tcl_ok) {
                tcl_ok = load_tcl(hMods[i]);
            }
            if (!tk_ok) {
                tk_ok = load_tk(hMods[i]);
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
    const char *path;

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

HPyDef_SLOT(_tkagg_hpy_exec, HPy_mod_exec)
static int _tkagg_hpy_exec_impl(HPyContext *ctx, HPy m)
{
    load_tkinter_funcs(ctx);
    if (HPyErr_Occurred(ctx)) {
        return 1;
#ifdef WIN32_DLL
    } else if (!TCL_SETVAR) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to load Tcl_SetVar");
        return 1;
#endif
    } else if (!TK_FIND_PHOTO) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to load Tk_FindPhoto");
        return 1;
    } else if (!TK_PHOTO_PUT_BLOCK) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to load Tk_PhotoPutBlock");
        return 1;
    }
    return 0;
}

static HPyDef *module_defines[] = {
    &_tkagg_hpy_exec,
    &mpl_tk_blit,
    &mpl_tk_enable_dpi_awareness,
    NULL
};

static HPyModuleDef moduledef = {
  .defines = module_defines,
};

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(default)

HPy_MODINIT(_tkagg_hpy, moduledef)

#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
