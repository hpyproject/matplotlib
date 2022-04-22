/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#define PY_SSIZE_T_CLEAN
#ifdef __linux__
#include <dlfcn.h>
#endif
#ifdef _WIN32
#include <Objbase.h>
#include <Shobjidl.h>
#include <Windows.h>
#endif
#include "hpy.h"

static HPy
mpl_display_is_valid(HPyContext *ctx, HPy module)
{
#ifdef __linux__
    void* libX11;
    // The getenv check is redundant but helps performance as it is much faster
    // than dlopen().
    if (getenv("DISPLAY")
        && (libX11 = dlopen("libX11.so.6", RTLD_LAZY))) {
        struct Display* display = NULL;
        struct Display* (* XOpenDisplay)(char const*) =
            dlsym(libX11, "XOpenDisplay");
        int (* XCloseDisplay)(struct Display*) =
            dlsym(libX11, "XCloseDisplay");
        if (XOpenDisplay && XCloseDisplay
                && (display = XOpenDisplay(NULL))) {
            XCloseDisplay(display);
        }
        if (dlclose(libX11)) {
            HPyErr_SetString(ctx, ctx->h_RuntimeError, dlerror());
            return HPy_NULL;
        }
        if (display) {
            return HPy_Dup(ctx, ctx->h_True);
        }
    }
    void* libwayland_client;
    if (getenv("WAYLAND_DISPLAY")
        && (libwayland_client = dlopen("libwayland-client.so.0", RTLD_LAZY))) {
        struct wl_display* display = NULL;
        struct wl_display* (* wl_display_connect)(char const*) =
            dlsym(libwayland_client, "wl_display_connect");
        void (* wl_display_disconnect)(struct wl_display*) =
            dlsym(libwayland_client, "wl_display_disconnect");
        if (wl_display_connect && wl_display_disconnect
                && (display = wl_display_connect(NULL))) {
            wl_display_disconnect(display);
        }
        if (dlclose(libwayland_client)) {
            HPyErr_SetString(ctx, ctx->h_RuntimeError, dlerror());
            return HPy_NULL;
        }
        if (display) {
            return HPy_Dup(ctx, ctx->h_True);
        }
    }
    return HPy_Dup(ctx, ctx->h_False);
#else
    Py_RETURN_TRUE;
#endif
}


static HPy
mpl_GetCurrentProcessExplicitAppUserModelID(HPyContext *ctx, HPy module)
{
#ifdef _WIN32
    wchar_t* appid = NULL;
    HRESULT hr = GetCurrentProcessExplicitAppUserModelID(&appid);
    if (FAILED(hr)) {
#if defined(PYPY_VERSION_NUM) && PYPY_VERSION_NUM < 0x07030600
        /* Remove when we require PyPy 7.3.6 */
        PyErr_SetFromWindowsErr(hr); // TODO: add to HPy
        return HPy_NULL;
#else
        // return PyErr_SetFromWindowsErr(hr); // TODO: add to HPy
        return HPy_NULL;
#endif
    }
    HPy py_appid = HPyUnicode_FromWideChar(ctx, appid, -1);
    CoTaskMemFree(appid);
    return py_appid;
#else
    return HPy_Dup(ctx, ctx->h_None);
#endif
}

static HPy
mpl_SetCurrentProcessExplicitAppUserModelID(HPyContext *ctx, HPy module, HPy arg)
{
#ifdef _WIN32
    wchar_t* appid = HPyUnicode_AsWideCharString(ctx, arg, NULL); // TODO: add to HPy
    if (!appid) {
        return HPy_NULL;
    }
    HRESULT hr = SetCurrentProcessExplicitAppUserModelID(appid);
    free(appid); // PyMem_Free is not in HPy
    if (FAILED(hr)) {
#if defined(PYPY_VERSION_NUM) && PYPY_VERSION_NUM < 0x07030600
        /* Remove when we require PyPy 7.3.6 */
        // PyErr_SetFromWindowsErr(hr); TODO: add to HPy
        return HPy_NULL;
#else
        // return PyErr_SetFromWindowsErr(hr); TODO: add to HPy
#endif
    }
    return HPy_Dup(ctx, ctx->h_None);
#else
    return HPy_Dup(ctx, ctx->h_None);
#endif
}

static HPy
mpl_GetForegroundWindow(HPyContext *ctx, HPy module)
{
#ifdef _WIN32
  return HPyLong_FromVoidPtr(GetForegroundWindow()); // TODO: add to HPy
#else
  return HPy_Dup(ctx, ctx->h_None);
#endif
}

static HPy
mpl_SetForegroundWindow(HPyContext *ctx, HPy module, HPy arg)
{
#ifdef _WIN32
  HWND handle = HPyLong_AsVoidPtr(ctx, arg);
  if (HPyErr_Occurred(ctx)) {
    return HPy_NULL;
  }
  if (!SetForegroundWindow(handle)) {
    return HPyErr_Format(ctx, ctx->h_RuntimeError, "Error setting window");
  }
  return HPy_Dup(ctx, ctx->h_None);
#else
  return HPy_Dup(ctx, ctx->h_None);
#endif
}

static HPy
mpl_SetProcessDpiAwareness_max(HPyContext *ctx, HPy module)
{
#ifdef _WIN32
#ifdef _DPI_AWARENESS_CONTEXTS_
    // These functions and options were added in later Windows 10 updates, so
    // must be loaded dynamically.
    typedef BOOL (WINAPI *IsValidDpiAwarenessContext_t)(DPI_AWARENESS_CONTEXT);
    typedef BOOL (WINAPI *SetProcessDpiAwarenessContext_t)(DPI_AWARENESS_CONTEXT);

    HMODULE user32 = LoadLibrary("user32.dll");
    IsValidDpiAwarenessContext_t IsValidDpiAwarenessContextPtr =
        (IsValidDpiAwarenessContext_t)GetProcAddress(
            user32, "IsValidDpiAwarenessContext");
    SetProcessDpiAwarenessContext_t SetProcessDpiAwarenessContextPtr =
        (SetProcessDpiAwarenessContext_t)GetProcAddress(
            user32, "SetProcessDpiAwarenessContext");
    if (IsValidDpiAwarenessContextPtr != NULL && SetProcessDpiAwarenessContextPtr != NULL) {
        if (IsValidDpiAwarenessContextPtr(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
            // Added in Creators Update of Windows 10.
            SetProcessDpiAwarenessContextPtr(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        } else if (IsValidDpiAwarenessContextPtr(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE)) {
            // Added in Windows 10.
            SetProcessDpiAwarenessContextPtr(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE);
        } else if (IsValidDpiAwarenessContextPtr(DPI_AWARENESS_CONTEXT_SYSTEM_AWARE)) {
            // Added in Windows 10.
            SetProcessDpiAwarenessContextPtr(DPI_AWARENESS_CONTEXT_SYSTEM_AWARE);
        }
    } else {
        // Added in Windows Vista.
        SetProcessDPIAware();
    }
    FreeLibrary(user32);
#else
    // Added in Windows Vista.
    SetProcessDPIAware();
#endif
#endif
    return HPy_Dup(ctx, ctx->h_None);
}
HPyDef_METH(mpl_display_is_valid_def, "display_is_valid", mpl_display_is_valid, HPyFunc_NOARGS,
    .doc = "display_is_valid()\n--\n\n"
    "Check whether the current X11 or Wayland display is valid.\n\n"
    "On Linux, returns True if either $DISPLAY is set and XOpenDisplay(NULL)\n"
    "succeeds, or $WAYLAND_DISPLAY is set and wl_display_connect(NULL)\n"
    "succeeds.  On other platforms, always returns True.");
HPyDef_METH(mpl_GetCurrentProcessExplicitAppUserModelID_def, "Win32_GetCurrentProcessExplicitAppUserModelID",
    mpl_GetCurrentProcessExplicitAppUserModelID, HPyFunc_NOARGS,
    .doc = "Win32_GetCurrentProcessExplicitAppUserModelID()\n--\n\n"
    "Wrapper for Windows's GetCurrentProcessExplicitAppUserModelID.  On \n"
    "non-Windows platforms, always returns None.");
HPyDef_METH(mpl_SetCurrentProcessExplicitAppUserModelID_def, "Win32_SetCurrentProcessExplicitAppUserModelID",
    mpl_SetCurrentProcessExplicitAppUserModelID, HPyFunc_O,
    .doc = "Win32_SetCurrentProcessExplicitAppUserModelID(appid, /)\n--\n\n"
    "Wrapper for Windows's SetCurrentProcessExplicitAppUserModelID.  On \n"
    "non-Windows platforms, a no-op.");
HPyDef_METH(mpl_GetForegroundWindow_def, "Win32_GetForegroundWindow",
    mpl_GetForegroundWindow, HPyFunc_NOARGS,
    .doc = "Win32_GetForegroundWindow()\n--\n\n"
    "Wrapper for Windows' GetForegroundWindow.  On non-Windows platforms, \n"
    "always returns None.");
HPyDef_METH(mpl_SetForegroundWindow_def, "Win32_SetForegroundWindow",
    mpl_SetForegroundWindow, HPyFunc_O,
    .doc = "Win32_SetForegroundWindow(hwnd, /)\n--\n\n"
    "Wrapper for Windows' SetForegroundWindow.  On non-Windows platforms, \n"
    "a no-op.");
HPyDef_METH(mpl_SetProcessDpiAwareness_max_def, "Win32_SetProcessDpiAwareness_max",
    mpl_SetProcessDpiAwareness_max, HPyFunc_NOARGS,
    .doc = "Win32_SetProcessDpiAwareness_max()\n--\n\n"
     "Set Windows' process DPI awareness to best option available.\n"
     "On non-Windows platforms, does nothing.");

static HPyDef *module_defines[] = {
    &mpl_display_is_valid_def,
    &mpl_GetCurrentProcessExplicitAppUserModelID_def,
    &mpl_SetCurrentProcessExplicitAppUserModelID_def,
    &mpl_GetForegroundWindow_def,
    &mpl_SetForegroundWindow_def,
    &mpl_SetProcessDpiAwareness_max_def,
    NULL
};

static HPyModuleDef util_module = {
  .name = "_c_internal_utils_hpy",
  .doc = 0,
  .size = -1,
  .defines = module_defines,
};

#pragma GCC visibility push(default)
HPy_MODINIT(_c_internal_utils_hpy)
static HPy init__c_internal_utils_hpy_impl(HPyContext *ctx)
{
  HPy module = HPyModule_Create(ctx, &util_module);
  if (HPy_IsNull(module)) {
      return HPy_NULL;
  }
  return module;
}
