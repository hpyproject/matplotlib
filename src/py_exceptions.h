/* -*- mode: c++; c-basic-offset: 4 -*- */
/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

#ifndef MPL_PY_EXCEPTIONS_H
#define MPL_PY_EXCEPTIONS_H

#include <exception>
#include <stdexcept>

namespace py
{
class exception : public std::exception
{
  public:
    const char *what() const throw()
    {
        return "python error has been set";
    }
};
}

#ifdef HPY
#include "hpy.h"

#define CALL_CPP_FULL_HPY(ctx, name, a, cleanup, errorcode)                           \
    try                                                                      \
    {                                                                        \
        a;                                                                   \
    }                                                                        \
    catch (const py::exception &)                                            \
    {                                                                        \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        /* return (errorcode);                                                */ \
        return HPy_NULL;                                                  \
    }                                                                        \
    catch (const std::bad_alloc &)                                           \
    {                                                                        \
        /* HPyErr_Format(ctx, ctx->h_MemoryError, "In %s: Out of memory", (name)); */ \
        HPyErr_SetString(ctx, ctx->h_MemoryError, "In XX Out of memory");     \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        /* return (errorcode);                                                */ \
        return HPy_NULL;                                                  \
    }                                                                        \
    catch (const std::overflow_error &e)                                     \
    {                                                                        \
        /* HPyErr_Format(ctx, ctx->h_OverflowError, "In %s: %s", (name), e.what());  */ \
        HPyErr_SetString(ctx, ctx->h_OverflowError, e.what());    \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        /* return (errorcode);                                                */ \
        return HPy_NULL;                                                  \
    }                                                                        \
    catch (const std::runtime_error &e)                                      \
    {                                                                        \
        /* HPyErr_Format(ctx, ctx->h_RuntimeError, "In %s: %s", (name), e.what());  */ \
        HPyErr_SetString(ctx, ctx->h_RuntimeError, e.what());    \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        /* return (errorcode);                                                */ \
        return HPy_NULL;                                                  \
    }                                                                        \
    catch (...)                                                              \
    {                                                                        \
        /* HPyErr_Format(ctx, ctx->h_RuntimeError, "Unknown exception in %s", (name)); */ \
        HPyErr_SetString(ctx, ctx->h_RuntimeError, name); \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        /* return (errorcode);                                                */ \
        return HPy_NULL;                                                  \
    }


#define CALL_CPP_FULL_HPY_RET_INT(ctx, name, a, cleanup, errorcode)                           \
    try                                                                      \
    {                                                                        \
        a;                                                                   \
    }                                                                        \
    catch (const py::exception &)                                            \
    {                                                                        \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    catch (const std::bad_alloc &)                                           \
    {                                                                        \
        /* HPyErr_Format(ctx, ctx->h_MemoryError, "In %s: Out of memory", (name)); */ \
        HPyErr_SetString(ctx, ctx->h_MemoryError, "In XX Out of memory");     \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    catch (const std::overflow_error &e)                                     \
    {                                                                        \
        /* HPyErr_Format(ctx, ctx->h_OverflowError, "In %s: %s", (name), e.what());  */ \
        HPyErr_SetString(ctx, ctx->h_OverflowError, e.what());    \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    catch (const std::runtime_error &e)                                      \
    {                                                                        \
        /* HPyErr_Format(ctx, ctx->h_RuntimeError, "In %s: %s", (name), e.what());  */ \
        HPyErr_SetString(ctx, ctx->h_RuntimeError, e.what());    \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    catch (...)                                                              \
    {                                                                        \
        /* HPyErr_Format(ctx, ctx->h_RuntimeError, "Unknown exception in %s", (name)); */ \
        HPyErr_SetString(ctx, ctx->h_RuntimeError, name); \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }

#define CALL_CPP_CLEANUP_HPY(ctx, name, a, cleanup) CALL_CPP_FULL_HPY(ctx, name, a, cleanup, 0)

#define CALL_CPP_HPY(ctx, name, a) CALL_CPP_FULL_HPY(ctx, name, a, , 0)

#define CALL_CPP_INIT_HPY(ctx, name, a) CALL_CPP_FULL_HPY_RET_INT(ctx, name, a, , -1)

#else

#define CALL_CPP_FULL(name, a, cleanup, errorcode)                           \
    try                                                                      \
    {                                                                        \
        a;                                                                   \
    }                                                                        \
    catch (const py::exception &)                                            \
    {                                                                        \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    catch (const std::bad_alloc &)                                           \
    {                                                                        \
        PyErr_Format(PyExc_MemoryError, "In %s: Out of memory", (name));     \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    catch (const std::overflow_error &e)                                     \
    {                                                                        \
        PyErr_Format(PyExc_OverflowError, "In %s: %s", (name), e.what());    \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    catch (const std::runtime_error &e)                                      \
    {                                                                        \
        PyErr_Format(PyExc_RuntimeError, "In %s: %s", (name), e.what());    \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    catch (...)                                                              \
    {                                                                        \
        PyErr_Format(PyExc_RuntimeError, "Unknown exception in %s", (name)); \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }

#define CALL_CPP_CLEANUP(name, a, cleanup) CALL_CPP_FULL(name, a, cleanup, 0)

#define CALL_CPP(name, a) CALL_CPP_FULL(name, a, , 0)

#define CALL_CPP_INIT(name, a) CALL_CPP_FULL(name, a, , -1)

#endif
#endif
