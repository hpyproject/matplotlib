/* -*- mode: c++; c-basic-offset: 4 -*- */
/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/*
  _ttconv.c

  Python wrapper for TrueType conversion library in ../ttconv.
 */
#define PY_SSIZE_T_CLEAN
#include "mplutils.h"

#include <cstring>
#include "ttconv/pprdrv.h"
#include "py_exceptions.h"
#include <vector>
#include <cassert>
#include "hpy.h"

/**
 * An implementation of TTStreamWriter that writes to a Python
 * file-like object.
 */
class PythonFileWriter : public TTStreamWriter
{
    HPy _write_method;
    HPyContext *_ctx;

  public:
    PythonFileWriter()
    {
        _write_method = HPy_NULL;
        _ctx = NULL;
    }

    ~PythonFileWriter()
    {
        if (_ctx) {
            HPy_Close(_ctx, _write_method);
            _ctx = NULL;
        }
    }

    void set(HPyContext *ctx, HPy write_method)
    {
        if (_ctx) {
            HPy_Close(_ctx, _write_method);
            _ctx = NULL;
        }
        _write_method = HPy_Dup(ctx, write_method);
        _ctx = ctx;
    }

    virtual void write(const char *a)
    {
        HPy result = HPy_NULL;
        if (!HPy_IsNull(_write_method)) {
            HPy decoded = HPyUnicode_DecodeLatin1(_ctx, a, strlen(a), "");
            if (HPy_IsNull(decoded)) {
                throw py::exception();
            }
            HPy tuple[] = {decoded};
            HPy argtuple = HPyTuple_FromArray(_ctx, tuple, 1);
            result = HPy_CallTupleDict(_ctx, _write_method, argtuple, HPy_NULL);
            HPy_Close(_ctx, decoded);
            HPy_Close(_ctx, argtuple);
            if (HPy_IsNull(result)) {
                throw py::exception();
            }
            HPy_Close(_ctx, result);
        }
    }
};

int fileobject_to_PythonFileWriter(HPyContext *ctx, HPy object, void *address)
{
    PythonFileWriter *file_writer = (PythonFileWriter *)address;

    HPy write_method = HPy_GetAttr_s(ctx, object, "write");
    if (HPy_IsNull(write_method) || !HPyCallable_Check(ctx, write_method)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "Expected a file-like object with a write method.");
        return 0;
    }

    file_writer->set(ctx, write_method);

    return 1;
}

int pyiterable_to_vector_int(HPyContext *ctx, HPy object, void *address)
{
    std::vector<int> *result = (std::vector<int> *)address;

    HPy_ssize_t nentries = HPy_Length(ctx, object);
    HPy item;
    for (HPy_ssize_t i = 0; i < nentries; ++i) {
        item = HPy_GetItem_i(ctx, object, i);
        long value = HPyLong_AsLong(ctx, item);
        HPy_Close(ctx, item);
        if (value == -1 && HPyErr_Occurred(ctx)) {
            return 0;
        }
        result->push_back((int)value);
    }

    return 1;
}

static HPy convert_ttf_to_ps(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    const char *filename;
    PythonFileWriter output;
    int fonttype;
    std::vector<int> glyph_ids;
    HPy h_filename = HPy_NULL;
    HPy h_output = HPy_NULL;
    HPy h_glyph_ids = HPy_NULL;

    HPyTracker ht;
    static const char *kwlist[] = { "filename", "output", "fonttype", "glyph_ids", NULL };
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
                                     kwds,
                                     "OOi|O:convert_ttf_to_ps",
                                     (const char **)kwlist,
                                     &h_filename,
                                     &h_output,
                                     &fonttype,
                                     &h_glyph_ids)) {
        return HPy_NULL;
    }

    if (!HPyBytes_Check(ctx, h_filename)) {
        HPyTracker_Close(ctx, ht);
        HPyErr_SetString(ctx, ctx->h_TypeError, "convert_ttf_to_ps");
        return HPy_NULL;
    }
    filename = HPyBytes_AsString(ctx, h_filename);
    if (!fileobject_to_PythonFileWriter(ctx, h_output, &output) || 
            (!HPy_IsNull(h_glyph_ids) && !pyiterable_to_vector_int(ctx, h_glyph_ids, &glyph_ids))) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "convert_ttf_to_ps"); // TODO
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (fonttype != 3 && fonttype != 42) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "fonttype must be either 3 (raw Postscript) or 42 "
                        "(embedded Truetype)");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    try
    {
        insert_ttfont(filename, output, (font_type_enum)fonttype, glyph_ids);
    }
    catch (TTException &e)
    {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, e.getMessage());
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    catch (const py::exception &)
    {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    catch (...)
    {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Unknown C++ exception");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    HPyTracker_Close(ctx, ht);
    return HPy_Dup(ctx, ctx->h_None);
}


class PythonDictionaryCallback : public TTDictionaryCallback
{
    HPy _dict;
    HPyContext *_ctx;

  public:
    PythonDictionaryCallback(HPyContext *ctx, HPy dict)
    {
        _dict = dict;
        _ctx = ctx;
    }

    virtual void add_pair(const char *a, const char *b)
    {
        assert(a != NULL);
        assert(b != NULL);
        HPy value = HPyBytes_FromString(_ctx, b);
        if (HPy_IsNull(value)) {
            throw py::exception();
        }
        if (HPy_SetItem_s(_ctx, _dict, a, value)) {
            HPy_Close(_ctx, value);
            throw py::exception();
        }
        HPy_Close(_ctx, value);
    }
};

static HPy py_get_pdf_charprocs(HPyContext *ctx, HPy h_self, HPy* args, HPy_ssize_t nargs, HPy kwds)
{
    const char *filename;
    std::vector<int> glyph_ids;
    HPy result, h_glyph_ids = HPy_NULL;

    HPyTracker ht;
    static const char *kwlist[] = { "filename", "glyph_ids", NULL };
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
                                     kwds,
                                     "y|O:get_pdf_charprocs",
                                     (const char **)kwlist,
                                     &filename,
                                     &h_glyph_ids)) {
        return HPy_NULL;
    }

    if (!HPy_IsNull(h_glyph_ids) && !pyiterable_to_vector_int(ctx, h_glyph_ids, &glyph_ids)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, ""); // TODO
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    result = HPyDict_New(ctx);
    if (HPy_IsNull(result)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    PythonDictionaryCallback dict(ctx, result);

    try
    {
        ::get_pdf_charprocs(filename, glyph_ids, dict);
    }
    catch (TTException &e)
    {
        HPy_Close(ctx, result);
        HPyErr_SetString(ctx, ctx->h_RuntimeError, e.getMessage());
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    catch (const py::exception &)
    {
        HPy_Close(ctx, result);
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    catch (...)
    {
        HPy_Close(ctx, result);
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Unknown C++ exception");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    HPyTracker_Close(ctx, ht);
    return result;
}

HPyDef_METH(convert_ttf_to_ps_def, "convert_ttf_to_ps", convert_ttf_to_ps, HPyFunc_KEYWORDS,
.doc = "convert_ttf_to_ps(filename, output, fonttype, glyph_ids)\n"
"\n"
"Converts the Truetype font into a Type 3 or Type 42 Postscript font, "
"optionally subsetting the font to only the desired set of characters.\n"
"\n"
"filename is the path to a TTF font file.\n"
"output is a Python file-like object with a write method that the Postscript "
"font data will be written to.\n"
"fonttype may be either 3 or 42.  Type 3 is a \"raw Postscript\" font. "
"Type 42 is an embedded Truetype font.  Glyph subsetting is not supported "
"for Type 42 fonts.\n"
"glyph_ids (optional) is a list of glyph ids (integers) to keep when "
"subsetting to a Type 3 font.  If glyph_ids is not provided or is None, "
"then all glyphs will be included.  If any of the glyphs specified are "
"composite glyphs, then the component glyphs will also be included.")
HPyDef_METH(py_get_pdf_charprocs_def, "get_pdf_charprocs", py_get_pdf_charprocs, HPyFunc_KEYWORDS,
.doc = "get_pdf_charprocs(filename, glyph_ids)\n"
"\n"
"Given a Truetype font file, returns a dictionary containing the PDF Type 3\n"
"representation of its paths.  Useful for subsetting a Truetype font inside\n"
"of a PDF file.\n"
"\n"
"filename is the path to a TTF font file.\n"
"glyph_ids is a list of the numeric glyph ids to include.\n"
"The return value is a dictionary where the keys are glyph names and\n"
"the values are the stream content needed to render that glyph.  This\n"
"is useful to generate the CharProcs dictionary in a PDF Type 3 font.\n")

static HPyDef *module_defines[] = {
    &convert_ttf_to_ps_def,
    &py_get_pdf_charprocs_def,
    NULL
};

static const char *module_docstring =
    "Module to handle converting and subsetting TrueType "
    "fonts to Postscript Type 3, Postscript Type 42 and "
    "Pdf Type 3 fonts.";

static HPyModuleDef moduledef = {
  .name = "_ttconv",
  .doc = module_docstring,
  .size = -1,
  .defines = module_defines,
};

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(default)
HPy_MODINIT(_ttconv)
static HPy init__ttconv_impl(HPyContext *ctx)
{
    return HPyModule_Create(ctx, &moduledef);
}

#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
