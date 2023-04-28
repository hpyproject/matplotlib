/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include "mplutils.h"
#include "ft2font.h"
#include "py_converters.h"
#include "py_exceptions.h"
#include "numpy_cpp.h"
#include "hpy_utils.h"

// From Python
#include <structmember.h>

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

static HPy convert_xys_to_array(HPyContext *ctx, std::vector<double> &xys)
{
    npy_intp dims[] = {(npy_intp)xys.size() / 2, 2 };
    if (dims[0] > 0) {
        return HPy_FromPyObject(ctx, PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, &xys[0]));
    } else {
        return HPy_FromPyObject(ctx, PyArray_SimpleNew(2, dims, NPY_DOUBLE));
    }
}

/**********************************************************************
 * FT2Image
 * */

typedef struct
{
    FT2Image *x;
    HPy_ssize_t shape[2];
    HPy_ssize_t strides[2];
    HPy_ssize_t suboffsets[2];
} PyFT2Image;

HPyType_HELPERS(PyFT2Image)

HPyDef_SLOT(PyFT2Image_new, HPy_tp_new)
static HPy PyFT2Image_new_impl(HPyContext *ctx, HPy type, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyFT2Image *self;
    HPy h_self = HPy_New(ctx, type, &self);
    self->x = NULL;
    return h_self;
}

HPyDef_SLOT(PyFT2Image_init, HPy_tp_init)
static int PyFT2Image_init_impl(HPyContext *ctx, HPy h_self, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyFT2Image* self = PyFT2Image_AsStruct(ctx, h_self);
    double width;
    double height;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "dd:FT2Image", &width, &height)) {
        return -1;
    }

    CALL_CPP_INIT_HPY(ctx,  "FT2Image", (self->x = new FT2Image(width, height)));

    return 0;
}

HPyDef_SLOT(PyFT2Image_dealloc, HPy_tp_destroy)
static void PyFT2Image_dealloc_impl(void *obj)
{
    PyFT2Image* self = (PyFT2Image*)obj;
    delete self->x;
    // Py_TYPE(self)->tp_free((PyObject *)self);
}

static const char PyFT2Image_draw_rect__doc__[] =
    "draw_rect(x0, y0, x1, y1)\n"
    "--\n\n"
    "Draw an empty rectangle to the image.\n";

HPyDef_METH(PyFT2Image_draw_rect, "draw_path", HPyFunc_VARARGS, .doc = PyFT2Image_draw_rect__doc__)
static HPy PyFT2Image_draw_rect_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Image* self = PyFT2Image_AsStruct(ctx, h_self);
    double x0, y0, x1, y1;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "dddd:draw_rect", &x0, &y0, &x1, &y1)) {
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "draw_rect", (self->x->draw_rect(x0, y0, x1, y1)));

    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyFT2Image_draw_rect_filled__doc__[] =
    "draw_rect_filled(x0, y0, x1, y1)\n"
    "--\n\n"
    "Draw a filled rectangle to the image.\n";

HPyDef_METH(PyFT2Image_draw_rect_filled, "draw_rect_filled", HPyFunc_VARARGS, .doc = PyFT2Image_draw_rect_filled__doc__)
static HPy PyFT2Image_draw_rect_filled_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Image* self = PyFT2Image_AsStruct(ctx, h_self);
    double x0, y0, x1, y1;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "dddd:draw_rect_filled", &x0, &y0, &x1, &y1)) {
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "draw_rect_filled", (self->x->draw_rect_filled(x0, y0, x1, y1)));

    return HPy_Dup(ctx, ctx->h_None);
}

HPyDef_SLOT(PyFT2Image_get_buffer, HPy_bf_getbuffer)
static int PyFT2Image_get_buffer_impl(HPyContext *ctx, HPy h_self, HPy_buffer* buf, int flags)
{
    PyFT2Image* self = PyFT2Image_AsStruct(ctx, h_self);
    HPy_Dup(ctx, h_self);
    FT2Image *im = self->x;

    buf->obj = h_self;
    buf->buf = im->get_buffer();
    buf->len = im->get_width() * im->get_height();
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 2;
    self->shape[0] = im->get_height();
    self->shape[1] = im->get_width();
    buf->shape = self->shape;
    self->strides[0] = im->get_width();
    self->strides[1] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}

static HPyGlobal PyFT2ImageType;


static HPyDef *PyFT2Image_defines[] = {
    // slots
    &PyFT2Image_new,
    &PyFT2Image_init,
    &PyFT2Image_get_buffer,
    &PyFT2Image_dealloc,

    // methods
    &PyFT2Image_draw_rect,
    &PyFT2Image_draw_rect_filled,
    NULL
};

static HPyType_Spec PyFT2Image_type_spec = {
    .name = "matplotlib.ft2font.FT2Image",
    .basicsize = sizeof(PyFT2Image),
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_BASETYPE,
    .defines = PyFT2Image_defines,
};

/**********************************************************************
 * Glyph
 * */

typedef struct
{
    size_t glyphInd;
    long width;
    long height;
    long horiBearingX;
    long horiBearingY;
    long horiAdvance;
    long linearHoriAdvance;
    long vertBearingX;
    long vertBearingY;
    long vertAdvance;
    FT_BBox bbox;
} PyGlyph;

HPyType_HELPERS(PyGlyph)

static HPyGlobal PyGlyphType;

static HPy
PyGlyph_new(HPyContext *ctx, const FT_Face &face, const FT_Glyph &glyph, size_t ind, long hinting_factor)
{
    PyGlyph *self;
    HPy h_PyGlyphType = HPyGlobal_Load(ctx, PyGlyphType);
    HPy h_self = HPy_New(ctx, h_PyGlyphType, &self);
    HPy_Close(ctx, h_PyGlyphType);

    if (HPy_IsNull(h_self))
        return HPy_NULL;

    self->glyphInd = ind;

    FT_Glyph_Get_CBox(glyph, ft_glyph_bbox_subpixels, &self->bbox);

    self->width = face->glyph->metrics.width / hinting_factor;
    self->height = face->glyph->metrics.height;
    self->horiBearingX = face->glyph->metrics.horiBearingX / hinting_factor;
    self->horiBearingY = face->glyph->metrics.horiBearingY;
    self->horiAdvance = face->glyph->metrics.horiAdvance;
    self->linearHoriAdvance = face->glyph->linearHoriAdvance / hinting_factor;
    self->vertBearingX = face->glyph->metrics.vertBearingX;
    self->vertBearingY = face->glyph->metrics.vertBearingY;
    self->vertAdvance = face->glyph->metrics.vertAdvance;

    return h_self;
}

HPyDef_GET(PyGlyph_get_bbox, "bbox")
static HPy PyGlyph_get_bbox_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyGlyph* self = PyGlyph_AsStruct(ctx, h_self);
    return HPy_BuildValue(ctx, 
        "llll", self->bbox.xMin, self->bbox.yMin, self->bbox.xMax, self->bbox.yMax);
}

HPyDef_MEMBER(width_member, "width", HPyMember_LONG, offsetof(PyGlyph, width), .readonly=1, .doc = "")
HPyDef_MEMBER(height_member, "height", HPyMember_LONG, offsetof(PyGlyph, height), .readonly=1, .doc = "")
HPyDef_MEMBER(horiBearingX_member, "horiBearingX", HPyMember_LONG, offsetof(PyGlyph, horiBearingX), .readonly=1, .doc = "")
HPyDef_MEMBER(horiBearingY_member, "horiBearingY", HPyMember_LONG, offsetof(PyGlyph, horiBearingY), .readonly=1, .doc = "")
HPyDef_MEMBER(horiAdvance_member, "horiAdvance", HPyMember_LONG, offsetof(PyGlyph, horiAdvance), .readonly=1, .doc = "")
HPyDef_MEMBER(linearHoriAdvance_member, "linearHoriAdvance", HPyMember_LONG, offsetof(PyGlyph, linearHoriAdvance), .readonly=1, .doc = "")
HPyDef_MEMBER(vertBearingX_member, "vertBearingX", HPyMember_LONG, offsetof(PyGlyph, vertBearingX), .readonly=1, .doc = "")
HPyDef_MEMBER(vertBearingY_member, "vertBearingY", HPyMember_LONG, offsetof(PyGlyph, vertBearingY), .readonly=1, .doc = "")
HPyDef_MEMBER(vertAdvance_member, "vertAdvance", HPyMember_LONG, offsetof(PyGlyph, vertAdvance), .readonly=1, .doc = "")

static HPyDef *PyGlyph_defines[] = {
    // getsets
    &PyGlyph_get_bbox,

    // members
    &width_member,
    &height_member,
    &horiBearingX_member,
    &horiBearingY_member,
    &horiAdvance_member,
    &linearHoriAdvance_member,
    &vertBearingX_member,
    &vertBearingY_member,
    &vertAdvance_member,
    NULL
};

static HPyType_Spec PyGlyph_type_spec = {
    .name = "matplotlib.ft2font.Glyph",
    .basicsize = sizeof(PyGlyph),
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_BASETYPE,
    .defines = PyGlyph_defines,
};

/**********************************************************************
 * FT2Font
 * */

typedef struct
{
    FT2Font *x;
    HPy fname;
    HPy py_file;
    FT_StreamRec stream;
    HPy_ssize_t shape[2];
    HPy_ssize_t strides[2];
    HPy_ssize_t suboffsets[2];
    HPyContext *ctx; // needed for *_file_callback
} PyFT2Font;

HPyType_HELPERS(PyFT2Font)

static unsigned long read_from_file_callback(FT_Stream stream,
                                             unsigned long offset,
                                             unsigned char *buffer,
                                             unsigned long count)
{
    HPyContext *ctx = ((PyFT2Font *)stream->descriptor.pointer)->ctx;
    HPy py_file = ((PyFT2Font *)stream->descriptor.pointer)->py_file;
    HPy seek_result = HPy_NULL, read_result = HPy_NULL;
    HPy_ssize_t n_read = 0;
    if (HPy_IsNull(seek_result = HPyPackLongAndCallMethod(ctx, py_file, "seek", offset))) {
        goto exit;
    }
    if(HPy_IsNull(read_result = HPyPackLongAndCallMethod(ctx, py_file, "read", count))) {
        goto exit;
    }
    
    if (!HPyBytes_Check(ctx, read_result)) {
        goto exit;
    }
    n_read = HPyBytes_Size(ctx, read_result);
    memcpy(buffer, HPyBytes_AsString(ctx, read_result), n_read);
exit:
    HPy_Close(ctx, seek_result);
    HPy_Close(ctx, read_result);
    if (HPyErr_Occurred(ctx)) {
        HPyErr_WriteUnraisable(ctx, py_file);
        if (!count) {
            return 1;  // Non-zero signals error, when count == 0.
        }
    }
    return n_read;
}

static void close_file_callback(FT_Stream stream)
{
    PyFT2Font *self = (PyFT2Font *)stream->descriptor.pointer;
    HPyContext *ctx = self->ctx;
    HPy close_result = HPy_NULL;
    if (HPy_IsNull(close_result = HPy_CallMethod_s(ctx, "close", &self->py_file, 1, HPy_NULL))) {
        goto exit;
    }
exit:
    HPy_Close(ctx, close_result);
    HPy_CLEAR(ctx, self->py_file);
    if (HPyErr_Occurred(ctx)) {
        // PyErr_WriteUnraisable(self); TODO
    }
}

HPyDef_SLOT(PyFT2Font_new, HPy_tp_new)
static HPy PyFT2Font_new_impl(HPyContext *ctx, HPy type, const HPy *args, HPy_ssize_t nargs, HPy kw)
{
    PyFT2Font *self;
    HPy h_self = HPy_New(ctx, type, &self);
    self->x = NULL;
    self->fname = HPy_NULL;
    self->py_file = HPy_NULL;
    self->ctx = NULL;
    memset(&self->stream, 0, sizeof(FT_StreamRec));
    return h_self;
}

static const char PyFT2Font_init__doc__[] =
    "FT2Font(ttffile)\n"
    "--\n\n"
    "Create a new FT2Font object.\n"
    "\n"
    "Attributes\n"
    "----------\n"
    "num_faces\n"
    "    Number of faces in file.\n"
    "face_flags, style_flags : int\n"
    "    Face and style flags; see the ft2font constants.\n"
    "num_glyphs\n"
    "    Number of glyphs in the face.\n"
    "family_name, style_name\n"
    "    Face family and style name.\n"
    "num_fixed_sizes\n"
    "    Number of bitmap in the face.\n"
    "scalable\n"
    "    Whether face is scalable; attributes after this one are only\n"
    "    defined for scalable faces.\n"
    "bbox\n"
    "    Face global bounding box (xmin, ymin, xmax, ymax).\n"
    "units_per_EM\n"
    "    Number of font units covered by the EM.\n"
    "ascender, descender\n"
    "    Ascender and descender in 26.6 units.\n"
    "height\n"
    "    Height in 26.6 units; used to compute a default line spacing\n"
    "    (baseline-to-baseline distance).\n"
    "max_advance_width, max_advance_height\n"
    "    Maximum horizontal and vertical cursor advance for all glyphs.\n"
    "underline_position, underline_thickness\n"
    "    Vertical position and thickness of the underline bar.\n"
    "postscript_name\n"
    "    PostScript name of the font.\n";

HPyDef_SLOT(PyFT2Font_init, HPy_tp_init)
static int PyFT2Font_init_impl(HPyContext *ctx, HPy h_self, const HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    HPy filename = HPy_NULL, data = HPy_NULL;
    FT_Open_Args open_args;
    long hinting_factor = 8;
    int kerning_factor = 0;
    const char *names[] = { "filename", "hinting_factor", "_kerning_factor", NULL };

    HPyTracker ht;
    if (!HPyArg_ParseKeywordsDict(ctx, &ht, args, nargs,
             kwds, "O|l$i:FT2Font", (const char **)names, &filename,
             &hinting_factor, &kerning_factor)) {
        return -1;
    }

    self->ctx = ctx; // needed for *_file_callback

    self->stream.base = NULL;
    self->stream.size = 0x7fffffff;  // Unknown size.
    self->stream.pos = 0;
    self->stream.descriptor.pointer = self;
    self->stream.read = &read_from_file_callback;
    memset((void *)&open_args, 0, sizeof(FT_Open_Args));
    open_args.flags = FT_OPEN_STREAM;
    open_args.stream = &self->stream;

    if (HPyBytes_Check(ctx, filename) || HPyUnicode_Check(ctx, filename)) {
        self->py_file = HPy_Open(ctx, filename, "rb");
        if (HPy_IsNull(self->py_file)) {
            goto exit;
        }
        self->stream.close = &close_file_callback;
    } else if (!HPy_HasAttr_s(ctx, filename, "read")
               || HPy_IsNull(data = HPyPackLongAndCallMethod(ctx, filename, "read", 0))
               || !HPyBytes_Check(ctx, data)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                        "First argument must be a path or binary-mode file object");
        HPy_CLEAR(ctx, data);
        goto exit;
    } else {
        HPy_Dup(ctx, filename);
        self->py_file = filename;
        self->stream.close = NULL;
    }
    HPy_CLEAR(ctx, data);

    CALL_CPP_FULL_HPY_RET_INT(ctx, 
        "FT2Font", (self->x = new FT2Font(open_args, hinting_factor)),
        HPy_CLEAR(ctx, self->py_file), -1);

    CALL_CPP_INIT_HPY(ctx,  "FT2Font->set_kerning_factor", (self->x->set_kerning_factor(kerning_factor)));

    HPy_Dup(ctx, filename);
    self->fname = filename;

exit:
    HPyTracker_Close(ctx, ht);
    return HPyErr_Occurred(ctx) ? -1 : 0;
}

HPyDef_SLOT(PyFT2Font_dealloc, HPy_tp_destroy)
static void PyFT2Font_dealloc_impl(void *obj)
{
    PyFT2Font* self = (PyFT2Font*)obj;
    delete self->x;
    // HPy_Close(ctx, self->py_file);
    // HPy_Close(ctx, self->fname);
    // Py_TYPE(self)->tp_free((PyObject *)self);
}

static const char PyFT2Font_clear__doc__[] =
    "clear()\n"
    "--\n\n"
    "Clear all the glyphs, reset for a new call to `.set_text`.\n";

HPyDef_METH(PyFT2Font_clear, "clear", HPyFunc_NOARGS, .doc = PyFT2Font_clear__doc__)
static HPy PyFT2Font_clear_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "clear", (self->x->clear()));

    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyFT2Font_set_size__doc__[] =
    "set_size(ptsize, dpi)\n"
    "--\n\n"
    "Set the point size and dpi of the text.\n";

HPyDef_METH(PyFT2Font_set_size, "set_size", HPyFunc_VARARGS, .doc = PyFT2Font_set_size__doc__)
static HPy PyFT2Font_set_size_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    double ptsize;
    double dpi;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "dd:set_size", &ptsize, &dpi)) {
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "set_size", (self->x->set_size(ptsize, dpi)));

    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyFT2Font_set_charmap__doc__[] =
    "set_charmap(i)\n"
    "--\n\n"
    "Make the i-th charmap current.\n";

HPyDef_METH(PyFT2Font_set_charmap, "set_charmap", HPyFunc_VARARGS, .doc = PyFT2Font_set_charmap__doc__)
static HPy PyFT2Font_set_charmap_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    int i;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "i:set_charmap", &i)) {
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "set_charmap", (self->x->set_charmap(i)));

    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyFT2Font_select_charmap__doc__[] =
    "select_charmap(i)\n"
    "--\n\n"
    "Select a charmap by its FT_Encoding number.\n";

HPyDef_METH(PyFT2Font_select_charmap, "select_charmap", HPyFunc_VARARGS, .doc = PyFT2Font_select_charmap__doc__)
static HPy PyFT2Font_select_charmap_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    unsigned long i;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "k:select_charmap", &i)) {
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "select_charmap", self->x->select_charmap(i));

    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyFT2Font_get_kerning__doc__[] =
    "get_kerning(left, right, mode)\n"
    "--\n\n"
    "Get the kerning between *left* and *right* glyph indices.\n"
    "*mode* is a kerning mode constant:\n"
    "  KERNING_DEFAULT  - Return scaled and grid-fitted kerning distances\n"
    "  KERNING_UNFITTED - Return scaled but un-grid-fitted kerning distances\n"
    "  KERNING_UNSCALED - Return the kerning vector in original font units\n";

HPyDef_METH(PyFT2Font_get_kerning, "get_kerning", HPyFunc_VARARGS, .doc = PyFT2Font_get_kerning__doc__)
static HPy PyFT2Font_get_kerning_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    FT_UInt left, right, mode;
    int result;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "III:get_kerning", &left, &right, &mode)) {
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "get_kerning", (result = self->x->get_kerning(left, right, mode)));

    return HPyLong_FromLong(ctx, result);
}

static const char PyFT2Font_set_text__doc__[] =
    "set_text(string, angle, flags=32)\n"
    "--\n\n"
    "Set the text *string* and *angle*.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "You must call this before `.draw_glyphs_to_bitmap`.\n"
    "A sequence of x,y positions is returned.\n";

HPyDef_METH(PyFT2Font_set_text, "set_text", HPyFunc_KEYWORDS, .doc = PyFT2Font_set_text__doc__)
static HPy PyFT2Font_set_text_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    HPy textobj;
    double angle = 0.0;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    std::vector<double> xys;
    static const char *names[] = { "string", "angle", "flags", NULL };

    /* This makes a technically incorrect assumption that FT_Int32 is
       int. In theory it can also be long, if the size of int is less
       than 32 bits. This is very unlikely on modern platforms. */
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
             kwnames, "O|di:set_text", names, &textobj, &angle, &flags)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    std::vector<uint32_t> codepoints;
    HPy_ssize_t size;

    if (HPyUnicode_Check(ctx, textobj)) {
        size = HPy_Length(ctx, textobj);
        codepoints.resize(size);
#if defined(PYPY_VERSION) && (PYPY_VERSION_NUM < 0x07040000)
        // PyUnicode_ReadChar is available from PyPy 7.3.2, but wheels do not
        // specify the micro-release version, so put the version bound at 7.4
        // to prevent generating wheels unusable on PyPy 7.3.{0,1}.
        // Py_UNICODE *unistr = PyUnicode_AsUnicode(textobj);
        for (HPy_ssize_t i = 0; i < size; ++i) {
            codepoints[i] = unistr[i];
        }
#else
        for (HPy_ssize_t i = 0; i < size; ++i) {
            codepoints[i] = HPyUnicode_ReadChar(ctx, textobj, i);
        }
#endif
    } else if (HPyBytes_Check(ctx, textobj)) {
        if (HPyErr_WarnEx(ctx, ctx->h_FutureWarning,
            "Passing bytes to FTFont.set_text is deprecated since Matplotlib "
            "3.4 and support will be removed in Matplotlib 3.6; pass str instead",
            1)) {
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        size = HPyBytes_Size(ctx, textobj);
        codepoints.resize(size);
        const char *bytestr = HPyBytes_AsString(ctx, textobj);
        for (HPy_ssize_t i = 0; i < size; ++i) {
            codepoints[i] = bytestr[i];
        }
    } else {
        HPyErr_SetString(ctx, ctx->h_TypeError, "String must be str or bytes");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    uint32_t* codepoints_array = NULL;
    if (size > 0) {
        codepoints_array = &codepoints[0];
    }
    CALL_CPP_HPY(ctx, "set_text", self->x->set_text(ctx, size, codepoints_array, angle, flags, xys));

    HPyTracker_Close(ctx, ht);
    return convert_xys_to_array(ctx, xys);
}

static const char PyFT2Font_get_num_glyphs__doc__[] =
    "get_num_glyphs()\n"
    "--\n\n"
    "Return the number of loaded glyphs.\n";

HPyDef_METH(PyFT2Font_get_num_glyphs, "get_num_glyphs", HPyFunc_NOARGS, .doc = PyFT2Font_get_num_glyphs__doc__)
static HPy PyFT2Font_get_num_glyphs_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_num_glyphs());
}

static const char PyFT2Font_load_char__doc__[] =
    "load_char(charcode, flags=32)\n"
    "--\n\n"
    "Load character with *charcode* in current fontfile and set glyph.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "Return value is a Glyph object, with attributes\n"
    "  width          # glyph width\n"
    "  height         # glyph height\n"
    "  bbox           # the glyph bbox (xmin, ymin, xmax, ymax)\n"
    "  horiBearingX   # left side bearing in horizontal layouts\n"
    "  horiBearingY   # top side bearing in horizontal layouts\n"
    "  horiAdvance    # advance width for horizontal layout\n"
    "  vertBearingX   # left side bearing in vertical layouts\n"
    "  vertBearingY   # top side bearing in vertical layouts\n"
    "  vertAdvance    # advance height for vertical layout\n";

HPyDef_METH(PyFT2Font_load_char, "load_char", HPyFunc_KEYWORDS, .doc = PyFT2Font_load_char__doc__)
static HPy PyFT2Font_load_char_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    long charcode;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    static const char *names[] = { "charcode", "flags", NULL };

    /* This makes a technically incorrect assumption that FT_Int32 is
       int. In theory it can also be long, if the size of int is less
       than 32 bits. This is very unlikely on modern platforms. */
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
             kwnames, "l|i:load_char", names, &charcode, &flags)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "load_char", (self->x->load_char(ctx, charcode, flags)));

    HPyTracker_Close(ctx, ht);
    return PyGlyph_new(ctx, self->x->get_face(),
                       self->x->get_last_glyph(),
                       self->x->get_last_glyph_index(),
                       self->x->get_hinting_factor());
}

static const char PyFT2Font_load_glyph__doc__[] =
    "load_glyph(glyphindex, flags=32)\n"
    "--\n\n"
    "Load character with *glyphindex* in current fontfile and set glyph.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "Return value is a Glyph object, with attributes\n"
    "  width          # glyph width\n"
    "  height         # glyph height\n"
    "  bbox           # the glyph bbox (xmin, ymin, xmax, ymax)\n"
    "  horiBearingX   # left side bearing in horizontal layouts\n"
    "  horiBearingY   # top side bearing in horizontal layouts\n"
    "  horiAdvance    # advance width for horizontal layout\n"
    "  vertBearingX   # left side bearing in vertical layouts\n"
    "  vertBearingY   # top side bearing in vertical layouts\n"
    "  vertAdvance    # advance height for vertical layout\n";

HPyDef_METH(PyFT2Font_load_glyph, "load_glyph", HPyFunc_KEYWORDS, .doc = PyFT2Font_load_glyph__doc__)
static HPy PyFT2Font_load_glyph_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    FT_UInt glyph_index;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    static const char *names[] = { "glyph_index", "flags", NULL };

    /* This makes a technically incorrect assumption that FT_Int32 is
       int. In theory it can also be long, if the size of int is less
       than 32 bits. This is very unlikely on modern platforms. */
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
             kwnames, "I|i:load_glyph", names, &glyph_index, &flags)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "load_glyph", (self->x->load_glyph(glyph_index, flags)));

    HPyTracker_Close(ctx, ht);
    return PyGlyph_new(ctx, self->x->get_face(),
                       self->x->get_last_glyph(),
                       self->x->get_last_glyph_index(),
                       self->x->get_hinting_factor());
}

static const char PyFT2Font_get_width_height__doc__[] =
    "get_width_height()\n"
    "--\n\n"
    "Get the width and height in 26.6 subpixels of the current string set by `.set_text`.\n"
    "The rotation of the string is accounted for.  To get width and height\n"
    "in pixels, divide these values by 64.\n";

HPyDef_METH(PyFT2Font_get_width_height, "get_width_height", HPyFunc_NOARGS, .doc = PyFT2Font_get_width_height__doc__)
static HPy PyFT2Font_get_width_height_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    long width, height;

    CALL_CPP_HPY(ctx, "get_width_height", (self->x->get_width_height(&width, &height)));

    return HPy_BuildValue(ctx, "ll", width, height);
}

static const char PyFT2Font_get_bitmap_offset__doc__[] =
    "get_bitmap_offset()\n"
    "--\n\n"
    "Get the (x, y) offset in 26.6 subpixels for the bitmap if ink hangs left or below (0, 0).\n"
    "Since Matplotlib only supports left-to-right text, y is always 0.\n";

HPyDef_METH(PyFT2Font_get_bitmap_offset, "get_bitmap_offset", HPyFunc_NOARGS, .doc = PyFT2Font_get_bitmap_offset__doc__)
static HPy PyFT2Font_get_bitmap_offset_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    long x, y;

    CALL_CPP_HPY(ctx, "get_bitmap_offset", (self->x->get_bitmap_offset(&x, &y)));

    return HPy_BuildValue(ctx, "ll", x, y);
}

static const char PyFT2Font_get_descent__doc__[] =
    "get_descent()\n"
    "--\n\n"
    "Get the descent in 26.6 subpixels of the current string set by `.set_text`.\n"
    "The rotation of the string is accounted for.  To get the descent\n"
    "in pixels, divide this value by 64.\n";

HPyDef_METH(PyFT2Font_get_descent, "get_descent", HPyFunc_NOARGS, .doc = PyFT2Font_get_descent__doc__)
static HPy PyFT2Font_get_descent_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    long descent;

    CALL_CPP_HPY(ctx, "get_descent", (descent = self->x->get_descent()));

    return HPyLong_FromLong(ctx, descent);
}

static const char PyFT2Font_draw_glyphs_to_bitmap__doc__[] =
    "draw_glyphs_to_bitmap()\n"
    "--\n\n"
    "Draw the glyphs that were loaded by `.set_text` to the bitmap.\n"
    "The bitmap size will be automatically set to include the glyphs.\n";

HPyDef_METH(PyFT2Font_draw_glyphs_to_bitmap, "draw_glyphs_to_bitmap", HPyFunc_KEYWORDS, .doc = PyFT2Font_draw_glyphs_to_bitmap__doc__)
static HPy PyFT2Font_draw_glyphs_to_bitmap_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    HPy h_antialiased = HPy_NULL;
    bool antialiased = true;
    static const char *names[] = { "antialiased", NULL };

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwnames, "|O:draw_glyphs_to_bitmap",
                                     names, &h_antialiased)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!convert_bool_hpy(ctx, h_antialiased, &antialiased)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_glyphs_to_bitmap"); // TODO
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "draw_glyphs_to_bitmap", (self->x->draw_glyphs_to_bitmap(antialiased)));

    HPyTracker_Close(ctx, ht);
    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyFT2Font_get_xys__doc__[] =
    "get_xys()\n"
    "--\n\n"
    "Get the xy locations of the current glyphs.\n";

HPyDef_METH(PyFT2Font_get_xys, "get_xys", HPyFunc_KEYWORDS, .doc = PyFT2Font_get_xys__doc__)
static HPy PyFT2Font_get_xys_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    HPy h_antialiased = HPy_NULL;
    bool antialiased = true;
    std::vector<double> xys;
    static const char *names[] = { "antialiased", NULL };

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwnames, "|O:get_xys",
                                     names, &h_antialiased)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!convert_bool_hpy(ctx, h_antialiased, &antialiased)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "get_xys"); // TODO
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    CALL_CPP_HPY(ctx, "get_xys", (self->x->get_xys(antialiased, xys)));

    HPyTracker_Close(ctx, ht);
    return convert_xys_to_array(ctx, xys);
}


static const char PyFT2Font_draw_glyph_to_bitmap__doc__[] =
    "draw_glyph_to_bitmap(bitmap, x, y, glyph)\n"
    "--\n\n"
    "Draw a single glyph to the bitmap at pixel locations x, y\n"
    "Note it is your responsibility to set up the bitmap manually\n"
    "with ``set_bitmap_size(w, h)`` before this call is made.\n"
    "\n"
    "If you want automatic layout, use `.set_text` in combinations with\n"
    "`.draw_glyphs_to_bitmap`.  This function is instead intended for people\n"
    "who want to render individual glyphs (e.g., returned by `.load_char`)\n"
    "at precise locations.\n";

HPyDef_METH(PyFT2Font_draw_glyph_to_bitmap, "draw_glyph_to_bitmap", HPyFunc_KEYWORDS, .doc = PyFT2Font_draw_glyph_to_bitmap__doc__)
static HPy PyFT2Font_draw_glyph_to_bitmap_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs, HPy kwnames)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    double xd, yd;
    HPy h_image = HPy_NULL, h_glyph = HPy_NULL, h_antialiased = HPy_NULL;
    bool antialiased = true;
    static const char *names[] = { "image", "x", "y", "glyph", "antialiased", NULL };

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs,
                                     kwnames,
                                     "OddO|O:draw_glyph_to_bitmap", // TODO '!'
                                     names,
                                     &h_image,
                                     &xd,
                                     &yd,
                                     &h_glyph,
                                     &h_antialiased)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!HPy_TypeCheck_g(ctx, h_image, PyFT2ImageType)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "arg must be FT2Image"); // TODO
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    } 
    if (!HPy_TypeCheck_g(ctx, h_glyph, PyGlyphType)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "mismatch type Glyph"); // TODO
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    } 
    if(!convert_bool_hpy(ctx, h_antialiased, &antialiased)) {
        if (!HPyErr_Occurred(ctx)) HPyErr_SetString(ctx, ctx->h_SystemError, "draw_glyph_to_bitmap"); // TODO
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    PyFT2Image *image = PyFT2Image_AsStruct(ctx, h_image);
    PyGlyph *glyph = PyGlyph_AsStruct(ctx, h_glyph);
    CALL_CPP_HPY(ctx, "draw_glyph_to_bitmap",
             self->x->draw_glyph_to_bitmap(*(image->x), xd, yd, glyph->glyphInd, antialiased));

    HPyTracker_Close(ctx, ht);
    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyFT2Font_get_glyph_name__doc__[] =
    "get_glyph_name(index)\n"
    "--\n\n"
    "Retrieve the ASCII name of a given glyph *index* in a face.\n"
    "\n"
    "Due to Matplotlib's internal design, for fonts that do not contain glyph\n"
    "names (per FT_FACE_FLAG_GLYPH_NAMES), this returns a made-up name which\n"
    "does *not* roundtrip through `.get_name_index`.\n";

HPyDef_METH(PyFT2Font_get_glyph_name, "get_glyph_name", HPyFunc_VARARGS, .doc = PyFT2Font_get_glyph_name__doc__)
static HPy PyFT2Font_get_glyph_name_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    unsigned int glyph_number;
    char buffer[128];
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "I:get_glyph_name", &glyph_number)) {
        return HPy_NULL;
    }
    CALL_CPP_HPY(ctx, "get_glyph_name", (self->x->get_glyph_name(glyph_number, buffer)));
    return HPyUnicode_FromString(ctx, buffer);
}

static const char PyFT2Font_get_charmap__doc__[] =
    "get_charmap()\n"
    "--\n\n"
    "Return a dict that maps the character codes of the selected charmap\n"
    "(Unicode by default) to their corresponding glyph indices.\n";

HPyDef_METH(PyFT2Font_get_charmap, "get_charmap", HPyFunc_NOARGS, .doc = PyFT2Font_get_charmap__doc__)
static HPy PyFT2Font_get_charmap_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    HPy charmap = HPyDict_New(ctx);
    if (HPy_IsNull(charmap)) {
        return HPy_NULL;
    }
    FT_UInt index;
    FT_ULong code = FT_Get_First_Char(self->x->get_face(), &index);
    while (index != 0) {
        HPy key = HPy_NULL, val = HPy_NULL;
        bool error = (HPy_IsNull(key = HPyLong_FromLong(ctx, code))
                      || HPy_IsNull(val = HPyLong_FromLong(ctx, index))
                      || (HPy_SetItem(ctx, charmap, key, val) == -1));
        HPy_Close(ctx, key);
        HPy_Close(ctx, val);
        if (error) {
            HPy_Close(ctx, charmap);
            return HPy_NULL;
        }
        code = FT_Get_Next_Char(self->x->get_face(), code, &index);
    }
    return charmap;
}


static const char PyFT2Font_get_char_index__doc__[] =
    "get_char_index(codepoint)\n"
    "--\n\n"
    "Return the glyph index corresponding to a character *codepoint*.\n";

HPyDef_METH(PyFT2Font_get_char_index, "get_char_index", HPyFunc_VARARGS, .doc = PyFT2Font_get_char_index__doc__)
static HPy PyFT2Font_get_char_index_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    FT_UInt index;
    FT_ULong ccode;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "k:get_char_index", &ccode)) {
        return HPy_NULL;
    }

    index = FT_Get_Char_Index(self->x->get_face(), ccode);

    return HPyLong_FromLong(ctx, index);
}


static const char PyFT2Font_get_sfnt__doc__[] =
    "get_sfnt()\n"
    "--\n\n"
    "Load the entire SFNT names table, as a dict whose keys are\n"
    "(platform-ID, ISO-encoding-scheme, language-code, and description)\n"
    "tuples.\n";

HPyDef_METH(PyFT2Font_get_sfnt, "get_sfnt", HPyFunc_NOARGS, .doc = PyFT2Font_get_sfnt__doc__)
static HPy PyFT2Font_get_sfnt_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);

    if (!(self->x->get_face()->face_flags & FT_FACE_FLAG_SFNT)) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "No SFNT name table");
        return HPy_NULL;
    }

    size_t count = FT_Get_Sfnt_Name_Count(self->x->get_face());

    HPy names = HPyDict_New(ctx);
    if (HPy_IsNull(names)) {
        HPy_Close(ctx, names);
        return HPy_NULL;
    }

    for (FT_UInt j = 0; j < count; ++j) {
        FT_SfntName sfnt;
        FT_Error error = FT_Get_Sfnt_Name(self->x->get_face(), j, &sfnt);

        if (error) {
            HPy_Close(ctx, names);
            HPyErr_SetString(ctx, ctx->h_ValueError, "Could not get SFNT name");
            return HPy_NULL;
        }

        HPy key = HPy_BuildValue(ctx, 
            "llll", (unsigned int)sfnt.platform_id, (unsigned int)sfnt.encoding_id, 
                    (unsigned int)sfnt.language_id, (unsigned int)sfnt.name_id);
        if (HPy_IsNull(key)) {
            HPy_Close(ctx, names);
            return HPy_NULL;
        }

        HPy val = HPyBytes_FromStringAndSize(ctx, (const char *)sfnt.string, sfnt.string_len);
        if (HPy_IsNull(val)) {
            HPy_Close(ctx, key);
            HPy_Close(ctx, names);
            return HPy_NULL;
        }

        if (HPy_SetItem(ctx, names, key, val)) {
            HPy_Close(ctx, key);
            HPy_Close(ctx, val);
            HPy_Close(ctx, names);
            return HPy_NULL;
        }

        HPy_Close(ctx, key);
        HPy_Close(ctx, val);
    }

    return names;
}

static const char PyFT2Font_get_name_index__doc__[] =
    "get_name_index(name)\n"
    "--\n\n"
    "Return the glyph index of a given glyph *name*.\n"
    "The glyph index 0 means 'undefined character code'.\n";

HPyDef_METH(PyFT2Font_get_name_index, "get_name_index", HPyFunc_VARARGS, .doc = PyFT2Font_get_name_index__doc__)
static HPy PyFT2Font_get_name_index_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    char *glyphname;
    long name_index;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "s:get_name_index", &glyphname)) {
        return HPy_NULL;
    }
    CALL_CPP_HPY(ctx, "get_name_index", name_index = self->x->get_name_index(glyphname));
    return HPyLong_FromLong(ctx, name_index);
}

static HPy build_str(HPyContext *ctx, const char *v, HPy_ssize_t len){
    if (v) {
        return HPyBytes_FromStringAndSize(ctx, v, len);
    }
    return HPy_Dup(ctx, ctx->h_None);
}

static const char PyFT2Font_get_ps_font_info__doc__[] =
    "get_ps_font_info()\n"
    "--\n\n"
    "Return the information in the PS Font Info structure.\n";

HPyDef_METH(PyFT2Font_get_ps_font_info, "get_ps_font_info", HPyFunc_NOARGS, .doc = PyFT2Font_get_ps_font_info__doc__)
static HPy PyFT2Font_get_ps_font_info_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    PS_FontInfoRec fontinfo;

    FT_Error error = FT_Get_PS_Font_Info(self->x->get_face(), &fontinfo);
    if (error) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Could not get PS font info");
        return HPy_NULL;
    }

    return HPy_BuildValue(ctx, "sssssliil",
                         fontinfo.version ? fontinfo.version : "",
                         fontinfo.notice ? fontinfo.notice : "",
                         fontinfo.full_name ? fontinfo.full_name : "",
                         fontinfo.family_name ? fontinfo.family_name : "",
                         fontinfo.weight ? fontinfo.weight : "",
                         fontinfo.italic_angle,
                         fontinfo.is_fixed_pitch,
                         fontinfo.underline_position,
                         (unsigned int)fontinfo.underline_thickness);
}

static const char PyFT2Font_get_sfnt_table__doc__[] =
    "get_sfnt_table(name)\n"
    "--\n\n"
    "Return one of the following SFNT tables: head, maxp, OS/2, hhea, "
    "vhea, post, or pclt.\n";

HPyDef_METH(PyFT2Font_get_sfnt_table, "get_sfnt_table", HPyFunc_VARARGS, .doc = PyFT2Font_get_sfnt_table__doc__)
static HPy PyFT2Font_get_sfnt_table_impl(HPyContext *ctx, HPy h_self, const HPy *args, size_t nargs)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    char *tagname;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "s:get_sfnt_table", &tagname)) {
        return HPy_NULL;
    }

    int tag;
    const char *tags[] = { "head", "maxp", "OS/2", "hhea", "vhea", "post", "pclt", NULL };

    for (tag = 0; tags[tag] != NULL; tag++) {
        if (strncmp(tagname, tags[tag], 5) == 0) {
            break;
        }
    }

    void *table = FT_Get_Sfnt_Table(self->x->get_face(), (FT_Sfnt_Tag)tag);
    if (!table) {
        return HPy_Dup(ctx, ctx->h_None);
    }

    switch (tag) {
    case 0: {
        char head_dict[] =
            "{s:(i,i), s:(i,i), s:l, s:l, s:i, s:i,"
            "s:(l,l), s:(l,l), s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i}";
        TT_Header *t = (TT_Header *)table;
        return HPy_BuildValue(ctx, head_dict,
                             "version",
                             (int)FIXED_MAJOR(t->Table_Version),
                             (int)FIXED_MINOR(t->Table_Version),
                             "fontRevision",
                             (int)FIXED_MAJOR(t->Font_Revision),
                             (int)FIXED_MINOR(t->Font_Revision),
                             "checkSumAdjustment",
                             t->CheckSum_Adjust,
                             "magicNumber",
                             t->Magic_Number,
                             "flags",
                             (int)t->Flags,
                             "unitsPerEm",
                             (int)t->Units_Per_EM,
                             "created",
                             t->Created[0],
                             t->Created[1],
                             "modified",
                             t->Modified[0],
                             t->Modified[1],
                             "xMin",
                             (int)t->xMin,
                             "yMin",
                             (int)t->yMin,
                             "xMax",
                             (int)t->xMax,
                             "yMax",
                             (int)t->yMax,
                             "macStyle",
                             (int)t->Mac_Style,
                             "lowestRecPPEM",
                             (int)t->Lowest_Rec_PPEM,
                             "fontDirectionHint",
                             (int)t->Font_Direction,
                             "indexToLocFormat",
                             (int)t->Index_To_Loc_Format,
                             "glyphDataFormat",
                             (int)t->Glyph_Data_Format);
    }
    case 1: {
        char maxp_dict[] =
            "{s:(i,i), s:i, s:i, s:i, s:i, s:i, s:i,"
            "s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i}";
        TT_MaxProfile *t = (TT_MaxProfile *)table;
        return HPy_BuildValue(ctx, maxp_dict,
                             "version",
                             (int)FIXED_MAJOR(t->version),
                             (int)FIXED_MINOR(t->version),
                             "numGlyphs",
                             (int)t->numGlyphs,
                             "maxPoints",
                             (int)t->maxPoints,
                             "maxContours",
                             (int)t->maxContours,
                             "maxComponentPoints",
                             (int)t->maxCompositePoints,
                             "maxComponentContours",
                             (int)t->maxCompositeContours,
                             "maxZones",
                             (int)t->maxZones,
                             "maxTwilightPoints",
                             (int)t->maxTwilightPoints,
                             "maxStorage",
                             (int)t->maxStorage,
                             "maxFunctionDefs",
                             (int)t->maxFunctionDefs,
                             "maxInstructionDefs",
                             (int)t->maxInstructionDefs,
                             "maxStackElements",
                             (int)t->maxStackElements,
                             "maxSizeOfInstructions",
                             (int)t->maxSizeOfInstructions,
                             "maxComponentElements",
                             (int)t->maxComponentElements,
                             "maxComponentDepth",
                             (int)t->maxComponentDepth);
    }
    case 2: {
        char os_2_dict[] =
            "{s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i,"
            "s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:O, s:(kkkk),"
            "s:O, s:i, s:i, s:i}";
        TT_OS2 *t = (TT_OS2 *)table;
        return HPy_BuildValue(ctx, os_2_dict,
                             "version",
                             (int)t->version,
                             "xAvgCharWidth",
                             (int)t->xAvgCharWidth,
                             "usWeightClass",
                             (int)t->usWeightClass,
                             "usWidthClass",
                             (int)t->usWidthClass,
                             "fsType",
                             (int)t->fsType,
                             "ySubscriptXSize",
                             (int)t->ySubscriptXSize,
                             "ySubscriptYSize",
                             (int)t->ySubscriptYSize,
                             "ySubscriptXOffset",
                             (int)t->ySubscriptXOffset,
                             "ySubscriptYOffset",
                             (int)t->ySubscriptYOffset,
                             "ySuperscriptXSize",
                             (int)t->ySuperscriptXSize,
                             "ySuperscriptYSize",
                             (int)t->ySuperscriptYSize,
                             "ySuperscriptXOffset",
                             (int)t->ySuperscriptXOffset,
                             "ySuperscriptYOffset",
                             (int)t->ySuperscriptYOffset,
                             "yStrikeoutSize",
                             (int)t->yStrikeoutSize,
                             "yStrikeoutPosition",
                             (int)t->yStrikeoutPosition,
                             "sFamilyClass",
                             (int)t->sFamilyClass,
                             "panose",
                             build_str(ctx, (const char *)t->panose,
                             HPy_ssize_t(10)),
                             "ulCharRange",
                             t->ulUnicodeRange1,
                             t->ulUnicodeRange2,
                             t->ulUnicodeRange3,
                             t->ulUnicodeRange4,
                             "achVendID",
                             build_str(ctx, (const char *)t->achVendID,
                             HPy_ssize_t(4)),
                             "fsSelection",
                             (int)t->fsSelection,
                             "fsFirstCharIndex",
                             (int)t->usFirstCharIndex,
                             "fsLastCharIndex",
                             (int)t->usLastCharIndex);
    }
    case 3: {
        char hhea_dict[] =
            "{s:(i,i), s:i, s:i, s:i, s:i, s:i, s:i, s:i,"
            "s:i, s:i, s:i, s:i, s:i}";
        TT_HoriHeader *t = (TT_HoriHeader *)table;
        return HPy_BuildValue(ctx, hhea_dict,
                             "version",
                             (int)FIXED_MAJOR(t->Version),
                             (int)FIXED_MINOR(t->Version),
                             "ascent",
                             (int)t->Ascender,
                             "descent",
                             (int)t->Descender,
                             "lineGap",
                             (int)t->Line_Gap,
                             "advanceWidthMax",
                             (int)t->advance_Width_Max,
                             "minLeftBearing",
                             (int)t->min_Left_Side_Bearing,
                             "minRightBearing",
                             (int)t->min_Right_Side_Bearing,
                             "xMaxExtent",
                             (int)t->xMax_Extent,
                             "caretSlopeRise",
                             (int)t->caret_Slope_Rise,
                             "caretSlopeRun",
                             (int)t->caret_Slope_Run,
                             "caretOffset",
                             (int)t->caret_Offset,
                             "metricDataFormat",
                             (int)t->metric_Data_Format,
                             "numOfLongHorMetrics",
                             (int)t->number_Of_HMetrics);
    }
    case 4: {
        char vhea_dict[] =
            "{s:(i,i), s:i, s:i, s:i, s:i, s:i, s:i, s:i,"
            "s:i, s:i, s:i, s:i, s:i}";
        TT_VertHeader *t = (TT_VertHeader *)table;
        return HPy_BuildValue(ctx, vhea_dict,
                             "version",
                             (int)FIXED_MAJOR(t->Version),
                             (int)FIXED_MINOR(t->Version),
                             "vertTypoAscender",
                             (int)t->Ascender,
                             "vertTypoDescender",
                             (int)t->Descender,
                             "vertTypoLineGap",
                             (int)t->Line_Gap,
                             "advanceHeightMax",
                             (int)t->advance_Height_Max,
                             "minTopSideBearing",
                             (int)t->min_Top_Side_Bearing,
                             "minBottomSizeBearing",
                             (int)t->min_Bottom_Side_Bearing,
                             "yMaxExtent",
                             (int)t->yMax_Extent,
                             "caretSlopeRise",
                             (int)t->caret_Slope_Rise,
                             "caretSlopeRun",
                             (int)t->caret_Slope_Run,
                             "caretOffset",
                             (int)t->caret_Offset,
                             "metricDataFormat",
                             (int)t->metric_Data_Format,
                             "numOfLongVerMetrics",
                             (int)t->number_Of_VMetrics);
    }
    case 5: {
        char post_dict[] = "{s:(i,i), s:(i,i), s:i, s:i, s:k, s:k, s:k, s:k, s:k}";
        TT_Postscript *t = (TT_Postscript *)table;
        return HPy_BuildValue(ctx, post_dict,
                             "format",
                             (int)FIXED_MAJOR(t->FormatType),
                             (int)FIXED_MINOR(t->FormatType),
                             "italicAngle",
                             (int)FIXED_MAJOR(t->italicAngle),
                             (int)FIXED_MINOR(t->italicAngle),
                             "underlinePosition",
                             (int)t->underlinePosition,
                             "underlineThickness",
                             (int)t->underlineThickness,
                             "isFixedPitch",
                             t->isFixedPitch,
                             "minMemType42",
                             t->minMemType42,
                             "maxMemType42",
                             t->maxMemType42,
                             "minMemType1",
                             t->minMemType1,
                             "maxMemType1",
                             t->maxMemType1);
    }
    case 6: {
        char pclt_dict[] =
            "{s:(i,i), s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:O, s:O, s:i, "
            "s:i, s:i}";
        TT_PCLT *t = (TT_PCLT *)table;
        return HPy_BuildValue(ctx, pclt_dict,
                             "version",
                             (int)FIXED_MAJOR(t->Version),
                             (int)FIXED_MINOR(t->Version),
                             "fontNumber",
                             (int)t->FontNumber,
                             "pitch",
                             (int)t->Pitch,
                             "xHeight",
                             (int)t->xHeight,
                             "style",
                             (int)t->Style,
                             "typeFamily",
                             (int)t->TypeFamily,
                             "capHeight",
                             (int)t->CapHeight,
                             "symbolSet",
                             (int)t->SymbolSet,
                             "typeFace",
                             build_str(ctx, (const char *)t->TypeFace,
                             HPy_ssize_t(16)),
                             "characterComplement",
                             build_str(ctx, (const char *)t->CharacterComplement,
                             HPy_ssize_t(8)),
                             "strokeWeight",
                             (int)t->StrokeWeight,
                             "widthType",
                             (int)t->WidthType,
                             "serifStyle",
                             (int)t->SerifStyle);
    }
    default:
        return HPy_Dup(ctx, ctx->h_None);
    }
}

static const char PyFT2Font_get_path__doc__[] =
    "get_path()\n"
    "--\n\n"
    "Get the path data from the currently loaded glyph as a tuple of vertices, "
    "codes.\n";

HPyDef_METH(PyFT2Font_get_path, "get_path", HPyFunc_NOARGS, .doc = PyFT2Font_get_path__doc__)
static HPy PyFT2Font_get_path_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    CALL_CPP_HPY(ctx, "get_path", return self->x->get_path(ctx));
}

static const char PyFT2Font_get_image__doc__[] =
    "get_image()\n"
    "--\n\n"
    "Return the underlying image buffer for this font object.\n";

HPyDef_METH(PyFT2Font_get_image, "get_image", HPyFunc_NOARGS, .doc = PyFT2Font_get_image__doc__)
static HPy PyFT2Font_get_image_impl(HPyContext *ctx, HPy h_self)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    FT2Image &im = self->x->get_image();
    npy_intp dims[] = {(npy_intp)im.get_height(), (npy_intp)im.get_width() };
    return HPy_FromPyObject(ctx, PyArray_SimpleNewFromData(2, dims, NPY_UBYTE, im.get_buffer()));
}

HPyDef_GET(PyFT2Font_postscript_name, "postscript_name")
static HPy PyFT2Font_postscript_name_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    const char *ps_name = FT_Get_Postscript_Name(self->x->get_face());
    if (ps_name == NULL) {
        ps_name = "UNAVAILABLE";
    }

    return HPyUnicode_FromString(ctx, ps_name);
}

HPyDef_GET(PyFT2Font_num_faces, "num_faces")
static HPy PyFT2Font_num_faces_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->num_faces);
}

HPyDef_GET(PyFT2Font_family_name, "family_name")
static HPy PyFT2Font_family_name_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    const char *name = self->x->get_face()->family_name;
    if (name == NULL) {
        name = "UNAVAILABLE";
    }
    return HPyUnicode_FromString(ctx, name);
}

HPyDef_GET(PyFT2Font_style_name, "style_name")
static HPy PyFT2Font_style_name_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    const char *name = self->x->get_face()->style_name;
    if (name == NULL) {
        name = "UNAVAILABLE";
    }
    return HPyUnicode_FromString(ctx, name);
}

HPyDef_GET(PyFT2Font_face_flags, "face_flags")
static HPy PyFT2Font_face_flags_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->face_flags);
}

HPyDef_GET(PyFT2Font_style_flags, "style_flags")
static HPy PyFT2Font_style_flags_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->style_flags);
}

HPyDef_GET(PyFT2Font_num_glyphs, "num_glyphs")
static HPy PyFT2Font_num_glyphs_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->num_glyphs);
}

HPyDef_GET(PyFT2Font_num_fixed_sizes, "num_fixed_sizes")
static HPy PyFT2Font_num_fixed_sizes_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->num_fixed_sizes);
}

HPyDef_GET(PyFT2Font_num_charmaps, "num_charmaps")
static HPy PyFT2Font_num_charmaps_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->num_charmaps);
}

HPyDef_GET(PyFT2Font_scalable, "scalable")
static HPy PyFT2Font_scalable_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    if (FT_IS_SCALABLE(self->x->get_face())) {
        return HPy_Dup(ctx, ctx->h_True);
    }
    return HPy_Dup(ctx, ctx->h_False);
}

HPyDef_GET(PyFT2Font_units_per_EM, "units_per_EM")
static HPy PyFT2Font_units_per_EM_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->units_per_EM);
}

HPyDef_GET(PyFT2Font_bbox, "bbox")
static HPy PyFT2Font_bbox_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    FT_BBox *bbox = &(self->x->get_face()->bbox);

    return HPy_BuildValue(ctx, "llll",
                         bbox->xMin, bbox->yMin, bbox->xMax, bbox->yMax);
}

HPyDef_GET(PyFT2Font_ascender, "ascender")
static HPy PyFT2Font_ascender_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->ascender);
}

HPyDef_GET(PyFT2Font_descender, "descender")
static HPy PyFT2Font_descender_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->descender);
}

HPyDef_GET(PyFT2Font_height, "height")
static HPy PyFT2Font_height_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->height);
}

HPyDef_GET(PyFT2Font_max_advance_width, "max_advance_width")
static HPy PyFT2Font_max_advance_width_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->max_advance_width);
}

HPyDef_GET(PyFT2Font_max_advance_height, "max_advance_height")
static HPy PyFT2Font_max_advance_height_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->max_advance_height);
}

HPyDef_GET(PyFT2Font_underline_position, "underline_position")
static HPy PyFT2Font_underline_position_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->underline_position);
}

HPyDef_GET(PyFT2Font_underline_thickness, "underline_thickness")
static HPy PyFT2Font_underline_thickness_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    return HPyLong_FromLong(ctx, self->x->get_face()->underline_thickness);
}

HPyDef_GET(PyFT2Font_fname, "fname")
static HPy PyFT2Font_fname_get(HPyContext *ctx, HPy h_self, void *closure)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    if (!HPy_IsNull(self->fname)) {
        return HPy_Dup(ctx, self->fname);
    }
    return HPy_Dup(ctx, ctx->h_None);
}

HPyDef_SLOT(PyFT2Font_get_buffer, HPy_bf_getbuffer)
static int PyFT2Font_get_buffer_impl(HPyContext *ctx, HPy h_self, HPy_buffer* buf, int flags)
{
    PyFT2Font* self = PyFT2Font_AsStruct(ctx, h_self);
    HPy_Dup(ctx, h_self);
    FT2Image &im = self->x->get_image();

    buf->obj = h_self;
    buf->buf = im.get_buffer();
    buf->len = im.get_width() * im.get_height();
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 2;
    self->shape[0] = im.get_height();
    self->shape[1] = im.get_width();
    buf->shape = self->shape;
    self->strides[0] = im.get_width();
    self->strides[1] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}


static HPyDef *PyFT2Font_defines[] = {
    // slots
    &PyFT2Font_new,
    &PyFT2Font_init,
    &PyFT2Font_get_buffer,
    &PyFT2Font_dealloc,
    // getset
    &PyFT2Font_postscript_name,
    &PyFT2Font_num_faces,
    &PyFT2Font_family_name,
    &PyFT2Font_style_name,
    &PyFT2Font_face_flags,
    &PyFT2Font_style_flags,
    &PyFT2Font_num_glyphs,
    &PyFT2Font_num_fixed_sizes,
    &PyFT2Font_num_charmaps,
    &PyFT2Font_scalable,
    &PyFT2Font_units_per_EM,
    &PyFT2Font_bbox,
    &PyFT2Font_ascender,
    &PyFT2Font_descender,
    &PyFT2Font_height,
    &PyFT2Font_max_advance_width,
    &PyFT2Font_max_advance_height,
    &PyFT2Font_underline_position,
    &PyFT2Font_underline_thickness,
    &PyFT2Font_fname,

    // methods
    &PyFT2Font_clear,
    &PyFT2Font_set_size,
    &PyFT2Font_set_charmap,
    &PyFT2Font_select_charmap,
    &PyFT2Font_get_kerning,
    &PyFT2Font_set_text,
    &PyFT2Font_get_num_glyphs,
    &PyFT2Font_load_char,
    &PyFT2Font_load_glyph,
    &PyFT2Font_get_width_height,
    &PyFT2Font_get_bitmap_offset,
    &PyFT2Font_get_descent,
    &PyFT2Font_draw_glyphs_to_bitmap,
    &PyFT2Font_get_xys,
    &PyFT2Font_draw_glyph_to_bitmap,
    &PyFT2Font_get_glyph_name,
    &PyFT2Font_get_charmap,
    &PyFT2Font_get_char_index,
    &PyFT2Font_get_sfnt,
    &PyFT2Font_get_name_index,
    &PyFT2Font_get_ps_font_info,
    &PyFT2Font_get_sfnt_table,
    &PyFT2Font_get_path,
    &PyFT2Font_get_image,
    NULL
};

static HPyType_Spec PyFT2Font_type_spec = {
    .name = "matplotlib.ft2font.FT2Font",
    .basicsize = sizeof(PyFT2Font),
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_BASETYPE,
    .defines = PyFT2Font_defines,
    .doc = PyFT2Font_init__doc__,
};


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

int add_dict_string(HPyContext *ctx, HPy dict, const char *key, const char *value)
{
    HPy valobj = HPyUnicode_FromString(ctx, value);
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

HPyDef_SLOT(ft2font_hpy_exec, HPy_mod_exec)
static int ft2font_hpy_exec_impl(HPyContext *ctx, HPy m)
{
    HPy tmp;

    if (!npy_import_array_hpy(ctx)) {
        return 1;
    }

    if (!HPyHelpers_AddType(ctx, m, "FT2Image", &PyFT2Image_type_spec, NULL)) {
        return 1;
    }

    tmp = HPy_GetAttr_s(ctx, m, "FT2Image");
    HPyGlobal_Store(ctx, &PyFT2ImageType, tmp);
    HPy_Close(ctx, tmp);

    if (!HPyHelpers_AddType(ctx, m, "PyGlyph", &PyGlyph_type_spec, NULL)) {
        return 1;
    }

    tmp = HPy_GetAttr_s(ctx, m, "PyGlyph"); // should be hidden
    HPyGlobal_Store(ctx, &PyGlyphType, tmp);
    HPy_Close(ctx, tmp);

    if (!HPyHelpers_AddType(ctx, m, "FT2Font", &PyFT2Font_type_spec, NULL)) {
        return 1;
    }

    if (add_dict_int(ctx, m, "SCALABLE", FT_FACE_FLAG_SCALABLE) ||
        add_dict_int(ctx, m, "FIXED_SIZES", FT_FACE_FLAG_FIXED_SIZES) ||
        add_dict_int(ctx, m, "FIXED_WIDTH", FT_FACE_FLAG_FIXED_WIDTH) ||
        add_dict_int(ctx, m, "SFNT", FT_FACE_FLAG_SFNT) ||
        add_dict_int(ctx, m, "HORIZONTAL", FT_FACE_FLAG_HORIZONTAL) ||
        add_dict_int(ctx, m, "VERTICAL", FT_FACE_FLAG_VERTICAL) ||
        add_dict_int(ctx, m, "KERNING", FT_FACE_FLAG_KERNING) ||
        add_dict_int(ctx, m, "FAST_GLYPHS", FT_FACE_FLAG_FAST_GLYPHS) ||
        add_dict_int(ctx, m, "MULTIPLE_MASTERS", FT_FACE_FLAG_MULTIPLE_MASTERS) ||
        add_dict_int(ctx, m, "GLYPH_NAMES", FT_FACE_FLAG_GLYPH_NAMES) ||
        add_dict_int(ctx, m, "EXTERNAL_STREAM", FT_FACE_FLAG_EXTERNAL_STREAM) ||
        add_dict_int(ctx, m, "ITALIC", FT_STYLE_FLAG_ITALIC) ||
        add_dict_int(ctx, m, "BOLD", FT_STYLE_FLAG_BOLD) ||
        add_dict_int(ctx, m, "KERNING_DEFAULT", FT_KERNING_DEFAULT) ||
        add_dict_int(ctx, m, "KERNING_UNFITTED", FT_KERNING_UNFITTED) ||
        add_dict_int(ctx, m, "KERNING_UNSCALED", FT_KERNING_UNSCALED) ||
        add_dict_int(ctx, m, "LOAD_DEFAULT", FT_LOAD_DEFAULT) ||
        add_dict_int(ctx, m, "LOAD_NO_SCALE", FT_LOAD_NO_SCALE) ||
        add_dict_int(ctx, m, "LOAD_NO_HINTING", FT_LOAD_NO_HINTING) ||
        add_dict_int(ctx, m, "LOAD_RENDER", FT_LOAD_RENDER) ||
        add_dict_int(ctx, m, "LOAD_NO_BITMAP", FT_LOAD_NO_BITMAP) ||
        add_dict_int(ctx, m, "LOAD_VERTICAL_LAYOUT", FT_LOAD_VERTICAL_LAYOUT) ||
        add_dict_int(ctx, m, "LOAD_FORCE_AUTOHINT", FT_LOAD_FORCE_AUTOHINT) ||
        add_dict_int(ctx, m, "LOAD_CROP_BITMAP", FT_LOAD_CROP_BITMAP) ||
        add_dict_int(ctx, m, "LOAD_PEDANTIC", FT_LOAD_PEDANTIC) ||
        add_dict_int(ctx, m, "LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH", FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH) ||
        add_dict_int(ctx, m, "LOAD_NO_RECURSE", FT_LOAD_NO_RECURSE) ||
        add_dict_int(ctx, m, "LOAD_IGNORE_TRANSFORM", FT_LOAD_IGNORE_TRANSFORM) ||
        add_dict_int(ctx, m, "LOAD_MONOCHROME", FT_LOAD_MONOCHROME) ||
        add_dict_int(ctx, m, "LOAD_LINEAR_DESIGN", FT_LOAD_LINEAR_DESIGN) ||
        add_dict_int(ctx, m, "LOAD_NO_AUTOHINT", (unsigned long)FT_LOAD_NO_AUTOHINT) ||
        add_dict_int(ctx, m, "LOAD_TARGET_NORMAL", (unsigned long)FT_LOAD_TARGET_NORMAL) ||
        add_dict_int(ctx, m, "LOAD_TARGET_LIGHT", (unsigned long)FT_LOAD_TARGET_LIGHT) ||
        add_dict_int(ctx, m, "LOAD_TARGET_MONO", (unsigned long)FT_LOAD_TARGET_MONO) ||
        add_dict_int(ctx, m, "LOAD_TARGET_LCD", (unsigned long)FT_LOAD_TARGET_LCD) ||
        add_dict_int(ctx, m, "LOAD_TARGET_LCD_V", (unsigned long)FT_LOAD_TARGET_LCD_V)) {
        return 1;
    }

    // initialize library
    int error = FT_Init_FreeType(&_ft2Library);

    if (error) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Could not initialize the freetype2 library");
        return 1;
    }

    {
        FT_Int major, minor, patch;
        char version_string[64];

        FT_Library_Version(_ft2Library, &major, &minor, &patch);
        sprintf(version_string, "%d.%d.%d", major, minor, patch);
        if (add_dict_string(ctx, m, "__freetype_version__", version_string)) {
            return 1;
        }
    }

    if (add_dict_string(ctx, m, "__freetype_build_type__", STRINGIFY(FREETYPE_BUILD_TYPE))) {
        return 1;
    }

    return 0;
}

static HPyDef *module_defines[] = {
    &ft2font_hpy_exec,
    NULL
};

static HPyGlobal *module_globals[] = {
    &PyFT2ImageType,
    &PyGlyphType,
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
HPy_MODINIT(ft2font_hpy, moduledef)

#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
