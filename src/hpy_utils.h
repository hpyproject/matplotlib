/*-----------------------------------------------------------------------------
| Copyright (c) 2023, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#ifndef SRC_HPY_UTILS_H_
#define SRC_HPY_UTILS_H_

static inline int
HPy_TypeCheck_g(HPyContext *ctx, HPy obj, HPyGlobal type)
{
    int res;
    HPy tmp = HPyGlobal_Load(ctx, type);
    res = HPy_TypeCheck(ctx, obj, tmp);
    HPy_Close(ctx, tmp);
    return res;
}

static inline HPy
HPy_CallMethod_s(HPyContext *ctx, const char *name, const HPy *args, size_t nargs, HPy kwnames)
{
    HPy h_name = HPyUnicode_FromString(ctx, name);
    if (HPy_IsNull(h_name)) {
        return HPy_NULL;
    }
    HPy h_result = HPy_CallMethod(ctx, h_name, args, nargs, kwnames);
    HPy_Close(ctx, h_name);
    return h_result;
}

#define HPy_CLEAR(ctx, op)          \
    do {                            \
        HPy _h_tmp = (op);          \
        if (!HPy_IsNull(_h_tmp)) {  \
            (op) = HPy_NULL;        \
            HPy_Close(ctx, _h_tmp); \
        }                           \
    } while (0)

static inline
HPy HPyPackLongAndCallMethod(HPyContext *ctx, HPy obj, const char *func_name, unsigned long val)
{
    HPy h_val = HPyLong_FromUnsignedLong(ctx, val);
    const HPy args[] = {obj, h_val};
    HPy result = HPy_CallMethod_s(ctx, func_name, args, 2, HPy_NULL);
    HPy_Close(ctx, h_val);
    return result;
}

static inline
HPy HPy_Open(HPyContext *ctx, HPy filename, const char *flags)
{
    HPy open = HPy_GetItem_s(ctx, ctx->h_Builtins, "open");
    if (HPy_IsNull(open))
        return HPy_NULL;
    HPy h_flags = HPyUnicode_FromString(ctx, flags);
    if (HPy_IsNull(h_flags))
        return HPy_NULL;
    const HPy tuple[] = {filename, h_flags};
    HPy result = HPy_Call(ctx, open, tuple, 2, HPy_NULL);
    HPy_Close(ctx, h_flags);
    HPy_Close(ctx, open);
    return result;
}

#endif /* SRC_HPY_UTILS_H_ */
