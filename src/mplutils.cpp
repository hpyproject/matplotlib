/* -*- mode: c++; c-basic-offset: 4 -*- */
/*-----------------------------------------------------------------------------
| Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
| Copyright (c) 2022, Oracle and/or its affiliates.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

#include "mplutils.h"

#ifndef HPY
int add_dict_int(PyObject *dict, const char *key, long val)
{
    PyObject *valobj;
    valobj = PyLong_FromLong(val);
    if (valobj == NULL) {
        return 1;
    }

    if (PyDict_SetItemString(dict, key, valobj)) {
        Py_DECREF(valobj);
        return 1;
    }

    Py_DECREF(valobj);

    return 0;
}
#endif
