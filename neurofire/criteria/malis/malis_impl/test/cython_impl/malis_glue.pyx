import numpy as np
cimport numpy as np

cdef extern from 'malis.hxx' namespace 'malis':
    void malis;
