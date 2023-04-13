# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange,parallel
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference, preincrement

def norm_aggr(data,edge_index,norm_aggr,num_proc=20):
    cdef int num_proc_view=num_proc
    cdef float [:] data_view=data
    cdef int length=data.shape[0]
    cdef int [:] edge_index_view=edge_index
    cdef float [:] norm_aggr_view=norm_aggr
    cdef int i
    for i in prange(length,schedule='static',nogil=True,num_threads=num_proc_view):
        data_view[i]=norm_aggr_view[edge_index_view[i]]