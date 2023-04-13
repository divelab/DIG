# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
cimport cython
from cython.parallel import prange,parallel
from cython.operator import dereference as deref, preincrement as inc
from cython cimport Py_buffer
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort, unique, lower_bound
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libcpp cimport bool
import time,math
import random
from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX

cimport graphsaint.cython_utils as cutils
import graphsaint.cython_utils as cutils



cdef class Sampler:
    cdef int num_proc,num_sample_per_proc
    cdef vector[int] adj_indptr_vec
    cdef vector[int] adj_indices_vec
    cdef vector[int] node_train_vec
    cdef vector[vector[int]] node_sampled
    cdef vector[vector[int]] ret_indptr
    cdef vector[vector[int]] ret_indices
    cdef vector[vector[int]] ret_indices_orig
    cdef vector[vector[float]] ret_data
    cdef vector[vector[int]] ret_edge_index

    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,*argv):
        cutils.npy2vec_int(adj_indptr,self.adj_indptr_vec)
        cutils.npy2vec_int(adj_indices,self.adj_indices_vec)
        cutils.npy2vec_int(node_train,self.node_train_vec)
        self.num_proc = num_proc
        self.num_sample_per_proc = num_sample_per_proc
        self.node_sampled = vector[vector[int]](num_proc*num_sample_per_proc)
        self.ret_indptr = vector[vector[int]](num_proc*num_sample_per_proc)
        self.ret_indices = vector[vector[int]](num_proc*num_sample_per_proc)
        self.ret_indices_orig = vector[vector[int]](num_proc*num_sample_per_proc)
        self.ret_data = vector[vector[float]](num_proc*num_sample_per_proc)
        self.ret_edge_index = vector[vector[int]](num_proc*num_sample_per_proc)

    cdef void adj_extract(self, int p) nogil:
        """
        Extract a subg adj matrix from the original training adj matrix
        ret_indices_orig:   the indices vector corresponding to node id in original G.
        """
        cdef int r = 0
        cdef int idx_g = 0
        cdef int i, i_end, v, j
        cdef int num_v_orig, num_v_sub
        cdef int start_neigh, end_neigh
        cdef vector[int] _arr_bit
        cdef int cumsum
        num_v_orig = self.adj_indptr_vec.size()-1
        while r < self.num_sample_per_proc:
            _arr_bit = vector[int](num_v_orig,-1)
            idx_g = p*self.num_sample_per_proc+r
            num_v_sub = self.node_sampled[idx_g].size()
            self.ret_indptr[idx_g] = vector[int](num_v_sub+1,0)
            self.ret_indices[idx_g] = vector[int]()
            self.ret_indices_orig[idx_g] = vector[int]()
            self.ret_data[idx_g] = vector[float]()
            self.ret_edge_index[idx_g]=vector[int]()
            i_end = num_v_sub
            i = 0
            while i < i_end:
                _arr_bit[self.node_sampled[idx_g][i]] = i
                i = i + 1
            i = 0
            while i < i_end:
                v = self.node_sampled[idx_g][i]
                start_neigh = self.adj_indptr_vec[v]
                end_neigh = self.adj_indptr_vec[v+1]
                j = start_neigh
                while j < end_neigh:
                    if _arr_bit[self.adj_indices_vec[j]] > -1:
                        self.ret_indices[idx_g].push_back(_arr_bit[self.adj_indices_vec[j]])
                        self.ret_indices_orig[idx_g].push_back(self.adj_indices_vec[j])
                        self.ret_edge_index[idx_g].push_back(j)
                        self.ret_indptr[idx_g][_arr_bit[v]+1] = self.ret_indptr[idx_g][_arr_bit[v]+1] + 1
                        self.ret_data[idx_g].push_back(1.)
                    j = j + 1
                i = i + 1
            cumsum = self.ret_indptr[idx_g][0]
            i = 0
            while i < i_end:
                cumsum = cumsum + self.ret_indptr[idx_g][i+1]
                self.ret_indptr[idx_g][i+1] = cumsum
                i = i + 1
            r = r + 1

    def get_return(self):
        """
        Convert the subgraph related data structures from C++ to python. So that cython
        can return them to the PyTorch trainer.

        Inputs:
            None

        Outputs:
            see outputs of the `par_sample()` function.
        """
        num_subg = self.num_proc*self.num_sample_per_proc
        l_subg_indptr = []
        l_subg_indices = []
        l_subg_data = []
        l_subg_nodes = []
        l_subg_edge_index = []
        offset_nodes = [0]
        offset_indptr = [0]
        offset_indices = [0]
        offset_data = [0]
        offset_edge_index = [0]
        for r in range(num_subg):
            offset_nodes.append(offset_nodes[r]+self.node_sampled[r].size())
            offset_indptr.append(offset_indptr[r]+self.ret_indptr[r].size())
            offset_indices.append(offset_indices[r]+self.ret_indices[r].size())
            offset_data.append(offset_data[r]+self.ret_data[r].size())
            offset_edge_index.append(offset_edge_index[r]+self.ret_edge_index[r].size())
        cdef vector[int] ret_nodes_vec = vector[int]()
        cdef vector[int] ret_indptr_vec = vector[int]()
        cdef vector[int] ret_indices_vec = vector[int]()
        cdef vector[int] ret_edge_index_vec = vector[int]()
        cdef vector[float] ret_data_vec = vector[float]()
        ret_nodes_vec.reserve(offset_nodes[num_subg])
        ret_indptr_vec.reserve(offset_indptr[num_subg])
        ret_indices_vec.reserve(offset_indices[num_subg])
        ret_data_vec.reserve(offset_data[num_subg])
        ret_edge_index_vec.reserve(offset_edge_index[num_subg])
        for r in range(num_subg):
            ret_nodes_vec.insert(ret_nodes_vec.end(),self.node_sampled[r].begin(),self.node_sampled[r].end())
            ret_indptr_vec.insert(ret_indptr_vec.end(),self.ret_indptr[r].begin(),self.ret_indptr[r].end())
            ret_indices_vec.insert(ret_indices_vec.end(),self.ret_indices[r].begin(),self.ret_indices[r].end())
            ret_edge_index_vec.insert(ret_edge_index_vec.end(),self.ret_edge_index[r].begin(),self.ret_edge_index[r].end())
            ret_data_vec.insert(ret_data_vec.end(),self.ret_data[r].begin(),self.ret_data[r].end())

        cdef cutils.array_wrapper_int wint_indptr = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_indices = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_nodes = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_float wfloat_data = cutils.array_wrapper_float()
        cdef cutils.array_wrapper_int wint_edge_index = cutils.array_wrapper_int()

        wint_indptr.set_data(ret_indptr_vec)
        ret_indptr_np = np.frombuffer(wint_indptr,dtype=np.int32)
        wint_indices.set_data(ret_indices_vec)
        ret_indices_np = np.frombuffer(wint_indices,dtype=np.int32)
        wint_nodes.set_data(ret_nodes_vec)
        ret_nodes_np = np.frombuffer(wint_nodes,dtype=np.int32)
        wfloat_data.set_data(ret_data_vec)
        ret_data_np = np.frombuffer(wfloat_data,dtype=np.float32)
        wint_edge_index.set_data(ret_edge_index_vec)
        ret_edge_index_np = np.frombuffer(wint_edge_index,dtype=np.int32)

        for r in range(num_subg):
            l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
            l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
            l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
            l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
            l_subg_edge_index.append(ret_edge_index_np[offset_indices[r]:offset_indices[r+1]])

        return l_subg_indptr,l_subg_indices,l_subg_data,l_subg_nodes,l_subg_edge_index

    cdef void sample(self, int p) nogil:
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def par_sample(self):
        """
        The main function for the sampler class. It launches multiple independent samplers
        in parallel (task parallelism by openmp), where the serial sampling function is defined
        in the corresponding sub-class. Then it returns node-induced subgraph by `_adj_extract()`,
        and convert C++ vectors to python lists / numpy arrays by `_get_return()`.

        Suppose we sample P subgraphs in parallel. Each subgraph has n nodes and e edges.

        Inputs:
            None

        Outputs (elements in the list of `ret`):
            l_subg_indptr       list of np array, length of list = P and length of each array is n+1
            l_subg_indices      list of np array, length of list = P and length of each array is m.
                                node IDs in the array are renamed to be subgraph ID (range: 0 ~ n-1)
            l_subg_data         list of np array, length of list = P and length of each array is m.
                                Normally, values in the array should be all 1.
            l_subg_nodes        list of np array, length of list = P and length of each array is n.
                                Element i in the array shows the training graph node ID of the i-th
                                subgraph node.
            l_subg_edge_index   list of np array, length of list = P and length of each array is m.
                                Element i in the array shows the training graph edge index of the
                                i-the subgraph edge.
        """
        cdef int p = 0
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc,schedule='dynamic'):
                self.sample(p)
                self.adj_extract(p)
        ret = self.get_return()
        _len = self.num_proc*self.num_sample_per_proc
        self.node_sampled.swap(vector[vector[int]](_len))
        self.ret_indptr.swap(vector[vector[int]](_len))
        self.ret_indices.swap(vector[vector[int]](_len))
        self.ret_indices_orig.swap(vector[vector[int]](_len))
        self.ret_data.swap(vector[vector[float]](_len))
        self.ret_edge_index.swap(vector[vector[int]](_len))
        return ret


# ----------------------------------------------------

cdef class MRW(Sampler):
    cdef int size_frontier,size_subg
    cdef int avg_deg
    cdef vector[int] arr_deg_vec
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        np.ndarray[int,ndim=1,mode='c'] p_dist,
                        int max_deg, int size_frontier, int size_subg):
        self.size_frontier = size_frontier
        self.size_subg = size_subg
        _arr_deg = np.clip(p_dist,0,max_deg)
        cutils.npy2vec_int(_arr_deg,self.arr_deg_vec)
        self.avg_deg = _arr_deg.mean()

    cdef void sample(self, int p) nogil:
        cdef vector[int] frontier
        cdef int i = 0
        cdef int num_train_node = self.node_train_vec.size()
        cdef int r = 0
        cdef int alpha = 2
        cdef vector[int] arr_ind0
        cdef vector[int] arr_ind1
        cdef vector[int].iterator it
        arr_ind0.reserve(alpha*self.avg_deg)
        arr_ind1.reserve(alpha*self.avg_deg)
        cdef int c, cnt, j, k
        cdef int v, vidx, vpop, vneigh, offset, vnext
        cdef int idx_begin, idx_end
        cdef int num_neighs_pop, num_neighs_next
        while r < self.num_sample_per_proc:
            # prepare initial frontier
            arr_ind0.clear()
            arr_ind1.clear()
            frontier.clear()
            i = 0
            while i < self.size_frontier:        # NB: here we don't care if a node appear twice
                frontier.push_back(self.node_train_vec[rand()%num_train_node])
                i = i + 1
            # init indicator array
            it = frontier.begin()
            while it != frontier.end():
                v = deref(it)
                cnt = arr_ind0.size()
                c = cnt
                while c < cnt + self.arr_deg_vec[v]:
                    arr_ind0.push_back(v)
                    arr_ind1.push_back(c-cnt)
                    c = c + 1
                arr_ind1[cnt] = -self.arr_deg_vec[v]
                inc(it)
            # iteratively update frontier
            j = self.size_frontier
            while j < self.size_subg:
                # select next node to pop out of frontier
                while True:
                    vidx = rand()%arr_ind0.size()
                    vpop = arr_ind0[vidx]
                    if vpop >= 0:
                        break
                # prepare to update arr_ind*
                offset = arr_ind1[vidx]
                if offset < 0:
                    idx_begin = vidx
                    idx_end = idx_begin - offset
                else:
                    idx_begin = vidx - offset
                    idx_end = idx_begin - arr_ind1[idx_begin]
                # cleanup 1: invalidate entries
                k = idx_begin
                while k < idx_end:
                    arr_ind0[k] = -1
                    arr_ind1[k] = 0
                    k = k + 1
                # cleanup 2: add new entries
                num_neighs_pop = self.adj_indptr_vec[vpop+1] - self.adj_indptr_vec[vpop]
                vnext = self.adj_indices_vec[self.adj_indptr_vec[vpop]+rand()%num_neighs_pop]
                self.node_sampled[p*self.num_sample_per_proc+r].push_back(vnext)
                num_neighs_next = self.arr_deg_vec[vnext]
                cnt = arr_ind0.size()
                c = cnt
                while c < cnt + num_neighs_next:
                    arr_ind0.push_back(vnext)
                    arr_ind1.push_back(c-cnt)
                    c = c + 1
                arr_ind1[cnt] = -num_neighs_next
                j = j + 1
            self.node_sampled[p*self.num_sample_per_proc+r].insert(self.node_sampled[p*self.num_sample_per_proc+r].end(),frontier.begin(),frontier.end())
            sort(self.node_sampled[p*self.num_sample_per_proc+r].begin(),self.node_sampled[p*self.num_sample_per_proc+r].end())
            self.node_sampled[p*self.num_sample_per_proc+r].erase(unique(self.node_sampled[p*self.num_sample_per_proc+r].begin(),\
                    self.node_sampled[p*self.num_sample_per_proc+r].end()),self.node_sampled[p*self.num_sample_per_proc+r].end())
            r = r + 1



# ----------------------------------------------------

cdef class RW(Sampler):
    cdef int size_root, size_depth
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        int size_root, int size_depth):
        self.size_root = size_root
        self.size_depth = size_depth

    cdef void sample(self, int p) nogil:
        cdef int iroot = 0
        cdef int idepth = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int v
        cdef int num_train_node = self.node_train_vec.size()
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            # sample root
            iroot = 0
            while iroot < self.size_root:
                v = self.node_train_vec[rand()%num_train_node]
                self.node_sampled[idx_subg].push_back(v)
                # sample random walk
                idepth = 0
                while idepth < self.size_depth:
                    if (self.adj_indptr_vec[v+1]-self.adj_indptr_vec[v]>0):
                        v = self.adj_indices_vec[self.adj_indptr_vec[v]+rand()%(self.adj_indptr_vec[v+1]-self.adj_indptr_vec[v])]
                        self.node_sampled[idx_subg].push_back(v)
                    idepth = idepth + 1
                iroot = iroot + 1
            r = r + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())




# ----------------------------------------------------

cdef class Edge(Sampler):
    cdef vector[int] row_train_vec
    cdef vector[int] col_train_vec
    cdef vector[float] prob_edge_vec
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        np.ndarray[int,ndim=1,mode='c'] row_train,
                        np.ndarray[int,ndim=1,mode='c'] col_train,
                        np.ndarray[float,ndim=1,mode='c'] prob_edge,*argv):
        cutils.npy2vec_int(row_train,self.row_train_vec)
        cutils.npy2vec_int(col_train,self.col_train_vec)
        cutils.npy2vec_float(prob_edge,self.prob_edge_vec)

    cdef void sample(self, int p) nogil:
        cdef int num_edge = self.row_train_vec.size()
        cdef int i=0
        cdef float ran=0.
        cdef int g=0
        cdef int idx_subg
        while g < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+g
            i = 0
            while i < num_edge:
                ran = (<float> rand()) / RAND_MAX
                if ran > self.prob_edge_vec[i]:
                    # edge not selected
                    i = i + 1
                    continue
                self.node_sampled[idx_subg].push_back(self.row_train_vec[i])
                self.node_sampled[idx_subg].push_back(self.col_train_vec[i])
                i = i + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())
            g = g + 1


cdef class Edge2(Sampler):
    """
    approximate version of the above Edge class
    """
    cdef vector[int] row_train_vec
    cdef vector[int] col_train_vec
    cdef vector[float] p_dist_cumsum_vec
    cdef int size_subg_e
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        np.ndarray[int,ndim=1,mode='c'] row_train,
                        np.ndarray[int,ndim=1,mode='c'] col_train,
                        np.ndarray[float,ndim=1,mode='c'] p_dist_cumsum,
                        int size_subg_e):
        self.size_subg_e = size_subg_e
        cutils.npy2vec_int(row_train,self.row_train_vec)
        cutils.npy2vec_int(col_train,self.col_train_vec)
        cutils.npy2vec_float(p_dist_cumsum,self.p_dist_cumsum_vec)

    cdef void sample(self, int p) nogil:
        cdef int i = 0
        cdef int r = 0
        cdef int e
        cdef int idx_subg
        cdef float ran = 0.
        cdef float ran_range = self.p_dist_cumsum_vec[self.p_dist_cumsum_vec.size()-1]
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            i = 0
            while i < self.size_subg_e:
                ran = (<float> rand()) / RAND_MAX * ran_range
                e = lower_bound(self.p_dist_cumsum_vec.begin(),self.p_dist_cumsum_vec.end(),ran)-self.p_dist_cumsum_vec.begin()
                self.node_sampled[idx_subg].push_back(self.row_train_vec[e])
                self.node_sampled[idx_subg].push_back(self.col_train_vec[e])
                i = i + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())
            r = r + 1

# ----------------------------------------------------

cdef class Node(Sampler):
    cdef int size_subg
    cdef vector[int] p_dist_cumsum_vec
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        np.ndarray[int,ndim=1,mode='c'] p_dist_cumsum,
                        int size_subg):
        self.size_subg = size_subg
        cutils.npy2vec_int(p_dist_cumsum,self.p_dist_cumsum_vec)

    cdef void sample(self, int p) nogil:
        cdef int i = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int sample
        cdef int rand_range = self.p_dist_cumsum_vec[self.node_train_vec.size()-1]
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            i = 0
            while i < self.size_subg:
                sample = rand()%rand_range
                self.node_sampled[idx_subg].push_back(self.node_train_vec[lower_bound(self.p_dist_cumsum_vec.begin(),self.p_dist_cumsum_vec.end(),sample)-self.p_dist_cumsum_vec.begin()])
                i = i + 1
            r = r + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())

# -----------------------------------------------------

cdef class FullBatch(Sampler):
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc):
        pass

    cdef void sample(self, int p) nogil:
        cdef int i = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int sample
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            i = 0
            while i < self.node_train_vec.size():
                sample = i
                self.node_sampled[idx_subg].push_back(self.node_train_vec[sample])
                i = i + 1
            r = r + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())
