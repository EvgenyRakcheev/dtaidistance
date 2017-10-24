"""
dtaidistance.dtw_c - Dynamic Time Warping
    Part of the DTAI distance code.
    Copyright 2016 KU Leuven, DTAI Research Group
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import math
import numpy as np
cimport numpy as np
cimport cython
import cython
import ctypes
from cpython cimport array, bool
from cython import parallel
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free, abs
from libc.stdio cimport printf
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, pow

DTYPE = np.double
ctypedef np.double_t DTYPE_t
cdef double inf = np.inf

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def distance(np.ndarray[DTYPE_t, ndim=1] s1, np.ndarray[DTYPE_t, ndim=1] s2,
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0):
    """
    Dynamic Time Warping (keep compact matrix)
    :param s1: First sequence (np.array(np.float64))
    :param s2: Second sequence
    :param window: Only allow for shifts up to this amount away from the two diagonals
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param max_length_diff: Max length difference between the two sequences
    :param penalty: Cost incurrend when performing compression or expansion
    Returns: DTW distance
    """
    assert s1.dtype == DTYPE and s2.dtype == DTYPE
    cdef int r = len(s1)
    cdef int c = len(s2)
    if max_length_diff != 0 and abs(r-c) > max_length_diff:
        return inf
    if window == 0:
        window = max(r, c)
    if max_step == 0:
        max_step = inf
    else:
        max_step *= max_step
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist *= max_dist
    penalty *= penalty
    cdef np.ndarray[DTYPE_t, ndim=2] dtw = np.full((2, min( c +1 ,abs( r -c ) + 2 *( window -1 ) + 1 + 1 +1)), inf)
    dtw[0, 0] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int skip = 0
    cdef int skipp = 0
    cdef int i0 = 1
    cdef int i1 = 0
    cdef DTYPE_t d
    for i in range(r):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        skipp = skip
        skip = max(0, i - window + 1)
        i0 = 1 - i0
        i1 = 1 - i1
        dtw[i1 ,:] = inf
        if dtw.shape[1] == c+ 1:
            skip = 0
        for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
            d = (s1[i] - s2[j])**2
            if d > max_step:
                continue
            dtw[i1, j + 1 - skip] = d + min(dtw[i0, j - skipp],
                                            dtw[i0, j + 1 - skipp] + penalty,
                                            dtw[i1, j - skip] + penalty)
            if dtw[i1, j + 1 - skip] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1, j + 1 - skip] = inf
                if prev_last_under_max_dist + 1 - skipp < j + 1 - skip:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            return inf
    # print(dtw)
    return math.sqrt(dtw[i1, min(c, c + window - 1) - skip])


def distance_nogil(double[:] s1, double[:] s2,
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0):
    """DTW distance.
    See distance(). This calls a pure c dtw computation that avoids the GIL.
    :param s1: First sequence (buffer of doubles)
    :param s2: Second sequence (buffer of doubles)
    """
    #return distance_nogil_c(s1, s2, len(s1), len(s2),
    # If the arrays (memoryviews) are not C contiguous, the pointer will not point to the correct array
    if isinstance(s1, (np.ndarray, np.generic)):
        if not s1.base.flags.c_contiguous:
            s1 = s1.copy()
    if isinstance(s2, (np.ndarray, np.generic)):
        if not s2.base.flags.c_contiguous:
            s2 = s2.copy()
    return distance_nogil_c(&s1[0], &s2[0], len(s1), len(s2),
                            window, max_dist, max_step, max_length_diff, penalty)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.infer_types(False)
cdef double distance_nogil_c(
             double *s1, double *s2,
             int r, # len_s1
             int c, # len_s2
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0) nogil:
    """DTW distance.
    See distance(). This is a pure c dtw computation that avoid the GIL.
    """
    #printf("%i, %i\n", r, c)
    if max_length_diff != 0 and abs(r-c) > max_length_diff:
        return inf
    if window == 0:
        window = max(r, c)
    if max_step == 0:
        max_step = inf
    else:
        max_step = pow(max_step, 2)
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist = pow(max_dist, 2)
    penalty = pow(penalty, 2)
    cdef int length = min(c+1,abs(r-c) + 2*(window-1) + 1 + 1 + 1)
    cdef double ** dtw
    dtw = <double **> malloc(sizeof(double *) * (r + 1))
    cdef int i
    for i in range(r + 1):
        dtw[i] = <double *> malloc(sizeof(double) * (c + 1))
    cdef int j
    for i in range(r + 1):
        for j in range(c + 1):
            dtw[i][j] = inf
    dtw[0][0] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int i0 = 1
    cdef int i1 = 0
    cdef int minj
    cdef int maxj
    cdef double minv
    cdef DTYPE_t d
    cdef double tempv
    for i in range(r):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        maxj = r - c
        if maxj < 0:
            maxj = 0
        maxj = i - maxj - window + 1
        if maxj < 0:
            maxj = 0
        i0 = i
        i1 = i + 1
        minj = c - r
        if minj < 0:
            minj = 0
        minj = i + minj + window
        if minj > c:
            minj = c
        for j in range(maxj, minj):
            d = pow(s1[i] - s2[j], 2)
            if d > max_step:
                continue
            minv = dtw[i0][j]
            tempv = dtw[i0][j + 1] + penalty
            if tempv < minv:
                minv = tempv
            tempv = dtw[i1][j] + penalty
            if tempv < minv:
                minv = tempv
            dtw[i1][j + 1] = d + minv
            if dtw[i1][j + 1] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1][j + 1] = inf
                if prev_last_under_max_dist < j + 1:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            return inf
    cdef int ** path
    path = <int **> malloc(sizeof(int *) * (r + c - 1))
    for i in range(r + c - 1):
        path[i] = <int *> malloc(sizeof(int) * 2)
        for j in range(2):
            path[i][j] = 0
    cdef int cur_i
    cdef int cur_j
    cdef int s1_len_cur
    cdef int s2_len_cur
    cur_i = r
    cur_j = c
    cdef double cur_cell = dtw[cur_i][cur_j]
    path[0] = [cur_i - 1, cur_j - 1]
    cdef int ind_of_path = 1
    while cur_i > 1 or cur_j > 1:
        s1_len_cur = cur_i
        s2_len_cur = cur_j
        if cur_i >= 1 and cur_j >= 1 and dtw[cur_i - 1][cur_j - 1] <= cur_cell:
            s1_len_cur = cur_i - 1
            s2_len_cur = cur_j - 1
            cur_cell = dtw[cur_i - 1][cur_j - 1]
            
        if cur_i >= 1 and dtw[cur_i - 1][cur_j] <= cur_cell:
            s1_len_cur = cur_i - 1
            s2_len_cur = cur_j
            cur_cell = dtw[cur_i - 1][cur_j]
            
        if cur_j >= 1 and dtw[cur_i][cur_j - 1] <= cur_cell:
            s1_len_cur = cur_i
            s2_len_cur = cur_j - 1
            cur_cell = dtw[cur_i][cur_j - 1]
            
        path[ind_of_path] = [s1_len_cur - 1, s2_len_cur - 1]
        ind_of_path += 1
        cur_i = s1_len_cur
        cur_j = s2_len_cur
    cdef int ** reversed_path
    reversed_path = <int **> malloc(sizeof(int *) * ind_of_path)
    for i in range(ind_of_path):
        reversed_path[i] = <int *> malloc(sizeof(int) * 2)
        reversed_path[i] = path[ind_of_path - i - 1]
    cdef int* path_begin = [0, 0]
    cdef int** path_end
    path_end = <int **> malloc(sizeof(int *) * (r + c - 1))
    for i in range(r + c - 1):
        path_end[i] = <int *> malloc(sizeof(int) * 2)
        for j in range(2):
            path_end[i][j] = 0
    cdef int ind_of_path_end = 0
    for i in range(ind_of_path):
        if reversed_path[i][0] == 0 or reversed_path[i][1] == 0:
            path_begin = reversed_path[i]
        if reversed_path[i][0] == r - 1 or reversed_path[i][1] == c - 1:
            path_end[ind_of_path_end] = reversed_path[i]
            ind_of_path_end += 1
    cdef double result = dtw[r][c] - dtw[path_begin[0] + 1][path_begin[1] + 1] +                          (s1[path_begin[0]]-s2[path_begin[1]])**2
    if ind_of_path_end > 1:
        for i in range(1, ind_of_path_end):
            result -= (s1[path_end[i][0]]-s2[path_end[i][1]])**2
    for i in range(r + 1):
        free(dtw[i])
    free(dtw)
    for i in range(r + c - 1):
        free(path[i])
        free(path_end[i])
    free(path)
    free(path_end)
    for i in range(ind_of_path):
        free(reversed_path[i])
    free(reversed_path)
    free(path_begin)
    return sqrt(result)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def distance_matrix(cur, double max_dist=inf, int max_length_diff=0,
                    int window=0, double max_step=0, double penalty=0, **kwargs):
    """Compute a distance matrix between all sequences given in `cur`.
    """
    if max_length_diff == 0:
        max_length_diff = 999999
    cdef double large_value = inf
    cdef np.ndarray[DTYPE_t, ndim=2] dists = np.zeros((len(cur), len(cur))) + large_value
    for r in range(len(cur)):
        for c in range(r + 1, len(cur)):
            if abs(len(cur[r]) - len(cur[c])) <= max_length_diff:
                dists[r, c] = distance(cur[r], cur[c], window=window,
                                       max_dist=max_dist, max_step=max_step,
                                       max_length_diff=max_length_diff,
                                       penalty=penalty)
    return dists


def distance_matrix_nogil(cur, double max_dist=inf, int max_length_diff=0,
                          int window=0, double max_step=0, double penalty=0, bool is_parallel=False, **kwargs):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the dtw computation that
    avoids the GIL.
    """
    # https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
    # Prepare for only c datastructures
    if max_length_diff == 0:
        max_length_diff = 999999
    cdef double large_value = inf
    dists_py = np.zeros((len(cur), len(cur))) + large_value
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] dists = dists_py
    #print('dists: {}, {}'.format(dists_py.shape, dists_py.shape[0]*dists_py.shape[1]))
    cdef double **cur2 = <double **> malloc(len(cur) * sizeof(double*))
    cdef int *cur2_len = <int *> malloc(len(cur) * sizeof(int))
    cdef long ptr;
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] cur_np;
    #for i in range(len(cur)):
    #    print(cur[i])
    if type(cur) in [list, set]:
        for i in range(len(cur)):
            ptr = cur[i].ctypes.data
            cur2[i] = <double *> ptr
            cur2_len[i] = len(cur[i])
    elif isinstance(cur, np.ndarray):
        if not cur.flags.c_contiguous:
            cur = cur.copy(order='C')
        cur_np = cur
        for i in range(len(cur)):
            cur2[i] = &cur_np[i,0]
            cur2_len[i] = cur_np.shape[1]
    else:
        return None
    if is_parallel:
        distance_matrix_nogil_c_p(cur2, len(cur), cur2_len, &dists[0,0], max_dist, max_length_diff, window, max_step, penalty)
    else:
        distance_matrix_nogil_c(cur2, len(cur), cur2_len, &dists[0,0], max_dist, max_length_diff, window, max_step, penalty)
    free(cur2)
    free(cur2_len)
    return dists_py


def distance_matrix_nogil_p(cur, double max_dist=inf, int max_length_diff=0,
                          int window=0, double max_step=0, double penalty=0, **kwargs):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the dtw computation that
    avoids the GIL and executes them in parallel.
    """
    return distance_matrix_nogil(cur, max_dist=max_dist, max_length_diff=max_length_diff,
                                 window=window, max_step=max_step, penalty=penalty,
                                 is_parallel=True, **kwargs)


cdef distance_matrix_nogil_c(double **cur, int len_cur, int* cur_len, double* output,
                             double max_dist=0, int max_length_diff=0,
                             int window=0, double max_step=0, double penalty=0):
    #for i in range(len_cur):
    #    print(i)
    #    print(cur_len[i])
    #    for j in range(cur_len[i]):
    #        printf("%f ", cur[i][j])
    #    printf("\n")
    #printf("---\n")
    cdef int r
    cdef int c
    for r in range(len_cur):
        for c in range(r + 1, len_cur):
            output[len_cur*r + c] = distance_nogil_c(cur[r], cur[c], cur_len[r], cur_len[c],
                                                     window=window, max_dist=max_dist,
                                                     max_step=max_step, max_length_diff=max_length_diff,
                                                     penalty=penalty)
            #for i in range(len_cur):
            #    for j in range(len_cur):
            #        printf("%f ", output[i*len_cur+j])
            #    printf("\n")
            #printf("---\n")


cdef distance_matrix_nogil_c_p(double **cur, int len_cur, int* cur_len, double* output,
                             double max_dist=0, int max_length_diff=0,
                             int window=0, double max_step=0, double penalty=0):
    # Requires openmp which is not supported for clang on mac
    cdef Py_ssize_t r
    cdef Py_ssize_t c

    with nogil, parallel():
        for r in prange(len_cur):
            for c in range(r + 1, len_cur):
                output[len_cur*r + c] = distance_nogil_c(cur[r], cur[c], cur_len[r], cur_len[c],
                                                         window=window, max_dist=max_dist,
                                                         max_step=max_step, max_length_diff=max_length_diff,
                                                         penalty=penalty)

