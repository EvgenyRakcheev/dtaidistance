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
    cdef int rows = len(s1)
    cdef int columns = len(s2)
    if max_length_diff != 0 and abs(rows - columns) > max_length_diff:
        return inf
    if window == 0:
        window = max(rows, columns)
    if max_step == 0:
        max_step = inf
    else:
        max_step *= max_step
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist *= max_dist
    penalty *= penalty
    cdef np.ndarray[DTYPE_t, ndim=2] dtw = np.full((2, min(columns + 1,
                                                    abs(rows - columns) + 2 *(window -1) + 3)), inf)
    dtw[0, 0] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int skip = 0
    cdef int skipp = 0
    cdef int i0 = 1
    cdef int i1 = 0
    cdef DTYPE_t d
    for i in range(rows):
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
        if dtw.shape[1] == columns + 1:
            skip = 0
        for j in range(max(0, i - max(0, rows - columns) - window + 1),
                       min(columns, i + max(0, columns - rows) + window)):
            dist = (s1[i] - s2[j])**2
            if dist > max_step:
                continue
            dtw[i1, j + 1 - skip] = dist + min(dtw[i0, j - skipp],
                                               dtw[i0, j + 1 - skipp] + penalty,
                                               dtw[i1, j - skip] + penalty)
            if dtw[i1, j + 1 - skip] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1, j + 1 - skip] = inf
                if prev_last_under_max_dist + 1 - skipp < j + 1 - skip:
                    break
        if last_under_max_dist == -1:
            return inf
    return math.sqrt(dtw[i1, min(columns, columns + window - 1) - skip])


def distance_nogil(double[:] s1, double[:] s2,
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0):
    """DTW distance.
    See distance(). This calls a pure c dtw computation that avoids the GIL.
    :param s1: First sequence (buffer of doubles)
    :param s2: Second sequence (buffer of doubles)
    """
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
             int row,  # len_s1
             int column,  # len_s2
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0) nogil:
    """DTW distance.
    See distance(). This is a pure c dtw computation that avoid the GIL.
    """
    if max_length_diff != 0 and abs(row - column) > max_length_diff:
        return inf
    if window == 0:
        window = max(row, column)
    if max_step == 0:
        max_step = inf
    else:
        max_step = pow(max_step, 2)
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist = pow(max_dist, 2)
    penalty = pow(penalty, 2)
    cdef double ** dtw
    dtw = <double **> malloc(sizeof(double *) * (row + 1))
    cdef int i
    for i in range(row + 1):
        dtw[i] = <double *> malloc(sizeof(double) * (column + 1))
    cdef int j
    for i in range(row + 1):
        for j in range(column + 1):
            dtw[i][j] = inf
    dtw[0][0] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int i0 = 1
    cdef int i1 = 0
    cdef double minv
    cdef DTYPE_t dist
    cdef double tempv
    for i in range(row):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        i0 = i
        i1 = i + 1
        for j in range(column):
            dist = pow(s1[i] - s2[j], 2)
            if dist > max_step:
                continue
            if j == column - 1:
                minv = dist + dtw[i0][j]
                tempv = dtw[i0][j + 1]
                if tempv < minv:
                    minv = tempv
                tempv = dist + dtw[i1][j] + penalty
                if tempv < minv:
                    minv = tempv
                dtw[i1][j + 1] = minv
            else:
                if j == 0:
                    dtw[i1][j + 1] = dist
                else:
                    minv = dtw[i0][j]
                    tempv = dtw[i0][j + 1] + penalty
                    if tempv < minv:
                        minv = tempv
                    tempv = dtw[i1][j] + penalty
                    if tempv < minv:
                        minv = tempv
                    dtw[i1][j + 1] = dist + minv
            if dtw[i1][j + 1] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1][j + 1] = inf
                if prev_last_under_max_dist < j + 1:
                    break
        if last_under_max_dist == -1:
            return inf
    cdef double result = dtw[row][column]
    for i in range(row + 1):
        free(dtw[i])
    free(dtw)
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
    for row in range(len(cur)):
        for column in range(row + 1, len(cur)):
            if abs(len(cur[row]) - len(cur[column])) <= max_length_diff:
                dists[row, column] = distance(cur[row], cur[column], window=window,
                                              max_dist=max_dist, max_step=max_step,
                                              max_length_diff=max_length_diff,
                                              penalty=penalty)
    return dists


def distance_matrix_nogil(cur, double max_dist=inf, int max_length_diff=0,
                          int window=0, double max_step=0, double penalty=0,
                          bool is_parallel=False, **kwargs):
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
    cdef double **cur2 = <double **> malloc(len(cur) * sizeof(double*))
    cdef int *cur2_len = <int *> malloc(len(cur) * sizeof(int))
    cdef long ptr;
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] cur_np;
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
        distance_matrix_nogil_c_p(cur2, len(cur), cur2_len, &dists[0,0],
                                  max_dist, max_length_diff, window, max_step, penalty)
    else:
        distance_matrix_nogil_c(cur2, len(cur), cur2_len, &dists[0,0],
                                max_dist, max_length_diff, window, max_step, penalty)
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
    cdef int row
    cdef int column
    cdef int left
    cdef int right
    for row in range(len_cur):
        for column in range(row + 1, len_cur):
            left = row
            right = column
            if cur_len[row] < cur_len[column]:
                left = column
                right = row
            output[len_cur*row + column] = distance_nogil_c(cur[left], cur[right],
                                                            cur_len[left], cur_len[right],
                                                            window=window, max_dist=max_dist,
                                                            max_step=max_step,
                                                            max_length_diff=max_length_diff,
                                                            penalty=penalty)


cdef distance_matrix_nogil_c_p(double **cur, int len_cur, int* cur_len, double* output,
                             double max_dist=0, int max_length_diff=0,
                             int window=0, double max_step=0, double penalty=0):
    # Requires openmp which is not supported for clang on mac
    cdef Py_ssize_t row
    cdef Py_ssize_t column
    cdef int left
    cdef int right
    with nogil, parallel():
        for row in prange(len_cur):
            for column in range(row + 1, len_cur):
                left = row
                right = column
                if cur_len[row] < cur_len[column]:
                    left = column
                    right = row
                output[len_cur*row + column] = distance_nogil_c(cur[left], cur[right],
                                                                cur_len[left], cur_len[right],
                                                                window=window, max_dist=max_dist,
                                                                max_step=max_step,
                                                                max_length_diff=max_length_diff,
                                                                penalty=penalty)

