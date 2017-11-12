"""
dtaidistance.dtw - Dynamic Time Warping
__license__ = "APL"
..
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
import os
import logging
import math
import numpy as np
import pyximport
pyximport.install()

logger = logging.getLogger("be.kuleuven.dtai.distance")
dtaidistance_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)

try:
    from . import dtw_c
except ImportError:
    # logger.info('C library not available')
    dtw_c = None

try:
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm library not available')
    tqdm = None


def lb_keogh(s1, s2, window=None, max_dist=None,
             max_step=None, max_length_diff=None):
    """Lowerbound LB_KEOGH"""
    # TODO: This implementation slower than distance() itself
    if window is None:
        window = max(len(s1), len(s2))

    t = 0
    for i in range(len(s1)):
        imin = max(0, i - max(0, len(s1) - len(s2)) - window + 1)
        imax = min(len(s2), i + max(0, len(s2) - len(s1)) + window)
        ui = np.max(s2[imin:imax])
        li = np.min(s2[imin:imax])
        ci = s1[i]
        if ci > ui:
            t += abs(ci - ui)
        else:
            t += abs(ci - li)
    return t


def distance(s1, s2, window=None, max_dist=None,
             max_step=None, max_length_diff=None, penalty=None,
             use_nogil=False):
    """
    Dynamic Time Warping (keep compact matrix)
    :param s1: First sequence
    :param s2: Second sequence
    :param window: Only allow for shifts up to this amount away from the two diagonals
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param max_length_diff: Return infinity if length of two series is larger
    :param penalty: Penalty to add if compression or expansion is applied
    :param use_nogil: Use fast pure c compiled functions
    Returns: DTW distance
    """
    if use_nogil:
        return distance_fast(s1, s2, window,
                             max_dist=max_dist,
                             max_step=max_step,
                             max_length_diff=max_length_diff,
                             penalty=penalty)
    s1_len, s2_len = len(s1), len(s2)
    if max_length_diff is not None and abs(s1_len - s2_len) > max_length_diff:
        return np.inf
    if window is None:
        window = max(s1_len, s2_len)
    if not max_step:
        max_step = np.inf
    else:
        max_step *= max_step
    if not max_dist:
        max_dist = np.inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    length = min(s2_len + 1, abs(s1_len - s2_len) + 2 * (window - 1) + 1 + 1 + 1)
    dtw = np.full((s1_len + 1, s2_len + 1), np.inf)
    dtw[0, 0] = 0
    last_under_max_dist = 0
    i0 = 1
    i1 = 0
    for i in range(s1_len):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = np.inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        i0 = i
        i1 = i + 1
        for j in range(s2_len):
            dist = (s1[i] - s2[j])**2
            if dist > max_step:
                continue
            assert j + 1 >= 0
            dtw[i1, j + 1] = dist
            if j == s2_len - 1:
                dtw[i1, j + 1] = min(dist + dtw[i0, j],
                                     dtw[i0, j + 1],
                                     dist + dtw[i1, j] + penalty)
            elif j == 0:
                dtw[i1, j + 1] = dist
            else:
                dtw[i1, j + 1] = dist + min(dtw[i0, j],
                                            dtw[i0, j + 1] + penalty,
                                            dtw[i1, j] + penalty)
            if dtw[i1, j + 1] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1, j + 1] = np.inf
                if prev_last_under_max_dist < j + 1:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            return np.inf
    result = dtw[s1_len, s2_len]
    return np.sqrt(result)


def distance_fast(s1, s2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None):
    """Fast C version of distance()"""
    if dtw_c is None:
        _print_library_missing()
        return None
    if window is None:
        window = 0
    if max_dist is None:
        max_dist = 0
    if max_step is None:
        max_step = 0
    if max_length_diff is None:
        max_length_diff = 0
    if penalty is None:
        penalty = 0
    dist = dtw_c.distance_nogil(s1, s2, window,
                                max_dist=max_dist,
                                max_step=max_step,
                                max_length_diff=max_length_diff,
                                penalty=penalty)
    return dist


def _distance_with_params(t):
    return distance(t[0], t[1], **t[2])


def _distance_c_with_params(t):
    return dtw_c.distance(t[0], t[1], **t[2])


def distances(s1, s2, window=None, max_dist=None,
              max_step=None, max_length_diff=None, penalty=None):
    """
    Dynamic Time Warping (keep full matrix)
    :param s1: First sequence
    :param s2: Second sequence
    :param window: Only allow for shifts up to this amount away from the two diagonals
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param max_length_diff: Return infinity if length of two series is larger
    :param penalty: Penalty to add if compression or expansion is applied
    Returns: DTW distance, DTW matrix
    """
    rows, columns = len(s1), len(s2)
    if max_length_diff is not None and abs(rows - columns) > max_length_diff:
        return np.inf
    if window is None:
        window = max(rows, columns)
    if not max_step:
        max_step = np.inf
    else:
        max_step *= max_step
    if not max_dist:
        max_dist = np.inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    dtw = np.full((rows + 1, columns + 1), np.inf)
    dtw[0, 0] = 0
    last_under_max_dist = 0
    i0 = 1
    i1 = 0
    for i in range(rows):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = np.inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        i0 = i
        i1 = i + 1
        for j in range(columns):
            dist = (s1[i] - s2[j])**2
            if max_step is not None and dist > max_step:
                continue
            dtw[i1, j + 1] = dist
            if j == columns - 1:
                dtw[i1, j + 1] = min(dist + dtw[i0, j],
                                     dtw[i0, j + 1],
                                     dist + dtw[i1, j] + penalty)
            elif j == 0:
                dtw[i1, j + 1] = dist
            else:
                dtw[i1, j + 1] = dist + min(dtw[i0, j],
                                            dtw[i0, j + 1] + penalty,
                                            dtw[i1, j] + penalty)
            if max_dist is not None:
                if dtw[i1, j + 1] <= max_dist:
                    last_under_max_dist = j
                else:
                    dtw[i1, j + 1] = np.inf
                    if prev_last_under_max_dist < j + 1:
                        break
        if max_dist is not None and last_under_max_dist == -1:
            return np.inf, dtw
    dtw = np.sqrt(dtw)
    return dtw[rows, columns], dtw


def distance_matrix_func(use_c=False, use_nogil=False, parallel=False, show_progress=False):
    def distance_matrix_wrapper(seqs, **kwargs):
        return distance_matrix(seqs, parallel=parallel, use_c=use_c,
                               use_nogil=use_nogil,
                               show_progress=show_progress, **kwargs)
    return distance_matrix_wrapper


def distance_matrix(s, max_dist=None, max_length_diff=None,
                    window=None, max_step=None, penalty=None,
                    parallel=False,
                    use_c=False, use_nogil=False, show_progress=False):
    """Distance matrix for all sequences in s.
    :param s: Iterable of series
    :param window: Only allow for shifts up to this amount away from the two diagonals
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param max_length_diff: Return infinity if length of two series is larger
    :param penalty: Penalty to add if compression or expansion is applied
    :param parallel: Use parallel operations
    :param use_c: Use c compiled Python functions
    :param use_nogil: Use pure c functions
    :param show_progress: Show progress using the tqdm library
    """
    if parallel and (not use_c or not use_nogil):
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            parallel = False
            mp = None
    else:
        mp = None
    dist_opts = {
        'max_dist': max_dist,
        'max_step': max_step,
        'window': window,
        'max_length_diff': max_length_diff,
        'penalty': penalty
    }
    dists = None
    if max_length_diff is None:
        max_length_diff = np.inf
    large_value = np.inf
    logger.info('Computing distances')
    if use_c:
        for key, value in dist_opts.items():
            if value is None:
                dist_opts[key] = 0.0
    if use_c and use_nogil:
        logger.info("Compute distances in pure C")
        if parallel:
            logger.info("Use parallel computation")
            dists = dtw_c.distance_matrix_nogil_p(s, **dist_opts)
        else:
            logger.info("Use serial computation")
            dists = dtw_c.distance_matrix_nogil(s, **dist_opts)
    if use_c and not use_nogil:
        logger.info("Compute distances in Python compiled C")
        if parallel:
            logger.info("Use parallel computation")
            dists = np.zeros((len(s), len(s))) + large_value
            idxs = np.triu_indices(len(s), k=1)
            with mp.Pool() as p:
                dists[idxs] = p.map(_distance_c_with_params,
                                    [(s[row], s[column], dist_opts) for column, row in zip(*idxs)])
        else:
            logger.info("Use serial computation")
            dists = dtw_c.distance_matrix(s, **dist_opts)
    if not use_c:
        logger.info("Compute distances in Python")
        if isinstance(s, np.matrix):
            s_reshaped = [np.asarray(s[i]).reshape(-1) for i in range(s.shape[0])]
            s = s_reshaped
        if parallel:
            logger.info("Use parallel computation")
            dists = np.zeros((len(s), len(s))) + large_value
            idxs = np.triu_indices(len(s), k=1)
            with mp.Pool() as p:
                dists[idxs] = p.map(_distance_with_params,
                                    [(s[row], s[column], dist_opts) for column, row in zip(*idxs)])
        else:
            logger.info("Use serial computation")
            dists = np.zeros((len(s), len(s))) + large_value
            it_r = range(len(s))
            if show_progress:
                it_r = tqdm(it_r)
            for row in it_r:
                for column in range(row + 1, len(s)):
                    if abs(len(s[row]) - len(s[column])) <= max_length_diff:
                        first_dist = distance(s[row], s[column], **dist_opts)
                        second_dist = distance(s[column], s[row], **dist_opts)
                        dists[row, column] = min(first_dist, second_dist)

    return dists


def distance_matrix_fast(s, max_dist=None, max_length_diff=None,
                         window=None, max_step=None, penalty=None,
                         parallel=True, show_progress=False):
    """Fast C version of distance_matrix()"""
    if dtw_c is None:
        _print_library_missing()
        return None
    return distance_matrix(s, max_dist=max_dist, max_length_diff=max_length_diff,
                           window=window, max_step=max_step, penalty=penalty,
                           parallel=parallel,
                           use_c=True, use_nogil=True, show_progress=show_progress)


def warp_path(from_s, to_s, **kwargs):
    dists = distances(from_s, to_s, **kwargs)
    if kwargs is not None:
        if 'penalty' in kwargs:
            penalty = kwargs['penalty']
        else:
            penalty = 0
    else:
        penalty = 0
    matrix = dists[1]
    path = []
    row = np.shape(matrix)[0]
    column = np.shape(matrix)[1]
    y_size = np.shape(matrix)[0]
    x_size = np.shape(matrix)[1]
    v = matrix[row - 1, column - 1]
    path.append((row - 2, column - 2))
    while row != 2 or column != 2:
        dist = (from_s[row - 2] - to_s[column - 2])**2
        if column == x_size:
            cells = [matrix[row - 2, column - 2] + dist,
                     matrix[row - 2, column - 1],
                     matrix[row - 1, column - 2] + dist + penalty]
        elif column == 2:
            cells = [np.inf, dist, np.inf]
        else:
            cells = [matrix[row - 2, column - 2],
                     matrix[row - 2, column - 1] + penalty,
                     matrix[row - 1, column - 2] + penalty]
        ind = np.argmin(cells)
        if ind == 0:
            path.append((row - 3, column - 3))
            row -= 1
            column -= 1
        if ind == 1:
            path.append((row - 3, column - 2))
            row -= 1
        if ind == 2:
            path.append((row - 2, column - 3))
            column -= 1
    path.reverse()
    return path


def warp(from_s, to_s, plot=False, **kwargs):
    """Warp a function to optimally match a second function.
    Same options as distances().
    """
    path = warp_path(from_s, to_s, **kwargs)
    from_s2 = np.zeros(len(to_s))
    from_s2_cnt = np.zeros(len(to_s))
    for ind_s1, ind_s2 in path:
        from_s2[ind_s2] += from_s[ind_s1]
        from_s2_cnt[ind_s2] += 1
    from_s2 /= from_s2_cnt

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        ax[0].plot(from_s, label="From")
        ax[0].legend()
        ax[1].plot(to_s, label="To")
        ax[1].legend()
        transFigure = fig.transFigure.inverted()
        lines = []
        line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
        for r_c, c_c in path:
            coord1 = transFigure.transform(ax[0].transData.transform([r_c, from_s[r_c]]))
            coord2 = transFigure.transform(ax[1].transData.transform([c_c, to_s[c_c]]))
            lines.append(mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                          transform=fig.transFigure, **line_options))
        ax[2].plot(from_s2, label="From-warped")
        ax[2].legend()
        for i in range(len(to_s)):
            coord1 = transFigure.transform(ax[1].transData.transform([i, to_s[i]]))
            coord2 = transFigure.transform(ax[2].transData.transform([i, from_s2[i]]))
            lines.append(mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                          transform=fig.transFigure, **line_options))
        fig.lines = lines
        plt.show(block=True)

    return from_s2


def plot_warping(s1, s2, **kwargs):
    path = warp_path(s1, s2, **kwargs)

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    ax[0].plot(s1)
    ax[1].plot(s2)
    transFigure = fig.transFigure.inverted()
    lines = []
    line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    for r_c, c_c in path:
        coord1 = transFigure.transform(ax[0].transData.transform([r_c, s1[r_c]]))
        coord2 = transFigure.transform(ax[1].transData.transform([c_c, s2[c_c]]))
        lines.append(mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                      transform=fig.transFigure, **line_options))
    fig.lines = lines
    plt.show(block=True)


def subsequence(s1, s2, penalty=0):
    path = warp_path(s1, s2, penalty=penalty)
    left_bound = 0
    right_bound = len(s2) - 1
    fl = 1
    for pair in path:
        if pair[1] == left_bound:
            begin = pair
        if pair[1] == right_bound and fl:
            end = pair
            fl = 0
    return begin[0], end[0]


def mean_cluster(clust_data):
    mean_s = medoid(clust_data)
    iter_num = 30  # set any iter_num > 0
    for i in range(iter_num):
        mean_s = mean_update(mean_s, clust_data)
    return mean_s


def medoid(clust_data):
    matrix = distance_matrix(clust_data)
    size = np.shape(matrix)[0]
    for i in range(size):
        for j in range(size):
            if i > j:
                matrix[i][j] = matrix[j][i]
            if i == j:
                matrix[i][j] = 0
    min_ind = np.argmin([np.sum([matrix[i][j]**2 for j in range(len(matrix[i]))])
                         for i in range(size)])
    return clust_data[min_ind]


def mean_update(mean_s, clust_data):
    length = len(mean_s)
    alignment = [[] for i in range(length)]
    for el in clust_data:
        alignment_el = dtw_multiple_alignment(mean_s, el)
        for i in range(length):
            for j in range(len(alignment_el[i])):
                alignment[i].append(alignment_el[i][j])
    mean_sequence = [0.0 for i in range(length)]
    for i in range(length):
        mean_sequence[i] = np.mean(alignment[i])
    return mean_sequence


def dtw_multiple_alignment(mean_s, el):
    length = len(mean_s)
    path = warp_path(el, mean_s)
    alignment = [[] for i in range(length)]
    j = 0
    for i in range(length):
        while path[j][1] == i:
            alignment[i].append(el[path[j][0]])
            j += 1
            if j == len(path):
                break
    return alignment


def _print_library_missing():
    logger.error("The compiled dtaidistance c library is not available.\n" +
                 "Run `cd {};python3 setup.py build_ext --inplace`.".format(dtaidistance_dir))

