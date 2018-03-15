import numpy as np
from numpy.lib.stride_tricks import as_strided
from find_shapelets_simpilified import get_training_data,separate_state,separating_list_dict,list_to_ndarray, removing_faulty_readings, ShapeletFinder
import matplotlib.pyplot as plt
import shapelet_utils

dim_s = (0,)
sigma_min = .3
ord = 1
metric = "cityblock"


def distances(s, t_subs, ord=ord):
    """
    Calculates the distances between a shapelet and a list of time series subequences.
    :param s: shapelet
    :type s: np.array, shape = (len(s), len(dim(s)))
    :param t_subs: list of subsequences from a time series t
    :type t_subs: np.array, shape = (len(t_subs), len(s), len(dim(s)))
    :param ord: 1 for manhattan, 2 for euclidean distance
    :type ord: int
    :return: list of distance between the shapelet and all subsequences for all dimensions
    :rtype: np.array, shape = (len(t), len(dim(s)))
    """
    return (np.linalg.norm((s - t_subs), ord=ord, axis=1)) / s.shape[0]


def z_normalize(t):
    """
    :param t: list of time series subsequences
    :type t: np.array, shape = (len(t), len(s), len(dim(s)))
    :return: list of z-normalized time series subsequences
    :rtype: np.array, shape = (len(t), len(s), len(dim(s)))
    """
    std = np.std(t, axis=1)
    if isinstance(std, float):
        if std < sigma_min:
            std = 1
    else:
        std[std < sigma_min] = 1.
    tmp_ts = ((t.swapaxes(0, 1) - np.mean(t, axis=1)) / std).swapaxes(0, 1)
    return tmp_ts


def subsequences(t, len_s):
    """
    :param t: multidimensional time series
    :type t: np.array, shape = (len(t), len(dim(t)))
    :param len_s: len(s), desired subsequence length
    :type len_s : intunfaulty_list_nd_array
    :return: list of all len_s long subsequences from t
    :rtype: np.array, shape = (len(t), len(s), len(dim(t)))
    """
    if t.ndim == 1:
        m = 1 + t.size - len_s
        s = t.itemsize
        shapelets = as_strided(np.copy(t), shape=(m, len_s), strides=(s, s))
        return shapelets
    else:
        shapelets = None
        for i in range(t.shape[1]):
            next_dim = subsequences(t[:, i], len_s)[..., None]
            if shapelets is None:
                shapelets = next_dim
            else:
                shapelets = np.concatenate((shapelets, next_dim), axis=2)
        return shapelets


def dist_shapelet_ts(s, t, dim_s):
    """
    :param s: a shapelet
    :type s: np.array, shape = (len(s), len(dim(s)))
    :param t: time series with length len(t) and at least d many dimensions
    :type t: np.array, shape = (len(t), len(dim(t))) with len(dim(s)) <= len(dim(t))
    :param dim_s: dim(s), ids of the shapelets dimensions
    :type dim_s: np.array, shape = (len(dim(s)),) with dim_s \in dim(t)
    :return: distances between the shapelet and all subsequences in t
    :rtype: np.array, shape = (len(t),)
    """
    subs = subsequences(t, s.shape[0])  # (len(x), len(shapelet), axis)
    subs = z_normalize(subs)  # (len(x), len(shapelet), axis)
    return distances(s, subs[:, :, dim_s]).mean(axis=1)  # (len(x),)


def estimate_sigma_min(self):
    """
    Estimates $\sigma_{min}$ by using the maximum standard deviation of shapelets in time series
    without label.
    """
    if self.sigma_min is None:
        sigma_min = 0
        for id, labels in enumerate(self.target):
            if len(labels) == 0:
                ts_subs = subsequences(self.data[id], min(self.windows))
                sigma_min = max(sigma_min, ts_subs.std(axis=1).max())
        print("sigma_min set to {}".format(sigma_min))
        self.sigma_min = sigma_min
    shapelet_utils.sigma_min = self.sigma_min
    return self.sigma_min


def divide_time_series(time_series):
    time_series = time_series[0:1600]
    sub_time_series = np.split(time_series, 32)
    return sub_time_series


def main():


    dict_training_data = get_training_data()
    list_dict = separate_state(dict_training_data)
    data_denso, states_denso = separating_list_dict(list_dict)
    list_nd_array, nd_states_dict, list_nd_time = list_to_ndarray(data_denso, states_denso)
    unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time = removing_faulty_readings(list_nd_array,
                                                                                                      nd_states_dict,list_nd_time)
    find_shapelet = ShapeletFinder()
    bsf_classifier, shapelets = find_shapelet.findingshapelets(unfaulty_list_nd_array, unfaulty_nd_states_dict)
    dict_shapelets = dict(shapelets)
    state_1 = '59d638667bfe0b5f22bd6427: Motek - White Part Unmount'
    state_2 = '59d638667bfe0b5f22bd6424: Pick Erebus'
    state_3 = '59d638667bfe0b5f22bd645d: Mount Erebus'
    state_4 = '59d638667bfe0b5f22bd6446: Pitasc-Sub - White Part Mount Tilted'

    sigma_min = estimate_sigma_min(find_shapelet)
    shapelets = dict_shapelets[state_1]
    time_series = unfaulty_list_nd_array[0]
    sub_time_series = divide_time_series(time_series)
    distances = list()
    for shapelet in shapelets:
        for sub in sub_time_series:
            distance = dist_shapelet_ts(shapelet, sub, dim_s)
            distances.append(distance)
    nd_distances = np.array(distances)
    plt.plot(nd_distances)
    plt.ylabel('BMD')
    plt.show()
    print "Finishing..."

if __name__ == '__main__':

    main()

