import numpy as np
from numpy.lib.stride_tricks import as_strided
from find_shapelets_simpilified import get_training_data,separate_state,separating_list_dict,list_to_ndarray, removing_faulty_readings, separate_state, ShapeletFinder
from find_shapelets import ConfusionMatrix
from collections import defaultdict
from classifier import ShapeletClassifier
import matplotlib.pyplot as plt


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


def divide_time_series(time_series):
    time_series = time_series[0:1000]
    sub_time_series = np.split(time_series, 20)
    return sub_time_series

def another_approach(self, time_series, bsf_classifier):

    self.confusion_matrix = defaultdict(lambda: ConfusionMatrix())
    x = time_series
    for label, (classifier, _) in bsf_classifier.items():
        self.confusion_matrix[label].deltas.append(classifier.delta)
        try:
            self.confusion_matrix[label].sec_ig.append(classifier.y)
        except:
            self.confusion_matrix[label].sec_ig.append(classifier.f_c_delta)

        self.confusion_matrix[label].shapelet_lengths.append(
            classifier.shapelet.shape[0])
        self.confusion_matrix[label].axis.append(classifier.dim_s)
        shapelet_length = classifier.shapelet.shape[0]
        shapelet_matches = np.array(classifier.predict_all(x)) + shapelet_length // 2


def main():


    dict_training_data = get_training_data()
    data_of_interest = dict_training_data['59d638667bfe0b5f22bd6427: Motek - White Part Unmount'][0]
    data = []
    targets = []

    for state_data in (data_of_interest):
        data.append(state_data.values())
    data_nd = np.array(data)
    data_nd = data_nd[:,0:3]

    list_dict = separate_state(dict_training_data)
    data_denso, states_denso = separating_list_dict(list_dict)
    list_nd_array, nd_states_dict, list_nd_time = list_to_ndarray(data_denso, states_denso)
    unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time = removing_faulty_readings(list_nd_array,
                                                                                                      nd_states_dict,list_nd_time)
    find_shapelet = ShapeletFinder()
    bsf_classifier, shapelets = find_shapelet.findingshapelets(unfaulty_list_nd_array, unfaulty_nd_states_dict)
    states = ['59d638667bfe0b5f22bd6427: Motek - White Part Unmount', '59d638667bfe0b5f22bd6424: Pick Erebus',
              '59d638667bfe0b5f22bd645d: Mount Erebus', '59d638667bfe0b5f22bd6446: Pitasc-Sub - White Part Mount Tilted']
    time_series = data_nd
    #distances = list()
    time_indicies = []
    #sub_time_series = divide_time_series(time_series)
    shapelets_dict = dict(shapelets)
    #for state in states:
    # for shapelet in shapelets_dict[states[0]]:
    #     classifier = ShapeletClassifier(shapelet)
    #     classifier.delta = 0.6435830967784079
    #     mins, ds = classifier.predict_all(time_series)
    #     time_indicies.append(mins)
    #     distances.append(ds)
    another_approach(find_shapelet, time_series, bsf_classifier)

    #nd_distances = np.array(distances)
    #classifier_threshold = 0.6435830967784079 # '59d638667bfe0b5f22bd6427: Motek - White Part Unmount'
    # for distance in distances:
    #     if distance < classifier_threshold:
    #         print"Successful process!"
    #         break

    #plt.plot(nd_distances)
    plt.ylabel('BMD')
    plt.show()
    print "Finishing..."

if __name__ == '__main__':

    main()

