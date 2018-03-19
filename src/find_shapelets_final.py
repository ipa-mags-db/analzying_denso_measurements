#!/usr/bin/env python
import argparse
import csv
import numpy as np
import time
from collections import defaultdict
from scipy.signal._peak_finding import argrelmax, argrelmin
import pylab as plt
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from import_csv_db import import_db
from utilities import keydefaultdict, powerset, find_nearest, Counter
from shapelet_utils import subsequences, z_normalize, distance_matrix3D
import shapelet_utils
from clustering import Clustering
from classifier import ShapeletClassifier
from find_shapelets_simpilified import get_training_data,separate_state,separating_list_dict,list_to_ndarray, removing_faulty_readings
import pandas as pd
import pickle

BLUE = "#2b83ba"
RED = "#d7191c"
GREEN = '#abdda4'
LT = 3
FONTSIZE = 14
colors = [RED, GREEN, BLUE, "k", "y", "m"]
label = ["x", "y", "z", "rx", "ry", "rz"]
import matplotlib

matplotlib.rcParams.update({'font.size': FONTSIZE})


class ShapeletFinder(object):
    def __init__(self, d_max=.6, N_max=3, w_ext=25, sigma_min=None, sl_max=50):
        """
        :param d_max: cluster radius
        :type d_max: float
        :param N_max: number of shapelet lengths to test between 0 and 'sl_max'
        :type N_max: int
        :param w_ext: window size for extrema pruning
        :type w_ext: int
        :param sigma_min: z-normalization threshold, will be estimated if it is None
        :type sigma_min: float or None
        :param sl_max: maximum shapelet length
        :type sl_max: int
        """
        self.d_max = d_max
        self.N_max = N_max
        self.w_ext = w_ext
        self.sigma_min = sigma_min
        self.sl_max = sl_max
        self.reset()

    def reset(self):
        """
        Deletes stored computations.
        """
        self.minima = keydefaultdict(lambda (key, axis): argrelmin(self.data[key][:, axis], order=self.w_ext)[0])
        self.maxima = keydefaultdict(lambda (key, axis): argrelmax(self.data[key][:, axis], order=self.w_ext)[0])
        self.derivative_minima = keydefaultdict(
            lambda (key, axis): argrelmin(np.diff(self.data[key][:, axis]), order=self.w_ext)[0])
        self.derivative_maxima = keydefaultdict(
            lambda (key, axis): argrelmax(np.diff(self.data[key][:, axis]), order=self.w_ext)[0])

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

    def get_unique_targets(self, target):
        """
        :param target: list that contains the event labels for each training instance.
        :type target: np.array
        :return: set of unique labels from target
        :rtype: set(string)
        """
        labels = set()
        for row in target:
            for label in row:
                labels.add(label)
        return labels

    def calc_windows(self, sl_max, N_max):
        """
        Reduces the number of possible shapelet lengths using two parameters.
        :param sl_max: maximum shapelet length
        :type sl_max: int
        :param N_max: number of shapelet lengths between 0 and sl_max
        :type N_max: int
        :return: list of shapelet lengths that will be tested
        :rtype: list(int)
        """
        windows = [(sl_max * i) / N_max for i in range(1, int(N_max + 1))]
        print("possible shapelet length {}".format(windows))
        return windows

    def cmp_classifier(self, classifier1, classifier2):
        """
        Returns the classifier with the highest information gain, or f_c if the gain is equal.
        :return: 1, if classifier1 < classifier2
                 -1, if classifier1 > classifier2
        :rtype: int
        """

        def cmp(x, y):
            c1 = x[0]
            c2 = y[0]
            gain_x = c1.information_gain
            gain_y = c2.information_gain
            f_c_delta1 = c1.f_c_delta
            f_c_delta2 = c2.f_c_delta
            if gain_x > gain_y or (gain_x == gain_y and f_c_delta1 < f_c_delta2):
                return -1
            else:
                return 1

        return cmp((classifier1,),
                   (classifier2,))

    def cluster(self, shapelets):
        """
        Uses a clustering algorithm to reduce the number of shapelets.
        :param shapelets: list of shapelet candidates
        :type shapelets: np.array, shape = (len(shapelets), len(s), len(dim(s)))
        :return: list of remaining shapelet candidates
        :rtype np.array, shape = (|remaining candidates|, len(s), len(dim(s)))
        """
        clustering = Clustering(self.d_max)
        clustering.fit(shapelets)
        return clustering.nn_centers()

    def findingshapelets(self, data, target):
        """
        Searches for a shapelet classifier for each label.
        :param data: list of training examples
        :type data: np.array
        :param target: list of event labels for each training example
        :type target: np.array
        :return: with label as key and (classifier, target) as value
        :rtype: dict
        """
        self.data = data
        self.target = target
        self.windows = self.calc_windows(self.sl_max, self.N_max)
        self.estimate_sigma_min()
        self.unique_labels = self.get_unique_targets(target)
        bsf_classifier = defaultdict(lambda: None)
        self.shapelets = dict()
        self.dimensions_subsets = list(powerset(range(data[0].shape[1])))[1:]

        self.precompute_z_norm(data)

        c = Counter(len(self.dimensions_subsets) * len(self.windows), prefix="generating shapelets")
        for i, dimension_subset in enumerate(self.dimensions_subsets):
            if dimension_subset == ():
                continue

            for j, window in enumerate(self.windows):
                shapelets = self.prune_shapelet_candidates(window, dimension_subset)
                for label in shapelets.keys():
                    self.shapelets[label, dimension_subset, window] = shapelets[label]
                c.printProgress(j + (i * len(self.windows)) + 1)

        self.precompute_bmd(data)

        for label in self.unique_labels:
            binary_target = np.array([int(label in x) for x in target])
            c = Counter(len(self.dimensions_subsets) * len(self.windows), prefix=label)
            c.printProgress(0)
            for ds_i, dimension_subset in enumerate(self.dimensions_subsets):
                for w_i, window in enumerate(self.windows):
                    key = (label, dimension_subset, window)
                    classifier_candidates = self.build_classifier(self.shapelets[key], binary_target, label,
                                                                  dimension_subset)
                    for c_i, classifier in enumerate(classifier_candidates):
                        try:
                            if self.cmp_classifier(bsf_classifier[label], classifier) > 0:
                                bsf_classifier[label] = classifier
                        except AttributeError:
                            bsf_classifier[label] = classifier
                    c.printProgress(ds_i * len(self.windows) + w_i + 1)
            bsf_classifier[label] = bsf_classifier[label], binary_target
        return bsf_classifier, shapelets

    def precompute_bmd(self, data):
        """
        Calculates the BMD between all shapelet candidates and all training examples.
        :param data: list of training examples
        :type data: np.array
        """
        self.dist_shapelet_ts = dict()
        c = Counter(data.shape[0], prefix="calculating min dist")
        c.printProgress(0)
        for ts_id in range(data.shape[0]):
            for axis in self.dimensions_subsets:
                for shapelet_length in self.windows:
                    muh = np.concatenate([self.shapelets[label, axis, shapelet_length] for label in self.unique_labels])
                    ts = np.concatenate([self.z_data[ts_id, shapelet_length][:, :, a][..., None] for a in axis],
                                        axis=-1)
                    d_m = distance_matrix3D(muh, ts).min(axis=1)
                    i = 0
                    for label in self.unique_labels:
                        key = (label, axis, shapelet_length)
                        for shapelet_id, shapelet in enumerate(self.shapelets[key]):
                            self.dist_shapelet_ts[ts_id, shapelet_id, label, shapelet_length, axis] = d_m[i]
                            i += 1
            c.printProgress(ts_id + 1)

    def precompute_z_norm(self, data):
        """
        Stores the z-norm of every possible shapelet from data in self.z_data.
        :param data: list of training examples
        :type data: np.array
        """
        self.z_data = dict()
        for w in self.windows:
            for ts_id, ts in enumerate(data):
                self.z_data[ts_id, w] = z_normalize(subsequences(ts, w))

    def prune_shapelet_candidates(self, shapelet_length, dim_s=(0,)):
        """
        Employs pruning techniques to reduce the number of shapelet candidates from self.z_data
        :param shapelet_length: length of the shapelets 
        :type shapelet_length: int
        :param dim_s: list of shapelet dimensions
        :type dim_s: tuple(int)
        :return: remaining shapelet candidates
        :rtype: np.array, shape = (|candidates|, shapelet_length, len(dim_s))
        """
        all_shapelets = defaultdict(lambda: None)
        max_candidates = defaultdict(lambda: 0.)
        for ts_id, ts in enumerate(self.data):
            ids = []
            for a in dim_s:
                ids.extend([self.minima[ts_id, a], self.maxima[ts_id, a], self.derivative_minima[ts_id, a],
                            self.derivative_maxima[ts_id, a]])
            ids = np.concatenate([x for x in ids if len(x) > 0])

            ids = ids - shapelet_length // 2
            ids[ids < 0] = 0
            ids[ids > ts.shape[0] - shapelet_length] = ts.shape[0] - shapelet_length
            ids = np.unique(ids)
            shapelets = np.concatenate([self.z_data[ts_id, shapelet_length][:, :, a][..., None] for a in dim_s],
                                       axis=-1)
            for label in self.target[ts_id]:
                max_candidates[label] += shapelets.shape[0]
            shapelets = shapelets[ids]

            for label in self.target[ts_id]:
                try:
                    all_shapelets[label] = np.vstack((all_shapelets[label], shapelets))
                except ValueError:
                    all_shapelets[label] = shapelets

        for k in all_shapelets.keys():
            all_shapelets[k] = self.cluster(all_shapelets[k])
        return all_shapelets

    def build_classifier(self, shapelets, target, label, dim_s=(0,)):
        """
        Creates classifiers for a list of shapelets
        :param shapelets: list of shapelet candidates
        :type shapelets: np.array, shape = (|candidates|, len(s), len(dim(s)))
        :param target: binary target, 1 if training examples contains 'label', 0 otherwise 
        :type target: np.array, shape = (len(dataset))
        :param label: event label for which the 'target' was created
        :type label: str
        :param dim_s: list of dimensions which the classifier has to use
        :type dim_s: tuple(int)
        :return: list containing a classifier for each shapelet
        :rtype: list(ShapeletClassifier)
        """
        classifiers = []
        shapelet_length = shapelets[0].shape[0]
        for shapelet_id, shapelet in enumerate(shapelets):
            cls = ShapeletClassifier(shapelet, dim_s=dim_s)

            dist_X = np.array([self.dist_shapelet_ts[ts_id, shapelet_id, label, shapelet_length, dim_s] for ts_id in
                               range(self.data.shape[0])])
            cls.fit_precomputed(dist_X, target)
            classifiers.append(cls)
        return classifiers


class ConfusionMatrix(object):
    def __init__(self):
        self.TP = 0.
        self.TN = 0.
        self.FP = 0.
        self.FN = 0.
        self.shapelet_lengths = []
        self.deltas = []
        self.ig = []
        self.sec_ig = []
        self.axis = []
        self.time_differences = []
        self.number_of_shapelets = []

    def avg_num_shapelets(self):
        return np.mean(self.number_of_shapelets)

    def avg_sl(self):
        return np.mean(self.shapelet_lengths)

    def delta_mean(self):
        return np.mean(self.deltas)

    def delta_std(self):
        return np.std(self.deltas)

    def axis_num(self):
        return np.mean([len(x) for x in self.axis])

    def recall(self):
        return self.TP / float(self.TP + self.FN)

    def fnr(self):
        return 1. - self.recall()

    def precision(self):
        return self.TP / float(self.TP + self.FP)

    def fpr(self):
        return self.FP / float(self.FP + self.TN)

    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)

    def delta_time(self):
        return np.average(self.time_differences) * (1 / 25.)

    def __str__(self):
        return "TP: {}, FP: {}, FN: {}, TN: {}, \n" \
               "td: {}  \n" \
            .format(self.TP, self.FP, self.FN, self.TN, np.average(self.time_differences) * (1 / 25.))


class Evaluation(object):
    def __init__(self, c, d_max, N_max, sigma_min, w_ext, sl_max):
        """
        :param c: classifier class
        :type c: ShapeletClassifier
        :param d_max: values for d_max that will be tested
        :type d_max: list(float)
        :param N_max: values for N_max that will be tested
        :type N_max: list(int)
        :param sigma_min: values for sigma_min that will be tested
        :type sigma_min: list(float)
        :param w_ext: values for w_ext that will be tested
        :type w_ext: list(int)
        :param sl_max: values for sl_max that will be tested
        :type sl_max: list(int)
        """
        self.d_max = d_max
        self.sigma_min = sigma_min
        self.sl_max = sl_max
        self.c = c
        self.n_max = N_max
        self.w_ext = w_ext
        self.confusion_matrix = defaultdict(lambda: ConfusionMatrix())

    def rate(self, ground_truth, prediction, label, ts_length, shapelet_length):
        """
        Computes TP, FP, TN, FN according to section IV-A.
        """
        matches = defaultdict(lambda: [])
        tp = []
        if len(prediction) > 0:
            for i, e in enumerate(ground_truth):
                match_id = find_nearest(prediction, e)
                match = prediction[match_id]
                distance = abs(match - e)
                if distance < shapelet_length:
                    matches[match].append((distance, e))

            for p in matches.keys():
                asdf = min(matches[p], key=lambda x: x[0])
                tp.append((p, asdf[1]))
        self.confusion_matrix[label].TP += len(tp)
        self.confusion_matrix[label].FP += len(prediction) - len(tp)
        self.confusion_matrix[label].FN += len(ground_truth) - len(tp)
        self.confusion_matrix[label].TN += (ts_length - len(ground_truth)) - self.confusion_matrix[label].FP
        for x, y in tp:
            self.confusion_matrix[label].time_differences.append(abs(x - y))

    def eval(self, data, ground_truth, mode="cv", result_file_name=""):
        """
        Evaluates our classification algorithm.
        :param data: the whole data set
        :type data: np.array, shape = (len(dataset),)
        :param ground_truth: list containing dicts for each training example. The dicts have the event label as key
                                and a list of time indices for each occurrence as value.
        :type ground_truth: np.array, shape = (len(dataset),)
        :param mode: "cv", for 10-fold cross validation
                     int, for int% training set and 100-int% test set
                     None, for 100% training and test set
        :param result_file_name: name of the file in which the results will be stored
        :type result_file_name:
        :return: result from the last training
        """
        d_max = self.d_max[0]
        sigma_min = self.sigma_min[0]
        N_max = self.n_max[0]
        w_ext = self.w_ext[0]
        sl_max = self.sl_max[0]

        # data_train, data_test, target_train, target_test = train_test_split(
        #     data, ground_truth, test_size=1, random_state=42)
        sml = self.c(d_max=d_max, N_max=N_max, sigma_min=sigma_min, w_ext=w_ext, sl_max=sl_max)
        print("d_max={}, sigma_min={}, w_ext={}, n_max={} sl_max={}----------------".format(d_max,sml.sigma_min, sml.w_ext, N_max, sl_max))
        target = np.array([x.keys() for x in ground_truth])
        times = []
        t = time.time()
        result, shapelets = sml.findingshapelets(data, ground_truth)
        times.append(time.time() - t)

        ts_data = []
        dict_training_data = get_training_data()
        data_of_interest = dict_training_data['59d638667bfe0b5f22bd6449: Pitasc - Insert Upright'][2]
        for state_data in data_of_interest:
            ts_data.append(state_data.values())
        data_nd = np.array(ts_data)
        data_nd = data_nd[:, 0:3]
        temp = np.copy(data_nd[:,0:1])
        data_nd[:, 0:1] = data_nd[:,1:2]
        data_nd[:, 1:2] = temp

        #classifier =  result['59d638667bfe0b5f22bd6427: Motek - White Part Unmount']
        dict_classifiers = {}
        for label, (classifier, _) in result.items():
            dict_classifiers[label] = classifier
        cls = dict_classifiers['59d638667bfe0b5f22bd6446: Pitasc-Sub - White Part Mount Tilted']
        mins, ds =cls.predict_all(data_nd)
        plt.plot(ds)
        plt.ylabel('BMD')
        plt.show()

        classifier_threshold = 0.2313831313864876# '59d638667bfe0b5f22bd6427: Motek - White Part Unmount'
        for distance in ds:
            if distance < classifier_threshold:
                print"Successful process!"
                break
        print("Finishing...")

#        self.confusion_matrix = defaultdict(lambda: ConfusionMatrix())
#        for idx, data in enumerate(data_test):

#             x = data[idx]
#              for label, (classifier, _) in result.items():
#                  self.confusion_matrix[label].deltas.append(classifier.delta)
#                  try:
#                      self.confusion_matrix[label].sec_ig.append(classifier.y)
#                  except:
#                      self.confusion_matrix[label].sec_ig.append(classifier.f_c_delta)
#
#                  self.confusion_matrix[label].shapelet_lengths.append(
#                      classifier.shapelet.shape[0])
#                  self.confusion_matrix[label].axis.append(classifier.dim_s)
#                  self.confusion_matrix[label].number_of_shapelets.append(
#                      sum([v.shape[0] for v in sml.shapelets.values()]))
#                  shapelet_length = classifier.shapelet.shape[0]
#                  shapelet_matches = np.array(classifier.predict_all(x)) + shapelet_length // 2
                 # self.rate(ground_truth[i].get(label, []), shapelet_matches, label, x.shape[0],
                 #           shapelet_length)
        # print("training time:{}".format(np.mean(times)))
        # results[d_max, sigma_min, N_max, w_ext, sl_max] = self.confusion_matrix, np.mean(
        #     times)
        # sml.reset()
        return result, shapelets

    def save_results(self, results, filename="result1"):
        with open(filename + ".csv", 'wb') as csvfile:
            w = csv.writer(csvfile, delimiter=';', quotechar='|')
            w.writerow(["d_max",
                        "sigma_min",
                        "N_max",
                        "w_ext",
                        "sl_max",
                        "TP",
                        "FP",
                        "TN",
                        "FN",
                        "delta times",
                        "shapelet_length",
                        "ig",
                        "sec_ig",
                        "deltas",
                        "axis",
                        "number_of_shapelets",
                        "train time"])
            for i, (k, (cm, train_time)) in enumerate(results.items()):
                w.writerow([(x if x is not None else "None") for x in k] +
                           [[(k, m.TP) for k, m in cm.items()]] +
                           [[(k, m.FP) for k, m in cm.items()]] +
                           [[(k, m.TN) for k, m in cm.items()]] +
                           [[(k, m.FN) for k, m in cm.items()]] +
                           [[(k, m.time_differences) for k, m in cm.items()]] +
                           [[(k, m.shapelet_lengths) for k, m in cm.items()]] +
                           [[(k, m.ig) for k, m in cm.items()]] +
                           [[(k, m.sec_ig) for k, m in cm.items()]] +
                           [[(k, m.deltas) for k, m in cm.items()]] +
                           [[(k, m.axis) for k, m in cm.items()]] +
                           [[(k, m.number_of_shapelets) for k, m in cm.items()]] +
                           [train_time])

    def table(self):
        """
        Prints the results as .tex table.
        """
        table_start = "\\begin{tabular}{c|c|c|c|c}\n"
        table_description = "label               & number  & prec    & sen     & $\\Delta$time [s] \\\\\hline\hline\n"
        table_end = "\\end{tabular}"
        table = ""
        avg_prec = []
        avg_sen = []
        avg_dt = []
        counts = []
        order = [
            ("wipe", "wipe_start"),
            ("wipe_end", "wipe_end"),
            ("force_inc", "force_inc"),
            ("force_dec", "force_dec"),
            ("slide", "slide_left_start"),
            ("slide_end", "slide_left_end"),
            ("slide_r", "slide_right_start"),
            ("slide_r_end", "slide_right_end"),
            ("push", "movable_box"),
            ("screw", "fixed_screw")
        ]
        for i, (label, title) in enumerate(order):
            matrix = self.confusion_matrix[label]
            table += "&".join(["{:^9.3}".format(x) if isinstance(x, float) else "{:^9}".format(x) for x in [
                "{:{}}".format(title, 20 - title.count("_")),
                int(matrix.TP + matrix.FN), matrix.precision(), matrix.recall(),
                matrix.delta_time()]]) + "\\\\\hline\n"
            avg_prec.append(matrix.precision())
            avg_sen.append(matrix.recall())
            avg_dt.append(matrix.delta_time())
            counts.append(int(matrix.TP + matrix.FN))
        table = table[:-1]
        table += "\hline\n"
        table += "&".join(["{:^9.3}".format(x) if isinstance(x, float) else str(x) for x in
                           ["{:20}".format("average"), "{:^9}".format("-"), np.mean(avg_prec), np.mean(avg_sen),
                            np.mean(avg_dt)]]) + "\\\\\hline\n"
        avg_prec = np.average(avg_prec, weights=counts)
        avg_sen = np.average(avg_sen, weights=counts)
        avg_dt = np.average(avg_dt, weights=counts)
        table += "&".join(["{:^9.3}".format(x) if isinstance(x, float) else str(x) for x in
                           ["{:20}".format("weighted avg"), "{:^9}".format("-"), avg_prec, avg_sen,
                            avg_dt]]) + "\\\\\hline\n"
        table = table[:-7]
        table += "\n"
        table = table.replace("_", "\\_")
        return table_start + table_description + table + table_end


def plot_shapelet(ax, shapelet, axis, time=None, linethickness=LT, colors=colors, label=label):
    if linethickness is None:
        linethickness = LT
    lines = dict()
    for i, a in enumerate(axis):
        if time is None:
            lines[label[a]] = ax.plot(np.arange(0, shapelet.shape[0]), shapelet[:, i], color=colors[a], label=label[a],
                                      linewidth=linethickness)[0]
        else:
            lines[label[a]] = ax.plot(time, shapelet[:, i], colors[a], label=label[a], linewidth=linethickness)[0]
    return lines


def plot_all_shapelets(result):
    f, rows = plt.subplots(5, 2, sharex=True)
    axs = list(rows[:, 0]) + list(rows[:, 1])
    lines = dict()
    classifiers = dict()
    for i, (label, (classifier, _)) in enumerate(result.items()):
        classifiers[label] = classifier
    order = [("wipe", "wipe_start"), ("force_inc", "force_inc"), ("slide", "slide_left_start"),
             ("slide_r", "slide_right_start"),
             ("push", "movable_box"), ("wipe_end", "wipe_end"), ("force_dec", "force_dec"),
             ("slide_end", "slide_left_end"),
             ("slide_r_end", "slide_right_end"), ("screw", "fixed_screw")]
    j = 0
    for i, (label, title) in enumerate(order):
        if classifiers.has_key(label):
            classifier = classifiers[label]
            axs[j].set_title(title)
            lines.update(plot_shapelet(axs[j], classifier.shapelet, classifier.dim_s))
            axs[j].set_xlim(0, 50)
            plt.setp(axs[j].get_yticklabels(), visible=False)
            j += 1
    classifiers = sorted(lines.items(), key=lambda x: x[0])
    plt.figlegend([x[1] for x in classifiers], [x[0] for x in classifiers], loc='lower center', ncol=5, labelspacing=0.)
    plt.subplots_adjust(left=.05, bottom=.15, right=.95, top=.93, wspace=.12, hspace=.51)
    plt.show()

    ################################## My implementaion##########################

def get_training_data():  # getting the dictionary from the pickeled file
    with open("../dataset/data.dat", "rb") as file:
        dict_training_data = pickle.load(file)
    # print "states: ",dict_training_data.keys()
    return dict_training_data

def separate_state(dict_training_data):
    '''
    input: dict_training_data
    output: list of dictionaries; every state appearance and its corresponding measurements
    '''

    list_data_state_dict = []
    data_list = []
    data_state_dict = {}
    list_dict = []
    dict = {}
    for state in dict_training_data.keys():
        for i in range(len(dict_training_data[state])):
            key = state
            data_list = dict_training_data[state][i]
            data_state_dict = {key: data_list}
            # list_data_state_dict = list(data_state_dict)
            list_dict.append(data_state_dict)
    return list_dict

def separating_list_dict(list_dict):

    '''
    input: list of dictionaries; every state appearance and its corresponding measurements
    outputs: list of data and the corresponding list of targets
    '''

    data = []
    targets = []

    for idx, state_data in enumerate(list_dict):
        targets.append(state_data.keys())
        data.append(state_data.values())
    return data, targets

def list_to_ndarray(data, states):

    '''
    input: list of data and the corresponding list of targets
    output:
    '''
    list_states_dict = []
    list_nd_array = []
    list_nd_time = []
    for idx, dat in enumerate(data):
        df = pd.DataFrame(data[idx][0])
        nd_array = df.values
        nd_time = nd_array[:, 0:1]
        nd_array = nd_array[:, 1:4]
        list_nd_array.append(nd_array)
        list_nd_time.append(nd_time)

    # print "list_nd_array", list_nd_array
    arr_list_nd_array = np.array(list_nd_array)

    # print "arr_list_nd_array", arr_list_nd_array

    for idx, state in enumerate(states):
        state = tuple(state)
        states_dict = {state[0]: [idx]}
        list_states_dict.append(states_dict)
    nd_states_dict = np.array(list_states_dict)

    # print "nd_time: ", list_nd_time
    # print"nd_targets: ", nd_targets
    # print "nd_states_dict: ", nd_states_dict
    # print "length of the data list of arries: ", len(list_nd_array)
    # print "shape of the states array", nd_states_dict.shape
    return arr_list_nd_array, nd_states_dict, list_nd_time

def removing_faulty_readings(list_nd_array, nd_states_dict, list_nd_time):

    unfaulty_list_nd_array = list()
    idx_fault = list()
    # print "len(list_nd_array): ", len(list_nd_array)
    for idx, data in enumerate(list_nd_array):
        if not data.size:
            idx_fault.append(idx)
        else:
            unfaulty_list_nd_array.append(data)
    # print "list_unfaulty_nd_array: ", np.array(unfaulty_list_nd_array)
    # print "list of faulty indexes", idx_fault
    # print "len(list_unfaulty_nd_array): ", len(unfaulty_list_nd_array)

    unfaulty_nd_states_dict = np.delete(nd_states_dict, idx_fault)
    unfaulty_list_nd_time = np.delete(list_nd_time, idx_fault)
    # print "len(unfaulty_nd_states_dict)", len(unfaulty_nd_states_dict)
    # print "unfaulty_nd_states_dict: ",np.array(unfaulty_nd_states_dict)

    return np.array(unfaulty_list_nd_array), np.array(unfaulty_nd_states_dict), np.array(
        unfaulty_list_nd_time)  # States of interests lie in this subset"White Part Mount Tilted"

def extract_state_interest(unfaulty_list_nd_array, unfaulty_nd_states_dict):
    states_interest = ['59d638667bfe0b5f22bd6443: Motek - White Part Mount Tilted','59d638667bfe0b5f22bd6446: Pitasc-Sub - White Part Mount Tilted',
                       '59d638667bfe0b5f22bd6449: Pitasc - Insert Upright', '59d638667bfe0b5f22bd6420: Motek - Erebus Unmount']
    #states_interest = ['59d638667bfe0b5f22bd6449: Pitasc - Insert Upright', '59d638667bfe0b5f22bd6420: Motek - Erebus Unmount']
    data = []
    states = []
    for idx, state in enumerate(unfaulty_nd_states_dict):
        for state_interest in states_interest:
            if state.keys()[0] == state_interest:
                state_dict = unfaulty_nd_states_dict[idx]
                states.append(state_dict)
                data.append(unfaulty_list_nd_array[idx])

    return np.array(data), np.array(states)

def reform_ground_truth(ground_truth_shapelet):

    labels = list()
    empty_idx = list()
    for idx, data in enumerate(ground_truth_shapelet):
        if not bool(data.keys()):
            # print"empty dict"
            empty_idx.append(idx)
        else:
            labels.append(data.keys()[0])
    for i in empty_idx:
        labels.insert(i, 'Null')
    # print"labels: ", labels
    labels = np.array(labels)
    list_dict = []
    for idx, dictionary in enumerate(ground_truth_shapelet):
        if idx not in empty_idx:
            simplified_dict = {labels[idx]: dictionary[labels[idx]]}
            list_dict.append(simplified_dict)
    for i in empty_idx:
        list_dict.insert(i, '{}')

    arr_list_dict = np.array(list_dict)
    # print "simplified_dict: ", arr_list_dict
    return arr_list_dict

def printing_shapelet_data(data, ground_truth):

    print "ground_truth: ", ground_truth
    # print "data: ", data
    print "shape.data)", data.shape
    # print "shape.ground_truth", ground_truth.shape

def printing_denso_data(unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time):

    # print "nd_states_dict: ", nd_states_dict
    # print "list_nd_array: ", list_nd_array
    print "unfaulty_list_nd_array.shape", len(unfaulty_list_nd_array[1])
    print "unfaulty_nd_states_dict.shape ", len(unfaulty_nd_states_dict[0])
    print "unfaulty_list_nd_time.shape ", len(unfaulty_list_nd_time[1])

def save_data(unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time):
    pickle.dump(unfaulty_list_nd_array, open("../denso_data/unfaulty_list_nd_array.dat", "wb"))
    pickle.dump(unfaulty_nd_states_dict, open("../denso_data/unfaulty_nd_states_dict.dat", "wb"))
    pickle.dump(unfaulty_list_nd_time, open("../denso_data/unfaulty_list_nd_time.dat", "wb"))

def saving_generated_shapelets(shapelets):

    pickle.dump(dict(shapelets), open("../shapelets_data/shapelets.dat", "wb"))


def main(mode):


    dict_training_data = get_training_data()
    list_dict = separate_state(dict_training_data)
    data_denso, states_denso = separating_list_dict(list_dict)
    list_nd_array, nd_states_dict, list_nd_time = list_to_ndarray(data_denso, states_denso)
    unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time= removing_faulty_readings(list_nd_array, nd_states_dict, list_nd_time)
    data_denso, states_denso = extract_state_interest(unfaulty_list_nd_array, unfaulty_nd_states_dict)
    save_data(unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time)
    printing_denso_data(unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time)


    #data, ground_truth = import_db()
    evaluation = Evaluation(c=ShapeletFinder, d_max=[.5], N_max=[3], w_ext=[25], sigma_min=[None], sl_max=[50])
    if mode == "cv":
        result = evaluation.eval(data_denso, states_denso, mode="cv")  # 10-fold cross validation
        print "Finishing.."
    plot_all_shapelets(result)


if __name__ == '__main__':
    # seeded to make the experiments repeatable
    np.random.seed(1)
    parse = argparse.ArgumentParser()
    parse.add_argument("--mode", type=str, default="cv")
    args, unknown = parse.parse_known_args()
    main(args.mode)
