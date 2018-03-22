import numpy as np
import pandas as pd
import pickle


class Dataset:
    'Class for handling the dataset which will be later provided to the finding shaplets algorithm'


    def __init__(self):

        pass


    def get_training_data(self):
        """
        Getting the dictionary dataset from the pickeled file
        return: dictonary contains the Denso states occurances and its corresponding force measurements
        rtype: dict
        """
        with open("../dataset/data.dat", "rb") as file:
            dict_training_data = pickle.load(file)
        return dict_training_data


    def extract_states(self, dict_training_data):
        """
        Extracting the states from the dictionary dataset
        param dict_training_data: dict_training_data
        type dict_training_data: dict
        return: list of dictionaries, every state appearance and its corresponding force measurements
        rtype: list(dict)
        """
        data_list = []
        data_state_dict = {}
        list_dict = []
        for state in dict_training_data.keys():
            for i in range(len(dict_training_data[state])):
                key = state
                data_list = dict_training_data[state][i]
                data_state_dict = {key: data_list}
                list_dict.append(data_state_dict)
        return list_dict


    def separating_list_dict(self, list_dict):

        """
        Separating the dataset into two different lists of contrainers, one for the data and one for the states
        param list_dict: list of dictionaries; every state appearance and its corresponding measurements
        type list(dict)
        return: list of data and the corresponding list of targets
        rtype: list
        """
        data = []
        states = []

        for idx, state_data in enumerate(list_dict):
            states.append(state_data.keys())
            data.append(state_data.values())
        return data, states

    def list_to_ndarray(self, data,states):

        """
        Converting the list of data and states into numpy array and extracting the time stamp from the data
        param data: list of dataset
        param states: The corresponding list of states
        return: ndarrays for the dataset, the states and the time stamps
        rtype: ndarray
        """
        list_states_dict = []
        list_nd_array = []
        list_nd_time = []
        for idx, dat in enumerate(data):
            df = pd.DataFrame(data[idx][0])
            nd_array=df.values
            nd_time = nd_array[:,0:1] # Slicing the time stamp
            nd_array = nd_array[:,1:4]
            list_nd_array.append(nd_array)
            list_nd_time.append(nd_time)
        arr_list_nd_array = np.array(list_nd_array)
        arr_list_nd_time = np.array(list_nd_time)

        for idx, state in enumerate(states):
            state = tuple(state)
            states_dict = {state[0]: [idx]} # Putting the states into a dictionary form
            list_states_dict.append(states_dict)
        nd_dict_states = np.array(list_states_dict)
        return arr_list_nd_array, nd_dict_states, arr_list_nd_time


    def removing_faulty_readings(self, arr_list_nd_array, nd_dict_states, arr_list_nd_time):
        """
        Removing the faulty data and the corresponding states from the dataset
        param arr_list_nd_array: ndarray of lists of arrays of the data
        param nd_dict_states: ndarray of dictionaries of states
        param arr_list_nd_time: ndarray of time stamps
        return: ndarrays of data, states and time stamps
        rtype: ndarray
        """
        unfaulty_list_nd_array = list()
        idx_fault = list()
        for idx, data in enumerate( arr_list_nd_array):
            if not data.size:
                idx_fault.append(idx)
            else:
                unfaulty_list_nd_array.append(data)
        unfaulty_nd_dict_states = np.delete(nd_dict_states, idx_fault)
        unfaulty_list_nd_time = np.delete(arr_list_nd_time, idx_fault)
        return np.array(unfaulty_list_nd_array), np.array(unfaulty_nd_dict_states) , np.array(unfaulty_list_nd_time)


    def printing_denso_data(self, unfaulty_list_nd_array, unfaulty_nd_dict_states, unfaulty_list_nd_time):
        """
        Printing the data, states and the time stamps
        param arr_list_nd_array: ndarray of lists of arrays of the data
        param nd_dict_states: ndarray of dictionaries of states
        param arr_list_nd_time: ndarray of time stamps
        """
        print "unfaulty_list_nd_array.shape", len(unfaulty_list_nd_array[1])
        print "unfaulty_nd_dict_states.shape ", len(unfaulty_nd_dict_states[0])
        print "unfaulty_list_nd_time.shape ", len(unfaulty_list_nd_time[1])


    def save_data(self, unfaulty_list_nd_array, unfaulty_nd_dict_states, unfaulty_list_nd_time):
        """
        Saving the data into a pickled files
        param unfaulty_list_nd_array: ndarray of lists of arrays of the data
        param nd_dict_states: ndarray of dictionaries of states
        param arr_list_nd_time: ndarray of time stamps

        """
        pickle.dump(unfaulty_list_nd_array, open( "../denso_data/unfaulty_list_nd_array.dat", "wb" ))
        pickle.dump(unfaulty_nd_dict_states, open( "../denso_data/unfaulty_nd_states_dict.dat", "wb" ))
        pickle.dump(unfaulty_list_nd_time, open( "../denso_data/unfaulty_list_nd_time.dat", "wb" ))


    def extract_states_interest(self, unfaulty_list_nd_array, unfaulty_nd_dict_states):
        """
        Extracting the states of interest from the non-faulty data
        param unfaulty_list_nd_array: ndarray of dataset
        param unfaulty_nd_dict_states: ndarray of states
        return: ndarrays of the extracted data and the corresponding states
        rtype: ndarray:
        """
        states_interest = ['59d638667bfe0b5f22bd6443: Motek - White Part Mount Tilted',
                           '59d638667bfe0b5f22bd6446: Pitasc-Sub - White Part Mount Tilted',
                           '59d638667bfe0b5f22bd6449: Pitasc - Insert Upright',
                           '59d638667bfe0b5f22bd6420: Motek - Erebus Unmount']
        data = []
        states = []
        for idx, state in enumerate(unfaulty_nd_dict_states):
            for state_interest in states_interest:
                if state.keys()[0] == state_interest:
                    state_dict = unfaulty_nd_dict_states[idx]
                    states.append(state_dict)
                    data.append(unfaulty_list_nd_array[idx])
        return np.array(data), np.array(states)


def bringup():
    dataset = Dataset()

    dict_training_data = dataset.get_training_data()
    list_dict = dataset.extract_states(dict_training_data)
    data_denso, states_denso = dataset.separating_list_dict(list_dict)
    list_nd_array, nd_states_dict, list_nd_time = dataset.list_to_ndarray(data_denso, states_denso)
    unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time= dataset.removing_faulty_readings(list_nd_array, nd_states_dict, list_nd_time)
    data_denso, states_denso = dataset.extract_states_interest(unfaulty_list_nd_array, unfaulty_nd_states_dict)
    #dataset.save_data(unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time)
    #dataset.printing_denso_data(unfaulty_list_nd_array, unfaulty_nd_states_dict, unfaulty_list_nd_time)
    return data_denso, states_denso


if __name__ =='__main__' :

    data_denso, states_denso = bringup()


##################################################

""""
The next two functions were used for the sake of debugging

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
"""