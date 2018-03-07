import csv
import numpy as np


def import_db():
    """
    :return: the training examples (data) and list of event labels (target) from db.csv
    :rtype: np.array, shape = (len(data), len(target))
    """
    data = []
    target = []
    with open('../shapelet_dataset/simplified_more_db.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            data.append(np.array([eval(row["x"]), eval(row["y"]), eval(row["z"])]).T)
            d = dict()
            k = eval(row["label"])
            v = eval(row["time_stamps"])
            for i in range(len(k)):
                d[k[i]] = v[i]
            target.append(d)


    data_arr = np.array(data) 

    #print"shape of data_arr: ", len(data_arr)
    #print"type of data_arr: ", type(data_arr)
    #print" first row: ", data_arr[48]
    #print "data: ", data_arr
    #np.savetxt("foo.csv", np.array(data_arr[1]), delimiter=",")
    #print "succseed..!"


    return np.array(data), np.array(target)
    
import_db()