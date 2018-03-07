#!/usr/bin/env python
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String
import pandas as pd
from std_srvs.srv import Empty
import matplotlib.pyplot as plt
import os
import threading
import pickle

# Global variables to simplify running numbers of states/skills
dict_training_data = {}
current_states = []
start_times = []
lock = threading.Lock()

def handle_show(req):
    global dict_training_data
    for state in dict_training_data:
        print state
        #print dict_training_data[state]
        try:
            ax = pd.DataFrame(dict_training_data[state][0]).plot()
            if len(dict_training_data[state]) > 1:
                for i in range(1, len(dict_training_data[state])):
                    pd.DataFrame(dict_training_data[state][i]).plot(ax=ax)
            plt.show()
        except:
            pass
    return []

def handle_save(req):
    global dict_training_data
    pickle.dump(dict_training_data, open( "data2.dat", "wb" ))
    return []

def handle_load(req):
    global dict_training_data, current_states, start_times

    # clean up running acqusition
    dict_training_data = pickle.load(open( "data2.dat", "rb" ))
    current_states = []
    start_times = []

    return []

# This function is called whenever a wrench data set arrives (now does nothing)
def callback_wrench(data):
    global dict_training_data, current_states, start_times, lock
    tme =  rospy.Time.now().to_sec() #TODO: Use timestamp
    with lock:

        for state in current_states:
        
                idx = current_states.index(state)
                start_time = start_times[idx]

                #print "---->", tme, start_time, tme - start_time

                meas = {'x': data.wrench.force.x,
                        'y': data.wrench.force.y,
                        'z': data.wrench.force.z,
                        'time': tme - start_time}
                dict_training_data[state][-1].append(meas) #Why -1 ?
# This function is called whenever a information about the active state arrives
def callback_log(data):
    global current_states, start_times, dict_training_data, lock
    tme = rospy.Time.now().to_sec()
    with lock:
        # Checks if a new state is entered
        if data.data.startswith("Entering state"):
            statename = data.data[15:] # Cuts the first 16 characters of the message

            #print "<--enter", statename, tme
            # Checks if statename as already occured, if not append to current_states
            if not statename in current_states:

                current_states.append(statename)
                start_times.append(tme) # dnb message does not have a timestamp
                if statename in dict_training_data:
                    dict_training_data[statename].append([])
                else:
                    # add state
                    dict_training_data[statename] = [[]] #empty list of list
            else:
                print "State appeared twice???????? PROBLEM????"

            print len(current_states), current_states

        else: #Leaving state
            statename = data.data[14:]
            if statename in current_states:
                idx = current_states.index(statename)
                del start_times[idx]
                current_states.remove(statename)

                print "exit-->", statename, tme

def listener():
    global current_states

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/wrench", WrenchStamped, callback_wrench)
    rospy.Subscriber("/dnb_executor/log", String, callback_log)

    rospy.Service('show', Empty, handle_show)
    rospy.Service('save', Empty, handle_save)
    rospy.Service('load', Empty, handle_load)

    # spin() simply keeps python from exiting until this node is stopped
    try:
        rospy.spin()
    finally:
        print "dict_training_data: "
        print "Should save data in the following states (still running)\n", current_states
        print "...but I don't!"

if __name__ == '__main__':
    listener()
