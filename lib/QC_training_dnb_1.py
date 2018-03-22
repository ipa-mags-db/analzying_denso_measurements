#!/usr/bin/env python
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String
import pandas as pd
from std_srvs.srv import Empty
import matplotlib.pyplot as plt
import os

# Global variables to simplify running numbers of states/skills
dict_training_data = {}
current_states = []
start_times = []

def handle_show(req):
    global dict_training_data
    for state in dict_training_data:
        print state
        #print dict_training_data[state]
        dict_training_data[state]['df'].plot()
        plt.show()
    return []

def handle_save(req):
    global dict_training_data
    for state in dict_training_data:
        dict_training_data[state]['df'].to_pickle("{}.{}.dat".format(state, dict_training_data[state]['cnt']))
        print "Pickled data: {}".format(state)
    return []

def handle_load(req):
    global dict_training_data, current_states, start_times

    # clean up running acqusition
    dict_training_data = {}
    current_states = []
    start_times = []

    for file in os.listdir("."):
        if file.endswith(".dat"):
            state_arr = file.split('.')
            state = '.'.join(state_arr[0:-2])
            cnt = state_arr[-2]

            print "Found file: {}.dat".format(state)
            print "Found cnt: {}".format(cnt)
            df = pd.read_pickle(os.path.join(".", file))
            dict_training_data[state] =\
                {'df': df,
                 'cnt' : cnt}


    return []

# This function is called whenever a wrench data set arrives (now does nothing)
def callback_wrench(data):
    global dict_training_data, current_states, start_times
    for state in current_states:
        idx = current_states.index(state)
        start_time = start_times[idx]
        cnt = dict_training_data[state]['cnt'] #Not understood


        meas = [{'x_{}'.format(cnt): data.wrench.force.x,
                 'y_{}'.format(cnt): data.wrench.force.y,
                 'z_{}'.format(cnt): data.wrench.force.z,
                 'time': rospy.Time.now().to_sec() - start_time}]

        df = pd.DataFrame(meas)
        df = df.set_index('time')

        dict_training_data[state]['df'] = dict_training_data[state]['df'].append(df)
    #print rospy.Time.now()
    #print pd.Series([[data.wrench.force.x, data.wrench.force.y, data.wrench.force.z]], index=[rospy.Time.now().to_sec()])

    #print state, ":\n", dict_training_data[state]
    #rospy.loginfo("{}\n{}".format(current_states, start_times))


# This function is called whenever a information about the active state arrives
def callback_log(data):
    global current_states, start_times, dict_training_data

    # Checks if a new state is entered
    if data.data.startswith("Entering state"):
        statename = data.data[15:] # Cuts the first 16 characters of the message

        # Checks if statename as already occured, if not append to current_states
        if not statename in current_states:
            print "Collecting data during state '{}' [{}]".format(statename,
                dict_training_data[statename]['cnt'] if statename in dict_training_data else 1)
            current_states.append(statename)
            start_times.append(rospy.Time.now().to_sec()) # dnb message does not have a timestamp
            if statename in dict_training_data:
                dict_training_data[statename]['cnt'] += 1
            else:
                # add state
                dict_training_data[statename] =\
                    {'df':pd.DataFrame(), 'cnt':1}

    else: #Leaving state
        statename = data.data[14:]
        if statename in current_states:
            idx = current_states.index(statename)
            del start_times[idx]
            current_states.remove(statename)

        #print statename, dict_training_data[statename], len(dict_training_data)

def listener():

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
    rospy.spin()

if __name__ == '__main__':
    listener()
