#!/usr/bin/env python
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String



# This function is called whenever a wrench data set arrives (now does nothing)
def callback_wrench(data):
    #rospy.loginfo("I heard %s", data.wrench)
    pass

# Global variables to simplify running numbers of states/skills
dict_status_num = {}
cnt = 0



# This function is called whenever an information about the active state arrives
def callback_log(data):
    global dict_status_num, cnt
    #rospy.logwarn("I heard %s", data.data)

    # Checks if a new state is entered
    if data.data.startswith("Entering state"):
        statename = data.data[15:] # Cuts the first 16 characters of the message

        # Checks if statename as already occured, if not gives a unique id and stores it
        if not statename in dict_status_num:
            dict_status_num[statename] = cnt
            cnt += 1

        # Outputs if some state was entered
        rospy.logwarn("%s, state %s", dict_status_num[statename], statename)


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/wrench", WrenchStamped, callback_wrench)
    rospy.Subscriber("/dnb_executor/log", String, callback_log)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
