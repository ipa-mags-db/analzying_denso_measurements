The work on this repository depends on the algorithm created by Stelter, Bartels and Beetz from Bremen university. They have come up with an algorithm (find_shapelets.py) that is able to classify and detect contact events in a multi-dimensional time series. 

Their research focused on detecting and classifying Contact events in Force measurements using a new primitive for Data Mining called "Shapelet" created by Lexiang Ye and Eamonn Keogh at University of Califorina. 

My work is to determine if e.g. an assembly process finished "successfully" or not. I used a simplified version of their find_shapelets algorithm in order to achieve my task. The main idea depends of finding the BMD(best match distance) between the generated shapelet and a random multi-dimensional time series. If any value this distance is less than a specific predetermined classifier threshold, this means that this shapelet belongs to this random multi-dimensional time series, otherwise, it does not belong to it. Additionall, this means that we had a prblem during our assembly process. 

There are three main files in this repo:
- First, find_shapelets_simplified.py: This is the main algorithm, it's trimmed to suit our needs, which outputs the generated shapelets and a classifier, those will be used later during the evaluation process. 
- Second, dataset_preparation.py: Our dataset is a dictionary(dict_training_data) that contains the states of the Denso robot during a specific assembly process and its corresponding force measurments. This is file, there's a Dataset class that contain some functions that work on the (dict_training_data) in order to put our dataset into an acceptable form to be fed to (find_shapelets_simplified.py)
- Third, shapelets_evaluation.py: During this file we find the BMD between a random multi-dimensional time series and a shapelet and compare the minimum value of BMD to a classifier threshold to determine if the procss is successful or not. 

Remarks and thoughts:
- Until this point the algorithm has been used for of offline classification. In order to do online classification. We will need only to run the algorithm once, after that we can save the classifier data in a pickled file which can be easily imported for later uses. 
- In the find_shapelet algorithm, there are five user-specified paramaters. These might need to be changed depending on special requirements. e.g. the maximum shapelet length. I have not changed the values of these parameters, yet the algorithm works as expected.

How to run the algorithm? 

Using Python 2.7, we run (find_shapelets_simplified.py), this is going to generate the classifier, saving it into a pickled file and finally ploting the shapelets.

How to evaluate the algorithm? 

In (shapelets_evaluation.py) file, we can change the random testing time series, intended classifier and it's correspondign classifier threshold. After setting those three variables, we run the file and the evaluation function will check the BMD, compare it with the pre-set classification threshold. If the min(BMD) <  Classification threshold, this means that the shapelet has been found within this random multi-dimensional time-series and the process is Successful. And vice versa.