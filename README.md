# Project_CV
If you are running this script fot the first time, you have to set the variable "LOAD" to False, doing so the script will create the bag of word dataset and save it in your directory. In the next runs you can set "LOAD" to true and the script will load your precedent bag of word dataset. 
The other flags are useful if you, for example, want to execute only Knn and not SVM or the other way around, or if you want to choose between test and cross validation.

#Scripts 
"main.py" launches the script.
"Dataset.py" contains class and methods to create your bag of word dataset.
"Histogram.py" contains methods to manipulate SIFT histograms.
"EMD.py" implements the earth mover's distance
"classifiers.py" contains custom classifiers classes
"test.py" contains the cross validation routine
"Evaluate.py" contains some functions to get the accuracy of predictions
