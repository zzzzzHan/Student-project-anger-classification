The neural network is to classify the Genuine/Posed of the anger. 
There are few tips for running the code:
1. The suggested operating system is MAC
2. Do not move the file 'anger.xlxs' outside the folder anger or the data will not be loaded sucessfully 

The code is divided into several parts:
1. Line 29- 61     load and preprocess the data; split the data for training and testing
3. Line 64-108     Some helpful function construction 
4. Line 109- 320.  Construction, parameter tuning, and evaluation for baseline network
5. Line 346- 506 FCNN Model 1 (refer to paper) construction and evaluation 
6. Line 512- 649 FCNN Model 4 (refer to paper) construction and evaluation 
7. Line 650- 762 FCNN Model 3 (refer to paper) construction and evaluation
8. Line 766- 897 Compare FCNN Model 3 with different numbers of clusters