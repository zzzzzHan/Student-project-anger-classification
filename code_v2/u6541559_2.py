#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''This file is the script of COMP4660 Assignment 2 for u6541559 including 
   the construction, parameter and evaluation of the a feedforward neural network.
   The task is to classify the realness of anger 
'''
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as p
from sklearn.metrics import confusion_matrix
pd.options.mode.chained_assignment = None  # default='warn'
import skfuzzy as fuzz
from time import perf_counter


# In[2]:


'''------------------Data Pre-processing--------------------
'''
raw_data = pd.read_excel('anger/Anger.xlsx')
#The task is to predict the realness of anger 
#so drop Video info as the  ground truth label is based on the Video source
raw_data =raw_data.drop(labels=['Video','Unnamed: 0'], axis=1)


raw_data.at[raw_data['Label'] == 'Posed', ['Label']] = 0
raw_data.at[raw_data['Label'] == 'Genuine', ['Label']] = 1
raw_data = raw_data.apply(pd.to_numeric)
raw_data_scaled = raw_data.copy()

columns = raw_data.columns[:-1]
#apply normalization techniques 
#avvoid the feature with large rangehave greater effect on the network.
# Compared two kinds of normalisation, the Z-score normalisation works best 
for i in columns:
    #raw_data_scaled[i] = (raw_data_scaled[i] - raw_data_scaled[i].min()) / (raw_data_scaled[i].max() - raw_data_scaled[i].min())    
    raw_data_scaled[i] = (raw_data_scaled[i] - raw_data_scaled[i].mean()) /raw_data_scaled[i].std()
# split the data inoto trainning data and testing data 
train,test = train_test_split(raw_data_scaled,test_size=0.2, random_state=1)


# In[3]:


train_x=train.drop(labels=['Label'],axis=1)
test_x = test.drop(labels=['Label'],axis=1)
train_y = train['Label']
test_y = test['Label']
X, Y = torch.tensor(np.array(train_x),dtype=torch.float), torch.tensor(np.array(train_y),dtype=torch.long)
X_test, Y_test = torch.tensor(np.array(test_x),dtype=torch.float), torch.tensor(np.array(test_y),dtype=torch.long)


# In[4]:


#theta = 0.5
def threshold_classifier(probability,theta):
    """ a self defined classifer that allows adjusting the value of threshold.
        parameter probability: membership(probability) of the entity belongng to the specific class
        parameter theta: threshold of activation function 
        return: predicted label 
        Normally, the threshold is 0.5.
        It is equivalent to torch.max(torch.softmax(Y_pred,dim=1)) when threshold=0.5

    """
    label_list =[]
    for row in range(0,len(probability)):
        if probability[row][0]>=theta:
            label_list.append(0)
        else:
            label_list.append(1)
    return torch.tensor(np.array(label_list),dtype=torch.long)

def correct_counter(predicted,actual):
    """count the number of corrct prediction
    """
    num =0
    for i in range(0,len(predicted)):
        if predicted[i]==actual[i]:
            num+=1
    return num
def evaluation(exp,act):
    """ The function is set to obtain various measures to evaluate the 
        outcome of trainnng data and testing data. 
    """
    df_confusion = pd.crosstab(act, exp, rownames=['Actual'], colnames=['Predicted'], margins=True)
    FN = df_confusion[1][0]
    FP = df_confusion[0][1]
    TP = df_confusion[1][1]
    TN = df_confusion[0][0]
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F = 2*(precision*recall)/(precision+recall)
    return TP+TN, FP,FN, accuracy,precision,recall,F    


# In[5]:


'''----------------------baseline NN-----------------------------------
    Parameter tuning and trainning 
'''
#parameter of NN
input_neurons = train_x.shape[1]
hidden_neurons = 20
output_neurons = 2
learning_rate = 0.01
num_epoch = 2000

# A standard backpropergation Neural network with two hidden layer
class Network(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Network, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
        

    def forward(self, x):

        # get hidden layer input
        out = self.hidden1(x)
        out = torch.sigmoid(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        y_pred = self.out(out)
        
        return y_pred
net = Network(input_neurons, hidden_neurons, output_neurons)
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser with regularisation term 
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.001)
# Try other optimiser
#optimiser = torch.optim.Adadelta(net.parameters(), lr=0.01)
#scheduler = lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.5)

# To store the loss of trainning and testing set when choose differnt hidden layer
diff_neurons_trainning =[]
diff_neurons_testing = []


def loss_differet_n(hidden_n):
    """ A help function to obtain the loss of the network with different hidden neurons
    """
    net = Network(input_neurons, hidden_n, output_neurons)
    for epoch in range(0,num_epoch+1):
        Y_pred = net(X)
        loss_1 = loss_func(Y_pred, Y)
        test_pred = net(X_test)
        loss_2 = loss_func(test_pred, Y_test)
        if epoch == 1000:
            predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
            correct = correct_counter(predicted.data,Y.data)        
            loss_a= loss_1.data.item()
            loss_b = loss_2.data.item()
        optimiser.zero_grad()
        loss_1.backward()
        optimiser.step()
 #   scheduler.step()
    return loss_a,loss_b

# store the loss of the network with different hidden neurons into list for visulisation 
for nodes in range(1,25):
    loss_a, loss_b= loss_differet_n(nodes)
    diff_neurons_trainning.append(loss_a)
    diff_neurons_testing.append(loss_b)

# For visualization loss respect to number of hidden neurons.  
begin = perf_counter()

x_nodes = list(range(1,25,1))
plt.plot(x_nodes, diff_neurons_trainning, color='green', label='training loss')
plt.plot(x_nodes, diff_neurons_testing, color='blue', label='testing loss')
plt.legend() 
plt.xlabel('number of hidden neurons')
plt.ylabel('loss')
plt.title('The loss changes when epoches=600 respect to number of hidden neurons ')
plt.show()


# After deciding to choose the suitable hidden neurons
# visulise the loss of trainning and testing respect to number of iterations
trainning_loss = []   
testing_loss =[]
accuracy_train_base=[]
accuracy_test_base=[]

for epoch in range(num_epoch):
    Y_pred = net(X)
    loss_1 = loss_func(Y_pred, Y)
    trainning_loss.append(loss_1.item())
    test_pred = net(X_test)
    loss_2 = loss_func(test_pred, Y_test)
    testing_loss.append(loss_2)

    # print progress
    if epoch % 50 == 0:
        predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
        correct = correct_counter(predicted.data,Y.data)        
        print('epoch: {}, loss: {}, accuracy:{}'.format(epoch, loss_1.data.item(),100 * correct/320),'%')
    if epoch % 40 == 0:
        predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
        correct = correct_counter(predicted.data,Y.data)
        accuracy_train_base.append(correct/len(predicted))
        Y_pred = net(X_test)
        predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
        correct = correct_counter(predicted.data,Y_test.data)
        accuracy_test_base.append(correct/len(predicted))

    optimiser.zero_grad()
    loss_1.backward()
    optimiser.step()
   # scheduler.step()

time =  perf_counter()-begin
print('time',time) 


# In[7]:


'''-------------Evaluation for baseline model------------------ 
'''
base_loss = trainning_loss
plt.figure()
plt.plot(trainning_loss,label='training loss')
plt.plot(testing_loss,label='testing loss')
plt.legend() 
plt.xlabel('number of iterations')
plt.ylabel('loss')
plt.title('The loss changes respect to number of epoches ')
plt.show()


# In[8]:


list_theta=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
trainning_outcome = pd.DataFrame(index=list_theta,columns=['correct', 'FP', 'FN', 'accuracy','precision','recall','F-score'])
testing_outcome = pd.DataFrame(index=list_theta,columns=['correct', 'FP', 'FN', 'accuracy','precision','recall','F-score'])

def evaluation_for_NN (list_theta,x,y,outcome_list,net):
    for theta in list_theta:
        predicted_1 = threshold_classifier(torch.softmax(net(x),dim=1),theta) 
        y_actual_1 =y.data
        y_predict_1 = predicted_1.data
        correct,FP,FN, accuracy,precision,recall,F = evaluation(y_predict_1,y_actual_1)
        outcome_list.loc[theta]['correct']=correct
        outcome_list.loc[theta]['FP']=FP
        outcome_list.loc[theta]['FN']=FN
        outcome_list.loc[theta]['accuracy']=accuracy
        outcome_list.loc[theta]['precision']=precision
        outcome_list.loc[theta]['recall']=recall
        outcome_list.loc[theta]['F-score']=F
    return outcome_list
        


# In[9]:


trainning_outcome_base = evaluation_for_NN (list_theta,X,Y,trainning_outcome,net)
testing_outcome_base = evaluation_for_NN (list_theta,X_test,Y_test,testing_outcome,net)


# In[10]:


print(trainning_outcome_base)
print(testing_outcome_base)


# In[11]:


'''--------------------Visulisation for baseline model---------------
'''
plt.figure(figsize=(8, 5))
x_axis =trainning_outcome_base.index
plt.title('Result for trainning data (baseline model)')
plt.plot(x_axis, trainning_outcome_base.accuracy, color='green', label='accuracy',linestyle="dashed")
plt.plot(x_axis, trainning_outcome_base.precision, color='red', label='precision',linestyle="-.")
plt.plot(x_axis, trainning_outcome_base.recall,  color='skyblue', label='recall',linestyle=":")
plt.plot(x_axis, trainning_outcome_base['F-score'], color='blue', label='F-score',linestyle="--")
#plt.plot([0.50,0.55],[0.98125,0.98125],marker='v',label="maximum",color='black',alpha=0.6,linewidth=3,markersize=10)
plt.legend() 

plt.xlabel('threshold')
plt.ylabel('rate')
plt.show()


# Plot the performance of the network with different threshold for testing set 
plt.figure(figsize=(8, 5))
x_axis =testing_outcome_base.index
plt.title('Result for testing data (baseline model)')
plt.plot(x_axis, testing_outcome_base.accuracy, color='green', label='accuracy',linestyle="dashed")
plt.plot(x_axis, testing_outcome_base.precision, color='red', label='precision',linestyle="-.")
plt.plot(x_axis, testing_outcome_base.recall,  color='skyblue', label='recall',linestyle=":")
plt.plot(x_axis, testing_outcome_base['F-score'], color='blue', label='F-score',linestyle="--")
accu = testing_outcome_base.accuracy
#plt.plot(0.45,0.825,marker='v',label="maximum",color='black',alpha=0.6,markersize=10)
plt.legend() 
plt.xlabel('threshold')
plt.ylabel('rate')
plt.show()


# In[12]:


# use PCA for visulise a 6-D data in a 2-D plane 
instance = PCA(n_components=2)
reduced_x = instance.fit_transform(np.array(train_x))
# This is the distribution of the data whose label are ground truth   
for i in range(len(reduced_x)):
    if train_y.iloc[i]==0:
        plt.scatter(reduced_x[i][0], reduced_x[i][1],  c='blue', alpha=0.5)
    else:
        plt.scatter(reduced_x[i][0], reduced_x[i][1],  c='green', alpha=0.5)

    
plt.title("Ground Truth")
plt.xlabel('Principle component1')
plt.ylabel('Principle component2')
plt.show()


# In[13]:


''' -----------------------FCNN Model 1 construction-------------------- 
 clustering the trainning data into two clusters ,use membership value as addtional feature
'''
colors = ['b', 'orange']
ncenters = 2
instance_trainning = PCA(n_components=2)
trainning_two_D = instance_trainning.fit_transform(np.array(train_x))
trainning_two_D_test = instance_trainning.fit_transform(np.array(test_x))
dim1 = trainning_two_D[:,0]
dim2 = trainning_two_D[:,1]
combined = np.vstack((dim1, dim2))
combined_test = np.vstack((trainning_two_D_test[:,0], trainning_two_D_test[:,1]))
cntroid, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(train_x.T, ncenters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
for i in range(ncenters):
    plt.plot(dim1[cluster_membership == i],dim2[cluster_membership == i], '.', color=colors[i],label='class '+str(i))
    for pt in cntroid:
        plt.plot(pt[0], pt[1], 'rs')
plt.title('Fuzzy clustering')
plt.xlabel('Principle component1')
plt.ylabel('Principle component2')
plt.legend()
print('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
_, u_test,_, _, _,_, fpc_test = fuzz.cluster.cmeans(combined_test, ncenters, 2, error=0.005, maxiter=1000, init=None)


# In[14]:


''' Enrich the raw data using clustering membership value 
'''
clustering_membership= u[0]
added_trainning = np.column_stack((train_x,clustering_membership))
added_testing = np.column_stack((test_x,u_test[0]))
input_neurons = added_trainning.shape[1]
hidden_neurons = 10
output_neurons = 2
learning_rate = 0.01
num_epoch = 600

# A standard backpropergation Neural network with two hidden layer
class Network(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Network, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
        self.dropout =torch.nn.Dropout(p=0.5)

    def forward(self, x):

        # get hidden layer input
        out = self.hidden1(x)
        out = torch.sigmoid(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        y_pred = self.out(out)
        
        return y_pred
net = Network(input_neurons, hidden_neurons, output_neurons)
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser with regularisation term 
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.001)
X_added, Y_added = torch.tensor(np.array(added_trainning),dtype=torch.float), torch.tensor(np.array(train_y),dtype=torch.long)
X_added_test, Y_added_test = torch.tensor(np.array(added_testing),dtype=torch.float), torch.tensor(np.array(test_y),dtype=torch.long)

input_neurons = added_trainning.shape[1]
begin = perf_counter() 
trainning_loss = []   
testing_loss =[]
for epoch in range(num_epoch):
    Y_pred = net(X_added)
    loss_1 = loss_func(Y_pred, Y_added)
    trainning_loss.append(loss_1.item())
    test_pred = net(X_added_test)
    loss_2 = loss_func(test_pred, Y_added_test)
    testing_loss.append(loss_2)

    # print progress
    if epoch % 40 == 0:
        predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
        correct = correct_counter(predicted.data,Y.data)        
        print('epoch: {}, loss: {}, accuracy:{}'.format(epoch, loss_1.data.item(),100 * correct/320),'%')

    optimiser.zero_grad()
    loss_1.backward()
    optimiser.step()
time =  perf_counter()-begin
print('time',time)     


# In[15]:


plt.figure()
plt.plot(trainning_loss,label='training loss')
plt.plot(testing_loss,label='testing loss')
plt.legend() 
plt.xlabel('number of iterations')
plt.ylabel('loss')
plt.title('The loss changes respect to number of epoches ')
plt.show()


# In[16]:


trainning_outcome = evaluation_for_NN (list_theta,X_added,Y_added,trainning_outcome,net)
testing_outcome = evaluation_for_NN (list_theta,X_added_test,Y_added_test,testing_outcome,net)


# In[17]:


print(trainning_outcome)
print(testing_outcome)


# In[18]:


plt.figure(figsize=(8, 5))
x_axis =testing_outcome_base.index
plt.title('Result for testing data (model with additional membership value)')
plt.plot(x_axis, testing_outcome.accuracy, color='green', label='accuracy',linestyle="dashed")
plt.plot(x_axis, testing_outcome.precision, color='red', label='precision',linestyle="-.")
plt.plot(x_axis, testing_outcome.recall,  color='skyblue', label='recall',linestyle=":")
plt.plot(x_axis, testing_outcome['F-score'], color='blue', label='F-score',linestyle="--")
accu = testing_outcome.accuracy
#plt.plot(0.45,0.825,marker='v',label="maximum",color='black',alpha=0.6,markersize=10)
plt.legend() 
plt.xlabel('threshold')
plt.ylabel('rate')
plt.show()


# In[19]:


#plot the comparison between model 1 and baseline model 

model1 = testing_outcome.iloc[5,:].values[3:]
basemodel = testing_outcome_base.iloc[5,:].values[3:]


# In[20]:


plt.figure(figsize=(16,8))
width = 0.5
basemodel = [0.80, 0.818, 0.73, 0.771] 
for i in range(0,4):
    plt.bar(1.2+(i*width*4)+(width*1), model1[i], width,color= 'y',alpha = 0.5)
    plt.bar(1.2+(i*width*4)+(width*2), basemodel[i], width,color='r',alpha = 0.5)
plt.xticks([2,4,6,8], ['accuracy','precision','recall','F-measure'])
plt.title("performance of two models",size=20)
plt.ylabel("rate", size=16)
plt.legend( labels=['FCNN model1','baseline model'])
plt.show()


# In[21]:


''' -----------------------FCNN Model 4 construction-------------------- 
 clustering the trainning data into two clusters ,stack the centres of clusters as additional trainning sample 
'''
data0 =  raw_data_scaled[raw_data_scaled['Label'].isin(['0'])]
data1 = raw_data_scaled[raw_data_scaled['Label'].isin(['1'])]
data0_x = data0.drop(labels=['Label'],axis=1)
data1_x = data1.drop(labels=['Label'],axis=1)
data0_y = data0['Label']
data1_y = data1['Label']


# In[22]:


# fuzzy c-means clustering 
number_of_clusters = 100   # change the number for comparison 
n=int(number_of_clusters/2)
cntroid_0, u_for_0, _, _, _, _, fpc0 =fuzz.cluster.cmeans(data0_x.T, n, 2, error=0.005, maxiter=1000, init=None)
cntroid_1, u_for_1, _, _, _, _, fpc1 =fuzz.cluster.cmeans(data1_x.T, n, 2, error=0.005, maxiter=1000, init=None)


# In[23]:



added_trainning = np.row_stack((train_x,cntroid_0,cntroid_1))
added_label = np.row_stack((train_y.values.reshape(-1,1),np.zeros(n).reshape(-1,1),np.ones(n).reshape(-1,1))).flatten()
X_added, Y_added = torch.tensor(np.array(added_trainning),dtype=torch.float), torch.tensor(np.array(added_label),dtype=torch.long)
X_test, Y_test = torch.tensor(np.array(test_x),dtype=torch.float), torch.tensor(np.array(test_y),dtype=torch.long)


# In[24]:


begin = perf_counter()

input_neurons = added_trainning.shape[1]
hidden_neurons = 20
output_neurons = 2
learning_rate = 0.001
num_epoch = 2000

# A standard backpropergation Neural network with two hidden layer
class Network(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Network, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
        self.dropout =torch.nn.Dropout(p=0.3)
        

    def forward(self, x):

        # get hidden layer input
        out = self.hidden1(x)
        out = torch.sigmoid(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        y_pred = self.out(out)
        
        return y_pred
net = Network(input_neurons, hidden_neurons, output_neurons)
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser with regularisation term 
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.001)
X_added, Y_added = torch.tensor(np.array(added_trainning),dtype=torch.float), torch.tensor(np.array(added_label),dtype=torch.long)
#X_added_test, Y_added_test = torch.tensor(np.array(added_testing),dtype=torch.float), torch.tensor(np.array(test_y),dtype=torch.long)

input_neurons = added_trainning.shape[1]

trainning_loss = []   
testing_loss =[]
for epoch in range(num_epoch):
    Y_pred = net(X_added)
    loss_1 = loss_func(Y_pred, Y_added)
    trainning_loss.append(loss_1.item())
    test_pred = net(X_test)
    loss_2 = loss_func(test_pred, Y_test)
    testing_loss.append(loss_2)

    # print progress
    if epoch % 200 == 0:
        predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
        correct = correct_counter(predicted.data,Y_added.data)        
        print('epoch: {}, loss: {}, accuracy:{}'.format(epoch, loss_1.data.item(),100 * correct/(320+number_of_clusters)),'%')

    optimiser.zero_grad()
    loss_1.backward()
    optimiser.step()
time =  perf_counter()-begin
print('time',time) 


# In[25]:


plt.figure()
plt.plot(trainning_loss,label='training loss')
plt.plot(testing_loss,label='testing loss')
plt.legend() 
plt.xlabel('number of iterations')
plt.ylabel('loss')
plt.title('The loss changes respect to number of epoches ')
plt.show()


# In[26]:


trainning_outcome_added = evaluation_for_NN (list_theta,X_added,Y_added,trainning_outcome,net)
testing_outcome_added = evaluation_for_NN (list_theta,X_test,Y_test,testing_outcome,net)
print(trainning_outcome_added)
print(testing_outcome_added)


# In[27]:


plt.figure(figsize=(8, 5))
x_axis =trainning_outcome_added.index
plt.title('Result for testing data (model with additional data)')
plt.plot(x_axis, testing_outcome_added.accuracy, color='green', label='accuracy',linestyle="dashed")
plt.plot(x_axis, testing_outcome_added.precision, color='red', label='precision',linestyle="-.")
plt.plot(x_axis, testing_outcome_added.recall,  color='skyblue', label='recall',linestyle=":")
plt.plot(x_axis, testing_outcome_added['F-score'], color='blue', label='F-score',linestyle="--")
#plt.plot(0.45,0.825,marker='v',label="maximum",color='black',alpha=0.6,markersize=10)
plt.legend() 
plt.xlabel('threshold')
plt.ylabel('rate')
plt.show()


# In[28]:


''' -----------------------FCNN Model 3 construction-------------------- 
reduce the number of trainning samples, use the centres of clusters as trainning data,discard raw data 
'''

# reduce the number of trainning samples, use the centres of clusters as trainning data  
reduced_trainning = np.row_stack((cntroid_0,cntroid_1))
reduced_label = np.row_stack((np.zeros(n).reshape(-1,1),np.ones(n).reshape(-1,1))).flatten()
X_reduced, Y_reduced = torch.tensor(np.array(reduced_trainning),dtype=torch.float), torch.tensor(np.array(reduced_label),dtype=torch.long)


# In[29]:


begin = perf_counter()
input_neurons = reduced_trainning.shape[1]
hidden_neurons = 20
output_neurons = 2
learning_rate = 0.001
num_epoch = 1200

# A standard backpropergation Neural network with two hidden layer
class Network(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Network, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
        

    def forward(self, x):

        # get hidden layer input
        out = self.hidden1(x)
        out = torch.sigmoid(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        y_pred = self.out(out)
        
        return y_pred
net = Network(input_neurons, hidden_neurons, output_neurons)
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser with regularisation term 
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.001)
X_added, Y_added = torch.tensor(np.array(added_trainning),dtype=torch.float), torch.tensor(np.array(added_label),dtype=torch.long)
#X_added_test, Y_added_test = torch.tensor(np.array(added_testing),dtype=torch.float), torch.tensor(np.array(test_y),dtype=torch.long)

input_neurons = added_trainning.shape[1]

trainning_loss = []   
testing_loss =[]
for epoch in range(num_epoch):
    Y_pred = net(X_reduced)
    loss_1 = loss_func(Y_pred, Y_reduced)
    trainning_loss.append(loss_1.item())
    test_pred = net(X_test)
    loss_2 = loss_func(test_pred, Y_test)
    testing_loss.append(loss_2)

    # print progress
    if epoch % 200 == 0:
        predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
        correct = correct_counter(predicted.data,Y_reduced.data)        
        print('epoch: {}, loss: {}, accuracy:{}'.format(epoch, loss_1.data.item(),100 * correct/len(X_reduced)),'%')

    optimiser.zero_grad()
    loss_1.backward()
    optimiser.step()
    
time =  perf_counter()-begin
print('time',time) 


# In[30]:


plt.figure()
plt.plot(trainning_loss,label='training loss')
plt.plot(testing_loss,label='testing loss')
plt.legend() 
plt.xlabel('number of iterations')
plt.ylabel('loss')
plt.title('The loss changes respect to number of epoches ')
plt.show()


# In[31]:


trainning_outcome_reduced = evaluation_for_NN (list_theta,X_reduced,Y_reduced,trainning_outcome,net)
testing_outcome_reduced = evaluation_for_NN (list_theta,X_test,Y_test,testing_outcome,net)
print(trainning_outcome_reduced)
print(testing_outcome_reduced)


# In[32]:


plt.figure(figsize=(8, 5))
x_axis =trainning_outcome_reduced.index
plt.title('Result for testing data (model with reduced data)')
plt.plot(x_axis, testing_outcome_reduced.accuracy, color='green', label='accuracy',linestyle="dashed")
plt.plot(x_axis, testing_outcome_reduced.precision, color='red', label='precision',linestyle="-.")
plt.plot(x_axis, testing_outcome_reduced.recall,  color='skyblue', label='recall',linestyle=":")
plt.plot(x_axis, testing_outcome_reduced['F-score'], color='blue', label='F-score',linestyle="--")
#plt.plot(0.45,0.825,marker='v',label="maximum",color='black',alpha=0.6,markersize=10)
plt.legend() 
plt.xlabel('threshold')
plt.ylabel('rate')
plt.show()


# In[33]:


'''--------------For FCNN Model 3,compare different numbers of clusters----------------- 
'''

cluster_num = [50,100,200]
df = []
accuracy_test = []
accuracy_train=[]
for n in cluster_num:
    begin = perf_counter()
    trainning_loss = []
    num_each = int(n/2)
    cntroid_0, u_for_0, _, _, _, _, fpc0 =fuzz.cluster.cmeans(data0_x.T, num_each, 2, error=0.005, maxiter=1000, init=None)
    cntroid_1, u_for_1, _, _, _, _, fpc1 =fuzz.cluster.cmeans(data1_x.T, num_each, 2, error=0.005, maxiter=1000, init=None)
    reduced_trainning = np.row_stack((cntroid_0,cntroid_1))
    reduced_label = np.row_stack((np.zeros(num_each).reshape(-1,1),np.ones(num_each).reshape(-1,1))).flatten()
    X_reduced, Y_reduced = torch.tensor(np.array(reduced_trainning),dtype=torch.float), torch.tensor(np.array(reduced_label),dtype=torch.long)
    input_neurons = reduced_trainning.shape[1]
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    net = Network(input_neurons, hidden_neurons, output_neurons)


    input_neurons = reduced_trainning.shape[1]
    hidden_neurons = 20
    output_neurons = 2
    learning_rate = 0.001
    num_epoch = 2000

# A standard backpropergation Neural network with two hidden layer
    class Network(torch.nn.Module):

        def __init__(self, n_input, n_hidden, n_output):
            super(Network, self).__init__()
            self.hidden1 = torch.nn.Linear(n_input, n_hidden)
            self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
            self.out = torch.nn.Linear(n_hidden, n_output)
        

        def forward(self, x):

        # get hidden layer input
            out = self.hidden1(x)
            out = torch.sigmoid(out)
            out = self.hidden2(out)
            out = torch.sigmoid(out)
            y_pred = self.out(out)
        
            return y_pred
    net = Network(input_neurons, hidden_neurons, output_neurons)
    loss_func = torch.nn.CrossEntropyLoss()

# define optimiser with regularisation term 
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.001)
    X_added, Y_added = torch.tensor(np.array(added_trainning),dtype=torch.float), torch.tensor(np.array(added_label),dtype=torch.long)
#X_added_test, Y_added_test = torch.tensor(np.array(added_testing),dtype=torch.float), torch.tensor(np.array(test_y),dtype=torch.long)

    input_neurons = added_trainning.shape[1]
    
    trainning_loss = [] 
    testing_loss =[]
    
    for epoch in range(num_epoch):
        Y_pred = net(X_reduced)
        loss_1 = loss_func(Y_pred, Y_reduced)
        optimiser.zero_grad()
        loss_1.backward()
        optimiser.step()
        trainning_loss.append(loss_1.item())
        if epoch % 40 == 0:
            predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
            correct = correct_counter(predicted.data,Y_reduced.data)
            accuracy_train.append(correct/len(predicted))
            Y_pred = net(X_test)
            predicted = threshold_classifier(torch.softmax(Y_pred,dim=1),0.5)
            correct = correct_counter(predicted.data,Y_test.data)
            accuracy_test.append(correct/len(predicted))

        
    df.append(evaluation_for_NN (list_theta,X_test,Y_test,testing_outcome,net)) 
    plt.plot(trainning_loss,label='segment ='+str(n))
    time =  perf_counter()-begin
    #print('time',time) 

plt.plot(base_loss,label='traditional NN')
plt.legend()
plt.xlabel('number of iterations')
plt.ylabel('loss')
plt.title('The loss changes respect to number of epoches for different segments ')
plt.show()


# In[36]:


plt.figure()
segment_50_train= accuracy_train[:50]
segment_100_train=accuracy_train[50:100]
segment_200_train=accuracy_train[100:]
plt.plot(segment_50_train,label='segment = 50')
plt.plot(segment_100_train,label='segment =100')
plt.plot(segment_200_train,label='segment = 200')
plt.plot(accuracy_train_base,label='baseline')

d = 6
plt.xticks([0,d*1,d*2,d*3,d*4,d*5,d*6,d*7,d*8], ['0','250','500','750','1000','1250','1500','1750','2000'])
plt.legend()
plt.xlabel('iterations')
plt.ylabel('accuracy rates')
plt.title('Trainning accuracy for model with different number of clusters')
plt.show()


# In[37]:


segment_50_test= accuracy_test[:50]
segment_100_test=accuracy_test[50:100]
segment_200_test=accuracy_test[100:]
plt.plot(segment_50_test,label='segment = 50')
plt.plot(segment_100_test,label='segment = 100')
plt.plot(segment_200_test,label='segment =200')
plt.plot(accuracy_test_base,label='baseline')

d = 6
plt.xticks([0,d*1,d*2,d*3,d*4,d*5,d*6,d*7,d*8], ['0','250','500','750','1000','1250','1500','1750','2000'])

plt.legend()
plt.xlabel('iterations')
plt.ylabel('accuracy rates')
plt.title('Testing accuracy for model with different number of clusters')
plt.show()


# In[ ]:




