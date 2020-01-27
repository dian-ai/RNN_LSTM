#predicting trend in Google Stock price with RNN, with LSTM
#we train our LSTM model on 5 years of Google stock price, then we predict for one whole month

#Part 1 - Preprocessing

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set, RNN will be trained on training set
# so import data as data frame, to read with panda with CSV function, then we need to select the right column, we need to make it a numpy array because the only numpy arrays can be the input of NN in keras

#this is our dataframe

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

#it is the training set that our RNN will be trained, with iloc method we get all the right indexes
#in brackets we say we want all of it so we put :, then we need to make a numpy array thats why we cannot just say that we need column 1, we need to write as range therefore we write 1:2, so it makes it a numpy array of 1 column.
#this part of code so far makes a dataframe, to make it as numpy array we need to add .values

training_set = dataset_train.iloc[:, 1:2].values

#Feature scaling
#either standardization, or normalization, here we go for normalization

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))

training_set_scaled =sc.fit_transform(training_set)


#creating specific data structure, what the RNN will need to remember when predicting the nect stock price : right number of time steps, because wrong number of time steps will lead to overfitting or nonsence predictions

#Creating a data structure with 60 timesteps and 1 output
#RNN will be looking at 60 stock prices before time T (60 days before time T) then it predicts the next out put, so we will have two entities, 1st for 60 previous financial days as input, and 2nd is for output for the next day. and we need to do it for everytime T.
# initialize them empty

X_train= []
Y_train= []

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    Y_train.append(training_set_scaled[i,0])
#X_train and Y_train are list and we need to make them numpy array, so that they can be accepted by our future RNN
    
X_train, Y_train = np.array(X_train), np.array(Y_train)


#Reshaping the Data
#adding more dimensionality to the data structure that we made, the diamond is unit, the number of predictors that we can use to predict what we want! (if you want)

#anytime you want to add  a dimension inot numpy array, always use reshape function, go tu keras documentation to see what the new shape should be 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



#Part 2 - Building the RNN

#buil the whole architecture of neural network, we make stacked LSTM with some drop out regularization to prevent overfitting

#importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialize the RNN
#Thats the first step, we will build the initiation of our sequential layers, then we will use pytorch

#creating a sequence of layers
regressor = Sequential()

#now we can add different layers to make it a powerful stacked LSTM

# Adding the first LSTM layer and some drop out regularization
#3 arguments for LSTM : 1- unit (number of LSTM cells you want to have in LSTM layer) 2-return sequences = True, because we are building staacked LSTM, when you add another LSTM on the initial one it should be True, at final layer when you are done you can set it to False or but its default 3- input shape, exact shape of input X_train that we created before, but only the last two number we enter

regressor.add(LSTM(units=50, return_sequences = True, input_shape=(X_train.shape[1], 1) ))

#dropout, rate of the neurons that you want to drop and ignor
regressor.add(Dropout(0.2))

#we will make 4 layers of LSTM and some drop out to each of them







#Part3 - Making the predictions and visualising the results



