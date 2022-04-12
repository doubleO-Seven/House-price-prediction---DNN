import tensorflow as tf
from tensorflow import keras #ML Library
import numpy as np #Mathematics Library
import pandas as pd #Data-handling Library

df = pd.read_csv('drive/Shareddrives/doubleOseven/Keras/2. Deep Neural Network/HousingPrices.csv')
df.head() #print 1st n(5) values of the dataset

X = df.drop(columns=['SalePrice']) # x axix e price bad e sob value thakbe. tai drop use korsi. drop e j value likha thakbe seta bad e
# baki sib gula add korbe, drop price mane price take drop korse x axix theke
Y = df[['SalePrice']]

model = keras.models.Sequential()
# 3-5 ,3ta Neuron 
model.add(keras.layers.Dense(8, activation='relu', input_shape=(8,))) #8ta input dicchi dense = 8 means nuron no, besically joto gula input toto gula neuron no dewa hoy
model.add(keras.layers.Dense(8, activation='relu')) # relu = rectifying linear units. 0 to 1 er majhe thake. o er kase hoile 0 ar 1 er kache value hoile 1 show kore
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=30, callbacks=[keras.callbacks.EarlyStopping(patience=3)]) #loss jodi khub choto hoy, loss<<30 tahole seta k early stop kore dibo
#so cpu er kaj kombe kichuta, usage kombe cpu er

test_data = np.array([1900,	854,	1710,	2,	1,	3,	7,	2000]) # x axix er 8ta input value
print(model.predict(test_data.reshape(1,8), batch_size=1))

model.save('saved_model.h5')
!ls