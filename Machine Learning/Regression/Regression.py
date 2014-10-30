#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#load data
dataFile = open("./data/data.csv")
data = np.loadtxt(dataFile)
train_x = np.array([[x] for x in data[:-2,1]])
train_y = [x for x in data[:-2,2]]
test_x = np.array([[x] for x in data[-2:,1]])
test_y = [x for x in data[-2:,2]]

#regression | fit the model
model = LinearRegression()
model.fit(train_x,train_y)

#predict
predict_y = model.predict(test_x)
print(model.predict(test_x) , test_y)
print(model.score(test_x, test_y))
print(model.coef_,model.intercept_)

#plot
plt.figure()

plt.scatter(test_x, test_y,  color='red')
plt.scatter(train_x,train_y,color='blue')

minmax_x = np.array([[x] for x in{min(data[:,1]) , max(data[:,1])} ])
plt.plot(minmax_x,model.predict(minmax_x),color='black',linewidth=3)

plt.text(13, 45, "$y={0}x+{1}$".format(model.coef_[0],model.intercept_))
plt.xlabel("Max depth of snow")
plt.ylabel("Area of irrigation")

plt.show()