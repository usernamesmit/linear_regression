import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle

data = pd.read_csv("student/student-mat.csv", sep=";")

# Attributes
data = data[['G1','G2','G3','studytime','failures','absences']]

# Label
predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# x_train = 90% of x & y_train = 90% of y
# x_test = 10% of x & y_train = 10% of y

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

# Training the model
linear.fit(x_train, y_train)

# Testing the model with accuracy
acc = linear.score(x_test, y_test)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])



# data = pd.read_csv("student-mat.csv", sep=";")

# data = data[["G1","G2","G3","studytime","failures","absences"]]

# predict = "G3"

# x = np.array(data.drop([predict], 1))
# y = np.array(data[predict])

# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# # linear = linear_model.LinearRegression()

# # linear.fit(x_train, y_train)
# # acc = linear.score(x_test, y_test)

# # Creating a pickle file which stores the trained model, so that we don't have to train it again.

# # with open("studentmodel.pickle","wb") as f:
    
# #     pickle.dump(linear, f)

# pickle_in = open("studentmodel.pickle","rb")
# linear = pickle.load(pickle_in)


# predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])

# p = "G1"

# pyplot.style.use("ggplot")
# pyplot.scatter(data[predict],data[p])
# pyplot.xlabel("Final Grades")
# pyplot.ylabel(p)
# pyplot.show()


    
