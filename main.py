import pandas as pd
#import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error

import numpy as np

#from sklearn.metrics import max_error

heart = pd.read_csv('heart.csv')
test = pd.read_csv('Heart_test.csv')
#values = [89,1,3,145,233,1,0,150,0,2.3,0,0,1]

#test_transposed = test.transpose()

X = heart.iloc[:, 0:13]
Y = heart.iloc[:, 13:14]

test = test.iloc[:, 0:13]

#print(X.head(), '\n', test.head(),'\n')

model = tree.DecisionTreeRegressor()
model = model.fit(X, Y)
y_prediction = model.predict(test)
#y_score = model.score(test,y_prediction)
#print(y_score)


def mean_value(y_data):
    mean = 0
    for i in range(0, len(y_data)):
        mean += y_data.iloc[i]
    mean = mean / len(y_data)
    return mean


'''
#softmax function
def softmax(x):
  sum = 0
  arr = []
  for i in range(0,len(x)):
    sum += np.exp(x.iloc[i].target)
    arr.append(np.exp(i)/sum)
  return(max(arr))
print(softmax(Y))
'''


def sigmoid(x):
    arr = []
    for i in range(0, len(x)):
        arr.append(1 / (1 + np.exp(-(x.iloc[i].target))))
    return max(arr)


print('Prediction on test data: ', y_prediction[0], '\nConfidence of model: ',
      sigmoid(Y))
'''
print('mean: ',mean_value(Y))
  
print('mean squared error with mean value: ',mean_squared_error(mean_value(Y),y_prediction))

print('softmax: ',softmaxe(Y))

print('mean squared error with softmax: ',mean_squared_error(softmaxe(Y),y_prediction[0]))

'''

#softmax_value = softmax(Y).reshape(-1,1)
#y_prediction_reshaped = y_prediction[0].reshape(-1,1)
#print('score with softmax: ', model.score(softmax_value,y_prediction_reshaped))

#print(model.tree_.node_count)

#print(np.zeros(shape=model.tree_.node_count,dtype=np.int64))
tree.plot_tree(model)
plt.show()
