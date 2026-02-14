# Implementation of Univariate Linear Regression
## Aim:
To implement univariate Linear Regression to fit a straight line using least squares.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
1.	Get the independent variable X and dependent variable Y.
2.	Calculate the mean of the X -values and the mean of the Y -values.
3.	Find the slope m of the line of best fit using the formula.
 ![eqn1](./eq1.jpg)
4.	Compute the y -intercept of the line by using the formula:
![eqn2](./eq2.jpg)  
5.	Use the slope m and the y -intercept to form the equation of the line.
6.	Obtain the straight line equation Y=mX+b and plot the scatterplot.
## Program
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics
boston=datasets.load_diabetes(return_X_y=False)

#defining feature matrix(X) and response vector (y)
x=boston.data
y=boston.target
#splitting x and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

#create linear regression object
reg=linear_model.LinearRegression()

#train the model using the training sets
reg.fit(x_train,y_train)

#regression coefficients
print("Coefficients",reg.coef_)

#variance score: 1means perfect prediction
print("Variance score: {}".format(reg.score(x_test,y_test)))

#plot for residual error
#setting plot style
plt.style.use("fivethirtyeight")

#plotting residual errors in training data
plt.scatter(reg.predict(x_train),reg.predict(x_train)-y_train,color='green',s=10,label="Train data")

#plotting residual errors in test data
plt.scatter(reg.predict(x_test),reg.predict(x_test)-y_test,color='blue',s=10,label="Test data")

#plotting line for zero residual error
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)

#plotting legend
plt.legend(loc='upper right')

#plot title
plt.title('Residual errors')

##method call for showing the plot
plt.show()





```
## Output
![WhatsApp Image 2026-02-14 at 10 52 55 AM](https://github.com/user-attachments/assets/1cccc1e7-0d30-45da-9ec2-1ef47dd2d94c)


## Result
Thus the univariate Linear Regression was implemented to fit a straight line using least squares.
