import numpy as nm   
import matplotlib.pyplot as mtp   
import pandas as pd   
#importing datasets   
data_set= pd.read_csv("E:/user_data.csv")    
#Extracting Independent and dependent Variable   
x= data_set.iloc[:, [1,2]].values   
y= data_set.iloc[:, 3].values   
# Splitting the dataset into training and test set.   
from sklearn.model_selection import train_test_split   
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)   
#feature Scaling   
from sklearn.preprocessing import StandardScaler     
st_x= StandardScaler()   
x_train= st_x.fit_transform(x_train)     
x_test= st_x.transform(x_test)     
#Fitting Decision Tree classifier to the training set   
from sklearn.tree import DecisionTreeClassifier 
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)   
classifier.fit(x_train, y_train)   
DecisionTreeClassifier(class_weight=None, 
max_depth=None,max_features=None, 
criterion='entropy', 
max_leaf_nodes=None,min_impurity_decrease=0.0,min_samples_leaf=1, 
min_samples_split=2,min_weight_fraction_leaf=0.0,random_state=0, splitter='best') 
#Predicting the test set result   
y_pred= classifier.predict(x_test)   
#Creating the Confusion matrix   
from sklearn.metrics import confusion_matrix   
cm= confusion_matrix(y_test, y_pred) 
print(cm) 
from sklearn.metrics import accuracy_score 
sc1 = accuracy_score(y_test, y_pred)*100 
print("Accuracy is",sc1) 
#Visulaizing the trianing set result   
from matplotlib.colors import ListedColormap 
x_set, y_set = x_train, y_train 
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01), 
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01)) 
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), 
alpha = 0.75, cmap = ListedColormap(('purple','green' ))) 
mtp.xlim(x1.min(), x1.max()) 
mtp.ylim(x2.min(), x2.max()) 
for i, j in enumerate(nm.unique(y_set)): 
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('purple', 'green'))(i),label = j) 
mtp.title('Decision Tree Algorithm (Training set)') 
mtp.xlabel('Age') 
mtp.ylabel('Estimated Salary') 
mtp.legend() 
mtp.show() 
#Visulaizing the test set result 
from matplotlib.colors import ListedColormap 
x_set, y_set = x_test, y_test 
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01), 
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01)) 
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), 
alpha = 0.75, cmap = ListedColormap(('purple','green' ))) 
mtp.xlim(x1.min(), x1.max()) 
mtp.ylim(x2.min(), x2.max()) 
for i, j in enumerate(nm.unique(y_set)): 
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('purple', 'green'))(i), label = j) 
mtp.title('Decision Tree Algorithm(Test set)') 
mtp.xlabel('Age') 
mtp.ylabel('Estimated Salary') 
mtp.legend() 
mtp.show()
