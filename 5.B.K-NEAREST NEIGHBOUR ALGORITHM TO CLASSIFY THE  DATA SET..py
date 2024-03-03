import numpy as np   
import matplotlib.pyplot as plt   
import pandas as pd     
data_set= pd.read_csv('Social_Network_Ads.csv')  
x= data_set.iloc[:,[2,3]].values   
y= data_set.iloc[:, 4].values     
from sklearn.model_selection import train_test_split   
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.20, random_state=0)  
print('len',len(y_test)) 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 
print(x_train) 
print(x_test) 
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)  
classifier.fit(x_train, y_train) 
y_pred = classifier.predict(x_test) 
print('y-Pred',y_pred) 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
print(cm) 
from sklearn.metrics import accuracy_score 
sc1 = accuracy_score(y_test, y_pred)*100 
print('The Accuracy   is  ',sc1)  
from matplotlib.colors import ListedColormap  
X_set, y_set = x_train, y_train  
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, 
step = 0.01),  
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,  
cmap = ListedColormap(('red', 'green')))  
plt.xlim(X1.min(), X1.max())  
plt.ylim(X2.min(), X2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), 
label = j)  
plt.title('K-NN (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()   
from matplotlib.colors import ListedColormap  
X_set, y_set = x_test, y_test  
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, 
step = 0.01),  
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,  
cmap = ListedColormap(('red', 'green')))  
plt.xlim(X1.min(), X1.max())  
plt.ylim(X2.min(), X2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), 
label = j)  
plt.title('K-NN (Test set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show() 
