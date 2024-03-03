import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB 
from sklearn import metrics data = pd.read_csv('navebyase.csv', names=['text','label']) 
print("\n The Dataset is :\n", data) 
print("\n The Dimensions of the dataset", data.shape) 
data['labelnum'] = data.label.map({'positive':1, 'negitive':0}) 
x= data.text 
y= data.labelnum 
print(x) 
print(y) 
vectorizer = TfidfVectorizer() 
data = vectorizer.fit_transform(x) 
print("\n the TF-IDF features of Dataset:\n") 
df = pd.DataFrame(data.toarray(), columns= vectorizer.get_feature_names()) 
df.head() 
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=2) 
print("\n the total number of taraning data :" ,y_train.shape) 
print("\n the total number of test data :" ,y_test.shape) 
clf = MultinomialNB().fit(x_train, y_train) 
predicted = clf.predict(x_test) 
print('\n Accuracy of classsifier :', metrics.accuracy_score(y_test, predicted)) 
print("\n confusion matrix: ",metrics.confusion_matrix(y_test, predicted)) 
print("\n classification report :",metrics.classification_report(y_test, predicted)) 
print("\n the value of precision: ", metrics.precision_score(y_test, predicted)) 
print("\n the value of recall: ",metrics.recall_score(y_test, predicted))
