import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:\\Users\\Nikhil\\Desktop\\spam.csv')
#print(df.head())

x=df["text"]
y=df["type"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

cv = CountVectorizer()
features = cv.fit_transform(x_train)

tuned_parameters = {'kernel':['linear','rbf'],'gamma':[1e-3,1e-4],
                    'C':[1,10,100,1000]}

model = GridSearchCV(svm.SVC(),tuned_parameters)
model.fit(features,y_train)

features_test = cv.transform(x_test)

print("Accuracy of model is ",model.score(features_test,y_test))