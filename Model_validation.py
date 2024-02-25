import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

SEED = 301
np.random.seed(SEED)

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
datas = pd.read_csv(uri)
print (datas)

change = {
    'no' : 0,
    'yes' : 1
}
datas.sold = datas.sold.map(change)

currentYear = datetime.today().year
datas['model_age'] = currentYear - datas.model_year
head = datas.head()
print(head)

datas['Kilometers_per_year'] = datas.mileage_per_year * 1.0934
head = datas.head()
print(head)

datas = datas.drop(columns = ['Unnamed: 0', 'mileage_per_year', 'model_year'], axis = 1)
head = datas.head()
print(head)

x = datas[['price', 'model_age', 'Kilometers_per_year']]
y = datas['sold']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.25, stratify = y)

dummy_stratified = DummyClassifier()
dummy_stratified.fit(trainX, trainY)
dummyAccuracy = dummy_stratified.score(testX, testY) * 100

print("%.2f%%" % dummyAccuracy)

rawTrainX, rawTestX, trainY, testY = train_test_split(x, y, test_size = 0.25, stratify = y)

model = DecisionTreeClassifier(max_depth = 2)
model.fit(rawTrainX, trainY)
predictions = model.predict(rawTestX)
accuracyScore = accuracy_score(testY, predictions)
print("The tree accuracy was: %.2f " % (accuracyScore * 100))

features = x.columns
dotData = export_graphviz(model, out_file=None,
                          filled = True, rounded = True,
                          class_names = ['no', 'yes'],
                          feature_names = features)
grafico = graphviz.Source(dotData)
#grafico.view()

def printResults (results):
    average = results['test_score'].mean()
    standardDeviation = results['test_score'].std()
    print("Test results = ", results['test_score'])
    print("Average accuracy = %.2f" % (average * 100))
    print("Range accuracy = [%.2f, %.2f]" % ((average - 2 * standardDeviation) * 100 , (average + 2 * standardDeviation) * 100))

cv = StratifiedKFold(n_splits= 10, shuffle=True)
results = cross_validate(model, x, y, cv = cv)
print('StratifiedKFold results: ')
printResults (results)

#Simulando modelos de carros aleatórios para poder usar o GroupKfold

np.random.seed(SEED)
datas['model'] = datas.model_age + np.random.randint(-2, 3, size=10000)
print(datas.model.unique())

cv = GroupKFold(n_splits= 10)
results = cross_validate(model, x, y, cv = cv, groups = datas.model)
print('GroupKFold results: ')
printResults (results)

#Cross validation com StandardScaler
scaler = StandardScaler()
scaler.fit (trainX)
scaledTrainX = scaler.transform(trainX)
scaledTestX = scaler.transform(testX)
print(scaledTestX)

#Usando o GroupKFold para o SVC
model = SVC()
model.fit(scaledTrainX, trainY)
predictions = model.predict(scaledTestX)
print(predictions)

accuracyScore = accuracy_score(testY, predictions) * 100
print('The accuracy was %.2f' % accuracyScore)

pipeline = Pipeline([('transformation', scaler), ('estimator', model)])

cv = GroupKFold(n_splits=10)
results = cross_validate(pipeline, x, y, cv=cv, groups=datas.model)
printResults (results)