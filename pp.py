from pandas import read_csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDRegressor, BayesianRidge
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings

dd = read_csv("Data.csv", header=0, sep=";", dtype={"latitude": float, "longitude": float})
print(dd)
X = dd[["year"]]
y = dd[["latitude", "longitude", "date", "month"]]

seed = 7
scoring = 'accuracy'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('SDGR', SGDRegressor()))
models.append(('BRD', BayesianRidge()))
    # linear_model.LassoLars(),
    # linear_model.ARDRegression(),
    # linear_model.PassiveAggressiveRegressor(),
    # linear_model.TheilSenRegressor(),
    # linear_model.LinearRegression()
results = []
names = []
final = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        final.append(cv_results.mean())