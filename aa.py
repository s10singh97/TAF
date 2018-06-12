from pandas import read_csv
from sklearn import linear_model
import warnings

dd = read_csv("Data.csv", header=0, sep=";")
X = dd[["year"]]
y1 = dd[["latitude"]]
y2 = dd[["longitude"]]
y3 = dd[["date"]]
y4 = dd[["month"]]
models = [('BayesianRidge', linear_model.BayesianRidge()),
    ('LassoLars', linear_model.LassoLars()),
    ('ARDRegression', linear_model.ARDRegression()),
    ('PassiveAggressiveRegressor', linear_model.PassiveAggressiveRegressor()),
    ('TheilSenRegressor', linear_model.TheilSenRegressor()),
    ('LinearRegression', linear_model.LinearRegression())]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    c = 1
    for name,i in models:
        print("\n", name)
        print("===========================================")
        clf = i
        clf.fit(X, y1)
        print("Latitude : ", clf.predict(2018))
        clf.fit(X, y2)
        print("Longitude : ", clf.predict(2018))
        clf.fit(X, y3)
        print("Date : ", clf.predict(2018))
        clf.fit(X, y4)
        print("Month : ", clf.predict(2018))
        c += 1