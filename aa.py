from pandas import read_csv
from sklearn import linear_model
#from googleplaces import GooglePlaces
from urllib.request import urlopen
import json
import warnings

def getplace(lat, lon):
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, lon)
    while True:
        try:
            v = urlopen(url).read()
            j = json.loads(v)
            components = j['results'][0]['formatted_address']
            flag = 0        # Successful data transfer in components 
            print(components)
        except:
            flag = 1        # Unsuccessful data transfer in components
        if flag == 0:
            break


dd = read_csv("Data.csv", header=0, sep=";")
X = dd[["year"]]
y1 = dd[["latitude"]]
y2 = dd[["longitude"]]
y3 = dd[["date"]]
y4 = dd[["month"]]
models = [('BayesianRidge', linear_model.BayesianRidge()),
    ('LassoLars', linear_model.LassoLars()),
    ('ARDRegression', linear_model.ARDRegression())]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    year_input = input("Enter year for prediction of Terrorist attacks : ")
    year_input = float(year_input)
    for name,i in models:
        print("\n", name)
        print("===========================================")
        clf = i
        clf.fit(X, y1)
        lat = clf.predict(year_input)
        for i in lat:
            new_lat = float("{0:.6f}".format(i))
        print("Latitude : ", new_lat)
        clf.fit(X, y2)
        lng = clf.predict(year_input)
        for i in lng:
            new_lng = float("{0:.6f}".format(i))
        print("Longitude : ", new_lng)
        clf.fit(X, y3)
        date = clf.predict(year_input)
        for i in date:
            dte = int(i)
        print("Date : ", dte)
        clf.fit(X, y4)
        month = clf.predict(year_input)
        for i in month:
            mnt = int(i)
        print("Month : ", mnt)
        print("Place deduced from latitude and longitude : \n========")
        getplace(new_lat, new_lng)