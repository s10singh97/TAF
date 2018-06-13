from pandas import read_csv
from sklearn import linear_model
from googleplaces import GooglePlaces
from urllib.request import urlopen
import json
import warnings

# def API(latitude, longitude):
#     API_KEY = "API Key goes here"
#     google_places = GooglePlaces(API_KEY)
#     coordinates = {'lat': latitude, 'lng': longitude}
#     query_result = google_places.nearby_search(lat_lng=coordinates)
#     if query_result.has_attributions:
#         print("yes")
#     else:
#         print("no")

def getplace(lat, lon):
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, lon)
    v = urlopen(url).read()
    j = json.loads(v)
    components = j['results'][0]['formatted_address']
    # country = town = None
    # for c in components:
    #     if "country" in c['types']:
    #         country = c['long_name']
    #     if "postal_town" in c['types']:
    #         town = c['long_name']
    # print(town)
    # print(country)
    print(components)

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
    c = 1
    for name,i in models:
        print("\n", name)
        print("===========================================")
        clf = i
        clf.fit(X, y1)
        lat = clf.predict(2018)
        for i in lat:
            new_lat = float("{0:.6f}".format(i))
        print("Latitude : ", new_lat)
        clf.fit(X, y2)
        lng = clf.predict(2018)
        for i in lng:
            new_lng = float("{0:.6f}".format(i))
        print("Longitude : ", new_lng)
        clf.fit(X, y3)
        print("Date : ", clf.predict(2018))
        clf.fit(X, y4)
        print("Month : ", clf.predict(2018))
        print("Place deduced from latitude and longitude : \n========")
        getplace(new_lat, new_lng)
        c += 1