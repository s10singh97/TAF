from pandas import read_csv

dd = read_csv("Data.csv", header=0, sep=";", dtype={"date(DD/MM/YYYY)": str})
print(dd)
# latitude = dd["latitude"].tolist()
# longitude = dd["longitude"].tolist()
# date = dd["date(DD/MM/YYYY)"].tolist()
# print(latitude)
# print(longitude)
# for i in date:
#     i = int(i)
#     print(i)
