# -*- coding: utf-8 -*-
import numpy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import flask
import pickle
import pandas as pd

app = flask.Flask(__name__)
with open('model.pkl', 'rb') as file:
    classifier = pickle.load(file)


@app.route('/', methods=['POST', 'GET'])
def prog():
    matplotlib.rc("font", size=18)
    data = pd.read_excel("zad.xlsx")
    data = data.iloc[0:].values
    figure = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection="3d")
    list_age = []
    list_exp = []
    list_pro = []
    list_w = []
    for i in data:
        list_age.append(i[1])
        list_exp.append(i[2])
        list_pro.append(i[3])
        list_w.append(i[4])
    X = [list_pro, list_w, list_age]
    ax.scatter(list_w, list_age, list_pro)
    [c1, c2, c3] = KMeans(n_clusters=3).fit(X).cluster_centers_
    plt.scatter(c1[0], numpy.mean(list_age), c1[2], marker="x", c="black")
    plt.scatter(c2[0], numpy.mean(list_age), c2[2], marker="x", c="black")
    plt.scatter(c3[0], numpy.mean(list_age), c3[2], marker="x", c="black")
    plt.show()


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
