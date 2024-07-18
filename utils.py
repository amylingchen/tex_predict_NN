'''
# -*- coding: utf-8 -*-
    @Author   : LingLing
    @Time     : 2024/7/4 16:03
    @File     : utils.py
    @Project  : assignment3
'''
import math
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

current_folder = os.getcwd()
def save_parameter(models, filename):
    fil_path = current_folder + "/data/" + filename + ".pkl"
    model_dict_=[]
    for model in models:
        model_dict_.append(model.__dict__)

    # save parameter to file
    with open(fil_path, 'wb') as f:
        pickle.dump(model_dict_, f)

def load_parameter(models, filename):
    fil_path = current_folder + "/data/" + filename + ".pkl"


    with open(fil_path, 'rb') as f:
        model_dict_= pickle.load(f)

    # load parameter to models
    for i in range(len(models)):
        models[i].__dict__ = model_dict_[i]

    return models

def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j


    return X, Y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def show_score(model, picname, times=10):
    '''
    show train_score and val_score plot with times parameter
    '''
    #get train_score and val_score with times parameter
    train_scores = model.train_losses[::times]
    val_scores = model.val_losses[::times]

    #draw plot
    plt.plot(train_scores, label='Train Cost')
    plt.plot(val_scores, label='Validation Cost')
    plt.ylabel('cost')
    plt.xlabel('epoch ( per ' + str(times) + ' )')
    plt.title(picname)
    plt.legend()

    #save plot to file
    pic_path = current_folder + "/images/" + picname + "_cost.png"
    plt.savefig(pic_path)
    plt.show()

    print(f"Plot saved : {picname}_cost.png")


def dataset_Preprocessing(X, y):
    transform_data = pd.DataFrame(X)
    transform_data['pickup_datetime'] = pd.to_datetime(transform_data['pickup_datetime'])

    transform_data['pickup_weekday'] = transform_data['pickup_datetime'].dt.dayofweek
    transform_data['pickup_month'] = transform_data['pickup_datetime'].dt.month
    transform_data['pickup_hour'] = transform_data['pickup_datetime'].dt.hour
    #     transform_data['pickup_time'] = transform_data['pickup_datetime'].dt.minute

    # transform_data['dropoff_datetime'] = pd.to_datetime(transform_data['dropoff_datetime'])
    # transform_data['dropoff_weekday'] = transform_data['dropoff_datetime'].dt.dayofweek
    # transform_data['dropoff_month'] = transform_data['dropoff_datetime'].dt.month
    # transform_data['dropoff_hour'] = transform_data['dropoff_datetime'].dt.hour
    #     transform_data['dropoff_time'] = transform_data['dropoff_datetime'].dt.minute

    # transform_data['store_and_fwd_flag'] = transform_data.store_and_fwd_flag.map({'N': 0, 'Y': 1})
    mean_longitude = np.mean(transform_data['pickup_longitude'])
    mean_latitude = np.mean(transform_data['pickup_latitude'])
    transform_data.loc[:, 'long'] = transform_data['pickup_longitude'] - mean_longitude
    transform_data.loc[:, 'lat'] = transform_data['pickup_latitude'] - mean_latitude

    transform_data['distance'] = transform_data.apply(
        lambda row: get_distance(row['pickup_latitude'], row['pickup_longitude'],
                                 row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
    transform_data.drop(
        ["id", "vendor_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
         'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'store_and_fwd_flag'],
        axis=1, inplace=True)

    #     transform_target = np.array(y).reshape(-1, 1)
    transform_target = pd.DataFrame(y)

    return transform_data, transform_target


def get_distance(lat1, lon1, lat2, lon2):
    # Calculate distance by latitude and longitude

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat1 - lat1
    dlon = lon1 - lon2

    # Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371  # Radius of the Earth in kilometers
    distance = R * c

    return distance


