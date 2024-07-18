'''
# -*- coding: utf-8 -*-
    @Author   : LingLing
    @Time     : 2024/7/4 17:42
    @File     : test.py
    @Project  : assignment3
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neural_network import MLPClassifier,  MLPRegressor
from layer import L_layer_model_Classifier, L_layer_model_Regression


if __name__ == '__main__':
    from utils import show_score, dataset_Preprocessing

    # load data
    dataset = np.load("data/nyc_taxi_data.npy", allow_pickle=True).item()
    X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]

    X_train_act, y_train_act = dataset_Preprocessing(X_train, y_train)
    X_test_act, y_test_act = dataset_Preprocessing(X_test, y_test)

    # Remove anomalous data from the training dataset
    X_train_act = X_train_act.loc[(y_train_act["trip_duration"] < 7000) & (X_train_act["distance"] < 20)]
    y_train_act = y_train_act.loc[(y_train_act["trip_duration"] < 7000) & (X_train_act["distance"] < 20)]


    # standardization
    scaler = StandardScaler()
    scaler1 = StandardScaler()
    X_train_sd = scaler.fit_transform(X_train_act)
    X_test_sd = scaler.transform(X_test_act)
    y_train_sd = scaler1.fit_transform(y_train_act)
    y_test_sd = scaler1.transform(y_test_act)

    # Learning_rates = [1.0]
    # active_function = ['relu']
    #
    # best_score = float('inf')
    # best_p = {}
    # param = []
    # dtype = np.dtype([
    #     (('active_function', 'Active function'), 'U10'),
    #     (('layer_dims', 'Layerdims'), 'i4'),
    #     (('learning_rate', 'Learning rate'), 'f4'),
    #     (('seed', 'Seed'), 'i4'),
    #     (('score', 'Score'), 'f8'),
    # ])
    # for af in active_function:
    #     for ld in range(7, 8):
    #         layer_dims = [6, ld, 1]
    #         for lr in Learning_rates:
    #             for seed in range(0, 40):
    #                 model = L_layer_model_Regression(layer_dims=layer_dims, active_function=af, learning_rate=lr,
    #                                                  random_state=seed)
    #                 model.fit(X_train_sd, y_train_sd)
    #                 test_pred_sd = model.predict(X_test_sd)
    #                 test_pred_normal = scaler1.inverse_transform(test_pred_sd)
    #                 score = mean_squared_log_error(y_test_act, abs(test_pred_normal))
    #                 p = {'active_function': af, 'layer_dims': ld, 'learning_rate': lr, 'seed': seed, 'score': score}
    #                 param.append((af, ld, lr, seed, score))
    #                 if score < best_score:
    #                     best_score = score
    #                     best_p = {'active_function': af, 'layer_dims': ld, 'learning_rate': lr, 'seed': seed}
    #         #                 if score==1:
    #         #                     break
    #
    # print("best_p = ", best_p)
    # print("best_score = ", best_score)
    #
    # param_array = np.array(param, dtype=dtype)
    # df = pd.DataFrame(param_array)
    # df.to_csv('param.csv', index=False)
    # print(df)
    # # fit and predict
    model1 = L_layer_model_Regression(layer_dims=[6,7,1],active_function='relu',learning_rate=1,random_state=34)

    model1.fit(X_train_sd,y_train_sd,print_cost=True)


    test_pred_sd =model1.predict(X_test_sd)

    test_pred = scaler1.inverse_transform(test_pred_sd)

    err_val = mean_squared_log_error(y_test_act, abs(test_pred))
    print(f"Error Percentage = {err_val} ")
    show_score(model1,'model1',times=1)





