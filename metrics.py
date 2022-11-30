import numpy as np
from sklearn.metrics import mean_squared_error

def RMSE(Y_real, Y_pre):
    '''
    Y_real为实际值，Y_pre为预测值
    计算RMSE
    '''
    return np.sqrt(mean_squared_error(Y_real, Y_pre))


def MSE(Y_real, Y_pre):
    '''
    Y_real为实际值，Y_pre为预测值
    计算MSE
    '''
    return mean_squared_error(Y_real, Y_pre)