from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def new_means(select_points_noise,select_points_signal,current_lamda_signal,current_lamda_noise, pms, pmm,length_signal,length_noise):
    p1=(length_signal*current_lamda_signal)+pms[1]
    m1=(current_lamda_signal*np.sum(select_points_signal))+pms[0]*pms[1]
    ms=np.random.normal(m1,1/(np.sqrt(p1)))

    # as above for noise
    p1 = (length_noise * current_lamda_noise) + pmm[1]
    m1 = (current_lamda_noise * np.sum(select_points_noise)) + pmm[0] * pmm[1]
    mn = np.random.normal(m1, 1 / (np.sqrt(p1)))
    return(ms,mn)


def new_precision(select_points_noise,select_points_signal,Current_signal_mean,Current_noise_mean, pila,length_signal,length_noise):

    p1=(length_signal/2)+pila[0]
    temp=select_points_signal-Current_signal_mean
    temp=np.square(temp)
    m1=pila[1]+sum(temp)/2
    ps=np.random.gamma(p1,1/m1)


    # as above for noise
    p1 = (length_noise / 2) + pila[0]
    temp = select_points_noise - Current_noise_mean
    temp = np.square(temp)
    m1 = pila[1] + sum(temp) / 2
    pn = np.random.gamma(p1, 1 / m1)
    return(ps,pn)