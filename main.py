from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import Base_Utils as bu

number_iteractions =10


img = Image.open('/Users/Ashley/Desktop/test.tif')
img.seek(5000)
out=np.array(img.copy())
[X,Y]=np.shape(out)
out_line=np.resize(out,[X*Y,1])
N=np.size(out_line)


mean_val=np.mean(out_line)                          #Select the mean value of the data and use it
selected_points = np.where(out_line > mean_val)     #to select the points above that value

selected_points_signal=np.zeros((N,1))              #Build empty array
selected_points_signal[selected_points]=True        #True means there is signal there
selected_points_noise = np.logical_not(selected_points_signal) #True means there is noise there

pms=np.zeros((2,1))
pmm=np.zeros((2,1))
#pms  =1 x 2 double [signal max and signal widrh
pms[0]=np.max(out_line)
pms[1]=10
#pmn =1 x 2 double [signal max and noise widrh
pmm[0]=np.mean(out_line)
pmm[1]=10
current_lamda_signal = 10
current_lamda_noise = 1


select_points_signal=out_line[np.uint(selected_points_signal)]
select_points_noise=out_line[np.uint(selected_points_noise)]
length_signal=np.sum(selected_points_signal)

length_noise=np.sum(selected_points_noise)

[one,two]=bu.new_means(select_points_noise,select_points_signal,current_lamda_signal,current_lamda_noise, pms, pmm,length_signal,length_noise)
print(one,two)
