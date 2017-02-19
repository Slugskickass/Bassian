import numpy as np
from scipy import stats
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('/Users/Ashley/Dropbox/test.tif')
img.seek(5000)
out=np.array(img.copy())
[X,Y]=np.shape(out)
out_line=np.squeeze(np.resize(out,[X*Y,1]))
print(np.shape(out_line))

print('The number of points',np.shape(out_line))
N=np.size(out_line)
mean_val=np.mean(out_line)
selected_points = np.squeeze(np.where(out_line > mean_val))






selected_points_signal = np.array(np.zeros((N,1)))
selected_points_signal[selected_points] = True
selected_points_noise = np.logical_not(selected_points_signal)



select_points_signal=out_line[np.uint(selected_points_signal)]
select_points_noise=out_line[np.uint(selected_points_noise)]
length_signal=np.sum(selected_points_signal)
length_noise=np.sum(selected_points_noise)

tot = np.sum(out_line)
sig = np.sum(select_points_signal)
noi = np.sum(select_points_noise)

plt.plot(selected_points,out_line[selected_points],'o')
plt.plot(out_line)
plt.plot(200*selected_points_signal)
plt.show()