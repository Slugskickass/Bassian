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


mean_val=np.mean(out_line)
selected_points = np.squeeze(np.where(out_line > mean_val))
print(np.shape(selected_points))
plt.plot(selected_points,out_line[selected_points],'o')
plt.plot(out_line)
plt.show()