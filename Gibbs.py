from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import Base_Utils as bu
import time
from scipy import stats


number_iteractions =500


#img = Image.open('/Users/Ashley/Dropbox/test.tif')
#img.seek(5000)
#out=np.array(img.copy())

start = time.time()
outs= np.uint16(np.array(np.random.normal(loc=1200.0, scale=20.0, size=[5000,1])))
out_no= np.uint16(np.random.normal(loc=800.0, scale=200.0, size=[10000,1]))
out_line= np.append(out_no,outs)

#[X,Y]=np.shape(out)
#out_line=np.resize(out,[X*Y,1])
N=np.size(out_line)
print('The Data has',N,'number of elements')
print('The number of interations is',number_iteractions)

mean_val=np.mean(out_line)
selected_points = np.where(out_line > mean_val)

cuS = np.zeros((N,1))
cuS[selected_points] = 1

number_elements_one =np.sum(cuS)
number_elements_zero = N-number_elements_one

#output_mean_noise=np.zeros((number_iteractions,1))
#output_mean_signal=np.zeros((number_iteractions,1))
#output_lamba_noise=np.zeros((number_iteractions,1))
#output_lamda_signal=np.zeros((number_iteractions,1))
#output_mean_noise=np.zeros((number_iteractions,1))

output_mean_noise=[]
output_mean_signal=[]
output_lamba_noise=[]
output_lamda_signal=[]
output_counts=[]
output_prob=[]


selected_points_signal=np.zeros((N,1))              #Build empty array
selected_points_signal[selected_points]=True        #True means there is signal there
selected_points_noise = np.logical_not(selected_points_signal) #True means there is noise there

pms=np.zeros((2,1))
pmm=np.zeros((2,1))
#pms  =1 x 2 double [signal max and signal widrh
pms[0]=1000 #np.max(out_line)
pms[1]=100
#pmn =1 x 2 double [signal max and noise widrh
pmm[0]=100 #np.mean(out_line)
pmm[1]=10
current_lamda_signal = 10
current_lamda_noise = 1


select_points_signal=out_line[np.uint(selected_points_signal)]
select_points_noise=out_line[np.uint(selected_points_noise)]
length_signal=np.sum(selected_points_signal)
print((length_signal))
length_noise=np.sum(selected_points_noise)
print((length_noise))
print(N-length_noise-length_signal)

culs = 1
culn =10
a=[0,0]
pila=[2,0.6]
hold=[]

for I in range(0,number_iteractions):
    cuP=np.random.beta(number_elements_one+0.5,number_elements_zero+0.5)

    [Current_signal_mean, Current_noise_mean] = bu.new_means(select_points_noise, select_points_signal, current_lamda_signal, current_lamda_noise, pms,
                              pmm, length_signal, length_noise)

    a[0]=Current_signal_mean
    a[1]=Current_noise_mean
    b = np.sort(a)
    b = b[::-1]

    Current_signal_mean=b[0]
    Current_noise_mean=b[1]


    [Current_signal_prec, Current_noise_prec] = bu.new_precision(select_points_noise, select_points_signal, Current_signal_mean, Current_noise_mean, pila,
                  length_signal, length_noise)

    auxn = stats.norm.logpdf(out_line, loc=Current_noise_mean, scale=(1/np.sqrt(Current_noise_prec)))
    auxs = stats.norm.logpdf(out_line, loc=Current_signal_mean, scale=(1/np.sqrt(Current_signal_prec)))

    pi = (1+ ((1-cuP)/cuP)*np.exp(auxn-auxs))
    pi=1/pi


    bunnies = np.random.binomial(1,pi)
    selected_points_signal =bunnies >0
    selected_points_noise = np.logical_not(selected_points_signal) #True means there is noise there
    select_points_signal=out_line[np.uint(selected_points_signal)]
    select_points_noise=out_line[np.uint(selected_points_noise)]
    length_signal=np.sum(selected_points_signal)
    length_noise=np.sum(selected_points_noise)

    output_mean_noise.append(Current_noise_mean)
    output_mean_signal.append(Current_signal_mean)
    output_lamba_noise.append(Current_noise_prec)
    output_lamda_signal.append(Current_signal_prec)
   # output_counts = output_counts+ bunnies
    output_prob.append(cuP)

# Image_stack.number_of_stacks, is the number of full chunks in the data
end = time.time()
print('Load time: ' + str(end - start) + ' seconds')
#temper=np.reshape(pi,[128,128])

calc_s= np.uint16(np.array(np.random.normal(loc=output_mean_signal[-1], scale=output_lamda_signal[-1], size=[5000,1])))

#plt.hist(calc_s)
#plt.hist(out_line)
#plt.show()

