import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft

t=np.linspace(0,3,12*1024)
N1 = 4

left =  [130.81,146.83,164.81,174.61] 
right = [261.63,293.66,329.63,349.23] 

ts = [0, 0.5, 1, 1.5]
T = [0.5, 0.5, 0.5, 1]

x = np.zeros(len(t))
for i in range(N1):
    x += np.sin(2 * np.pi * left[i] * (t - ts[i])) * (t >= ts[i]) * (t < ts[i] + T[i])
    x += np.sin(2 * np.pi * right[i] * (t - ts[i])) * (t >= ts[i]) * (t < ts[i] + T[i])
    
N = 3*1024
f=np.linspace(0,512,int(N/2))

xf =fft(x)
xf =2/N*np.abs(xf[0:int(N/2)])


f1 = np.random.randint(0, 512)
f2 = np.random.randint(0, 512)
n = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

xn = x + n

xnf = fft(xn)
xnf = 2/N * np.abs(xnf[0:int(N/2)])

max1 = np.argmax(xnf)
max2 = np.argpartition(xnf, -2)[-2]

f1rounded = round(f[max1])
f2rounded = round(f[max2])

x_filtered = xn - (np.sin(2 * np.pi * f1rounded * t) + np.sin(2 * np.pi * f2rounded * t))

x_filteredf = fft(x_filtered)
x_filteredf = 2/N * np.abs(x_filteredf[0:int(N/2)])

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(t, x)
plt.title('Original Signal')
plt.subplot(3,1,2)
plt.plot(t, xn)
plt.title('Noisy Signal')
plt.subplot(3,1,3)
plt.plot(t, x_filtered)
plt.title('Filtered Signal')

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(f, xf)
plt.title('Original Signal')
plt.subplot(3,1,2)
plt.plot(f, xnf)
plt.title('Noisy Signal')
plt.subplot(3,1,3)
plt.plot(f, x_filteredf)
plt.title('Filtered Signal')

sd.play(x_filtered, 4 * 1024)
