import os,sys,math,csv,matplotlib.pyplot as plt,numpy as npy
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
# import cmath
from scipy import interpolate

func_value = [2.05908203,	1.57226562,	0.51000977,	2.32226562,	2.32275391,	2.75585938,	4.13681641,	4.65234375,	4.91464844,	4.83984375,	4.61933594,	4.44140625,	4.08408203,	3.79296875,	3.35546875,	3.35107422,	2.94140625,	2.46513672,	1.95507812,	1.48032227,	0.20744629]	
time = [272.5,	273.57142857,	280,	287.5,	287.50001,	289.28571429,	295,	297.14285714,	302.5,	305,	310,	312.85714286,	317.5,	320.71428571,	325,	325.0001,	328.57142857,	332.5,	336.42857143,	340,	349.88888889]

# time = [0,1,2,3,4,5,6,7]
# func_value = [0,1,4,9,16,25,36,49]
# time = npy.array(time)
# func_value = npy.array(func_value)

# print time
# print func_value

f = interpolate.interp1d(time, func_value,kind='linear')
time_new1 = npy.arange(273, 352, 5)
func_value_new1 = f(time_new1)   # use interpolation function returned by `interp1d`

f = interpolate.interp1d(time_new1, func_value_new1,kind='cubic')
time_new = npy.arange(273, 345.01, 0.01)
func_value_new = f(time_new)   # use interpolation function returned by `interp1d`

plt.plot(func_value,time, 'o', func_value_new, time_new, '-')
plt.show()