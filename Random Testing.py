# https://pundit.pratt.duke.edu/wiki/Python:Interpolation

import os,sys,math,csv,numpy as npy
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# from scipy.interpolate import interp2d
# lib_path = os.path.abspath(os.path.join('..'))
# sys.path.append(lib_path)
# from loadData import loadPVTData
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import AutoMinorLocator
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
import types
from sympy import *
from scipy.optimize import bisect

r = 4.65374716332
P = 9.7
Psstar = 419.91552
Ptilde = P/Psstar
x = Symbol('x',real=True)
y = Ptilde + (x**2) + (((-2*x)/((1-(1/r))-(1/(1-x))))*(((1-(1/r))*x)+ln(1-x)))
answer = nsolve(y,x,0.1065,verify=True)
print answer

def density_at_discontinuities(x,r,Ptilde):

	y = Ptilde + (x**2) + (((-2*x)/((1-(1/r))*(1-x)-1))*(((1-(1/r))*x*(1-x))+((1-x)*ln(1-x))))

	return y

x1 = bisect(density_at_discontinuities,0,(1/(sqrt(r)+1)),args=(r,Ptilde))
x2 = bisect(density_at_discontinuities,(1/(sqrt(r)+1)),1,args=(r,Ptilde))

print x1, x2

'''
from inspect import currentframe #To get line number in Print
def get_linenumber():
	#To get line number in Print
    cf = currentframe()
    return cf.f_back.f_lineno

print get_linenumber()
'''

'''
# print npy.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])
# array([ True, False, False, False, False,  True])

# Y = -5.79829066331+4.55640490659j
Y = float(0)
# Z = (Y.real, Y.imag)

A = Y.real
B = Y.imag

print A
print B

if isinstance(Y, types.ComplexType):
	print 'yes it is complex'

if Y.real<0:
	print 'yes'
if Y.imag<5:
	print 'yes again'

'''