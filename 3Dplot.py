# https://pundit.pratt.duke.edu/wiki/Python:Interpolation

import os,sys,math,csv,numpy as npy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import interp2d
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from loadData import loadPVTData
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


coeff = [1, -6, 11, -6]
answer = npy.roots(coeff)

print answer
# print 


'''
x=np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2])
y=np.array([0.57,0.85,0.66,0.84,0.59,0.55,0.61,0.76,0.54,0.55,0.48])

x_new = np.linspace(x.min(), x.max(),500)

f = interp1d(x, y, kind='quadratic')
y_smooth=f(x_new)

plt.plot (x_new,y_smooth)
plt.scatter (x, y)

plt.show()
'''

'''
P0,T0,R0,M0,I0 = loadPVTData('DataCO2/CO2_Superheat_Hassan.csv','isobar')

d = []                                                                
d.append(range(0,5))                                                    
d.append(range(2,4))

print d

sumRes_precise =  npy.nan

# a = (sumRes_precise != npy.nan)
# print a

b = math.isnan(sumRes_precise)
print b
'''

if False:
	#Figure of Exp Data
	fig1 = plt.figure(num=1, clear=True)
	ax = fig1.add_subplot(1, 1, 1, projection='3d')
	ax.scatter(P0, T0, R0, c='r', marker='o')
	ax.set(xlabel="Pressure", ylabel="Temperature", zlabel="Density")
	ax.view_init(elev=45, azim=45)
	# fig1.savefig('Experimental Data')


	P0m, T0m = npy.meshgrid(P0, T0)
	P_array_basic = list(npy.linspace(5, 15, 30))
	T_array_basic = list(npy.linspace(350, 400, 30))

	# print P_array
	# print T_array
	zinterp = interp2d(P0, T0, R0, kind='linear')
	R_matrix = zinterp(P_array_basic, T_array_basic)

	R_array = []
	for i in range(len(R_matrix)):
		R_array = npy.append(R_array, R_matrix[i])

	T_array = npy.repeat(T_array_basic,len(P_array_basic))
	P_array = (P_array_basic * len(T_array_basic))

	# print R_matrix
	# print P_array_basic
	# print T_basic

	# print R_array
	# print P_array
	# print T_array

	# fig2 = plt.figure(num=1, clear=True)
	# ax = fig2.add_subplot(1, 1, 1, projection='3d')
	ax.scatter(P_array, T_array, R_array, c='b', marker='o')
	ax.set(xlabel="Pressure", ylabel="Temperature", zlabel="Density")
	ax.view_init(elev=45, azim=45)
	# fig2.savefig('Experimental Data')


	print(zinterp(7.5, 350))


	plt.show()



	'''
	# Convert to matrices for graphics
	xmodelm, ymodelm = npy.meshgrid(xmodelc, ymodelc)
	zmodelm = zmodelf.reshape(100, 100)
	# print zmodelm
	# print zmodelf
	# fig = plt.figure(num=2, clear=True)
	# ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax = fig.add_subplot(111, projection='3d')
	# ax.plot_wireframe(xmodelm, ymodelm, zmodelm, color='r')
	ax.set(xlabel="Pressure", ylabel="Temperature", zlabel="Density")

	# ax.scatter(xdatam, ydatam, zdatam, c='y', marker='o')
	# ax.scatter(xdatam, ydatam, zdatam, color='r', linewidth=12)

	# ax.plot_wireframe(xdatam, ydatam, zdatam, color='r')
	ax.scatter(14.5, 375.5, zinterp(14.5, 375.5), color='g', linewidth=12)
	ax.view_init(elev=45, azim=45)
	# fig.savefig('Interpolated Surface')

	##################################
	##################################

	#Setting font size
	axis_size = 6
	title_size = 6
	size = 6
	label_size = 6
	plt.rcParams['xtick.labelsize'] = label_size
	plt.rcParams['ytick.labelsize'] = label_size

	M = 44.01

	#Initializing the array of pressure and temperature.
	p = npy.linspace(3,15,200)
	t = npy.linspace(300,450,200)
	P,T = npy.meshgrid(p,t)

	#==============================================================================================================
	#Calculating the PVT surface.
	#==============================================================================================================

	#Initializing the PVT surface array.
	rho_surf = npy.zeros((len(P),len(T)))

	#Calculating the PVT surface.
	for i in range(0,len(p)):
		for j in range(0,len(t)):
			# rho_surf[i,j] = calculateDensity(p[i],t[j],M,Pstar=Pstar,Tstar=Tstar,Rstar=Rstar)
			rho_surf[i,j] = zinterp(p[i],t[j])    #calculateDensity(p[i],t[j],M,Pstar=Pstar,Tstar=Tstar,Rstar=Rstar)

	#==================================================================================
	#Relative deviation versus T plots.
	# fig = plt.figure(num=None, figsize=(2.00, 1.70), dpi=300, facecolor='w', edgecolor='k')
	# ax = fig.add_subplot(1, 1, 1, projection='3d')

	# ax = fig.add_subplot(111,projection='3d')
	# levels = npy.arange(0.01, 0.99, 0.01)
	#plt.contour(P,T,rho_surf,levels,linewidths=0.5)
	#RdYlBu
	surf = ax.plot_surface(P,T,rho_surf,rstride=3,cstride=3,cmap=cm.coolwarm,linewidth=0, antialiased=True)

	#Setting the axis limits.
	# ax.set_xlim3d(1.0,68)
	# ax.set_ylim3d(310.0,1100.0)
	# ax.set_zlim3d(0.0,1.0)

	#title = 'CO2 Sanchez-Lacombe EOS'
	param_graphic = r'My Plot'

	#Setting the axis labels and title.
	ax.set_xlabel('Pressure',fontsize=axis_size)
	ax.set_ylabel('Temperature',fontsize=axis_size)
	ax.set_zlabel('Density',fontsize=axis_size)
	ax.set_title(param_graphic,fontsize=axis_size)

	#Removing ticks and tick labels.
	# ax.set_xticks([])
	# ax.set_yticks([])
	# ax.set_zticks([])

	#Changing the figure orientation.
	#ax.view_init(elev=20.0,azim=45.0)

	#Saving the figure at the appropriate resolution. Note, saving the figure
	#displayed after plt.show() will not preserve resolution.
	# fig.savefig('TOC_graphic.png', dpi=300)

	#Show plot windows.
	plt.show()
	'''