# https://pundit.pratt.duke.edu/wiki/Python:Interpolation

import os,sys,math,csv,numpy as npy
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from loadExperimentalDataCO2 import *
from isListOrNpyArray import *

def myinterpolation_isobars(P,T,P0,T0,R0):
	# print P,T,P0,T0,R0
	found = 0
	for i in range(len(P0)):
			if P==P0[i] and T==T0[i]:
				R_inter = R0[i]
				found = 1
				P_high,Tbelow_Phigh,Rbelow_Phigh,Tabove_Phigh,Rabove_Phigh,P_low,Tbelow_Plow,Rbelow_Plow,Tabove_Plow,Rabove_Plow = float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan') 
				break
			elif P==P0[i] and T<T0[i]:
				R_inter = (((R0[i]-R0[i-1])/(T0[i]-T0[i-1]))*(T-T0[i-1]))+R0[i-1]
				found = 1
				Tbelow_Phigh = T0[i-1]
				Rbelow_Phigh = R0[i-1]
				Tabove_Phigh = T0[i]
				Rabove_Phigh = R0[i]
				P_high = P
				P_low,Tbelow_Plow,Rbelow_Plow,Tabove_Plow,Rabove_Plow = float('nan'),float('nan'),float('nan'),float('nan'),float('nan')
				break

	if found ==0:	
		for i in range(len(P0)):
				if P<P0[i]:
					P_low = P0[i-1]
					P_high = P0[i]
					break
		for i in range(len(P0)):
				if P_high == P0[i] and T<T0[i]:
					R_inter_high = (((R0[i]-R0[i-1])/(T0[i]-T0[i-1]))*(T-T0[i-1]))+R0[i-1]
					Tbelow_Phigh = T0[i-1]
					Rbelow_Phigh = R0[i-1]
					Tabove_Phigh = T0[i]
					Rabove_Phigh = R0[i]
					break
		for i in range(len(P0)):
				if P_low == P0[i] and T<T0[i]:
					R_inter_low = (((R0[i]-R0[i-1])/(T0[i]-T0[i-1]))*(T-T0[i-1]))+R0[i-1]
					Tbelow_Plow = T0[i-1]
					Rbelow_Plow = R0[i-1]
					Tabove_Plow = T0[i]
					Rabove_Plow = R0[i]
					break
		
		R_inter = (((R_inter_high-R_inter_low)/(P_high-P_low))*(P-P_low))+R_inter_low
	
	return [P,T,R_inter,P_high,Tbelow_Phigh,Rbelow_Phigh,Tabove_Phigh,Rabove_Phigh,P_low,Tbelow_Plow,Rbelow_Plow,Tabove_Plow,Rabove_Plow]


def myinterpolation_isotherms(P,T,P0,T0,R0):
	found = 0
	for i in range(len(T0)):
			if T==T0[i] and P==P0[i]:
				R_inter = R0[i]
				found = 1
				T_high,Pbelow_Thigh,Rbelow_Thigh,Pabove_Thigh,Rabove_Thigh,T_low,Pbelow_Tlow,Rbelow_Tlow,Pabove_Tlow,Rabove_Tlow = float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan') 
				break
			elif T==T0[i] and P<P0[i]:
				R_inter = (((R0[i]-R0[i-1])/(P0[i]-P0[i-1]))*(P-P0[i-1]))+R0[i-1]
				found = 1
				Pbelow_Thigh = P0[i-1]
				Rbelow_Thigh = R0[i-1]
				Pabove_Thigh = P0[i]
				Rabove_Thigh = R0[i]
				T_high = T
				T_low,Pbelow_Tlow,Rbelow_Tlow,Pabove_Tlow,Rabove_Tlow = float('nan'),float('nan'),float('nan'),float('nan'),float('nan')
				break

	if found ==0:	
		for i in range(len(T0)):
				if T<T0[i]:
					T_low = T0[i-1]
					T_high = T0[i]
					break
		for i in range(len(T0)):
				if T_high == T0[i] and P<P0[i]:
					R_inter_high = (((R0[i]-R0[i-1])/(P0[i]-P0[i-1]))*(P-P0[i-1]))+R0[i-1]
					Pbelow_Thigh = P0[i-1]
					Rbelow_Thigh = R0[i-1]
					Pabove_Thigh = P0[i]
					Rabove_Thigh = R0[i]
					break
		for i in range(len(T0)):
				if T_low == T0[i] and P<P0[i]:
					R_inter_low = (((R0[i]-R0[i-1])/(P0[i]-P0[i-1]))*(P-P0[i-1]))+R0[i-1]
					Pbelow_Tlow = P0[i-1]
					Rbelow_Tlow = R0[i-1]
					Pabove_Tlow = P0[i]
					Rabove_Tlow = R0[i]
					break
		
		R_inter = (((R_inter_high-R_inter_low)/(T_high-T_low))*(T-T_low))+R_inter_low
	
	return [P,T,R_inter,T_high,Pbelow_Thigh,Rbelow_Thigh,Pabove_Thigh,Rabove_Thigh,T_low,Pbelow_Tlow,Rbelow_Tlow,Pabove_Tlow,Rabove_Tlow]

def calculateR_interCO2(P,T,P0,T0,R0):
	
	if not isListOrNpyArray(P) and not isListOrNpyArray(T):
		answer = myinterpolation_isobars(P,T,P0,T0,R0)
		R_inter = answer[2]
		result = R_inter
    
  	elif not isListOrNpyArray(P) and isListOrNpyArray(T):
		result = [[range(0,len(T))] for x in range(2)]
		R_inter = range(0,len(T))
		for i in range(0,len(T)):
			answer = myinterpolation_isobars(P,T[i],P0,T0,R0)
			R_inter[i] = answer[2]
		result[0] = R_inter
		result[1] = T
    
  	elif isListOrNpyArray(P) and not isListOrNpyArray(T):

		result = [[range(0,len(P))] for x in range(2)]
		R_inter = range(0,len(P))
		for i in range(0,len(P)):
			answer = myinterpolation_isobars(P[i],T,P0,T0,R0)
			R_inter[i] = answer[2]
		result[0] = R_inter
		result[1] = P

  	elif isListOrNpyArray(P) and isListOrNpyArray(T):
		result = [[range(0,len(P))] for x in range(2)]
		R_inter = range(0,len(P))
		for i in range(0,len(P)):
			answer = myinterpolation_isobars(P[i],T[i],P0,T0,R0)
			R_inter[i] = answer[2]
		result[0] = R_inter
		result[1] = P

	return result


'''
P=14.5
T=375.5

#Isobars
result = myinterpolation_isobars(P,T,P0_isobars,T0_isobars,R0_isobars)
print result

#Isotherms
result = myinterpolation_isotherms(P,T,P0_isotherms,T0_isotherms,R0_isotherms)
print result
'''






'''

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(P0, T0, R0, color='r', marker='o')
ax.set(xlabel="Pressure", ylabel="Temperature", zlabel="Density")
ax.view_init(elev=45, azim=45)
# fig.savefig('Experimental Data')

plt.show()

'''