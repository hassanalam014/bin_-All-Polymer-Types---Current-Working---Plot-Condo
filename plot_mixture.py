# Author: Kier von Konigslow
# Date: May 2017
#
# Description: The purpose of this file is to plot the saturated PMMA/CO2 binary mixture information
#			   based on both experimental and theoretical data.
#

import os,sys,math,csv,matplotlib.pyplot as plt,numpy as npy
# from p_params import *
# from s_params import *
from loadExperimentalData import *
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from loadPhysicalConstants import *
from calculatePureVariables import calculateNewMolecularParameters
from wrapperFunctions import calculateBinarySolubilitySwelling
from calculateBinaryResidual import calculateBinarySSQ
from calculateBinaryVariablesCHV import *
from Parameters_of_Different_Polymers import *
from Parameters_for_Mixtures_and_Tg import *
# from All_Functions import *
from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
# from Tait_Parameters_of_Different_Polymers import *
# from loadExperimentalDataCO2 import *
# from CO2PVT_interpolation import *
from Split_Exp_Data_in_Isotherms import*
from collections import OrderedDict			#For Exotic Line Styles

Polymer_Type='PMMA'
Solvent='CO2'
Parameters_Paper ='Self_Grassia'			# P*T*R* and g,epsilon_2,x (PVT-Tg Data Paper or Direct P*T*R* Values Reference)
Cp_Polymer_Weight = '02kilo_POST_THESIS'	# g,epsilon_2,x (Cp Paper Reference)
Paper_Number = 'Paper15'						# Solubility or Swelling Data Reference
#for PS: 'Paper4_11_12'
#for PMMA: 'Paper15'
kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight}

Ppstar,Tpstar,Rpstar,Mp,Psstar,Tsstar,Rsstar,Ms,P_exp,Tg_exp=Parameters_of_Different_Polymers(**kwargs)
P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete=loadExperimentSwXData(**kwargs)
Far_Above_Data=False
P0_X,P0_X_above_Tg,P0_X_far_above_Tg,T0_X,T0_X_above_Tg,T0_X_far_above_Tg,X0_X,X0_X_above_Tg,X0_X_far_above_Tg,Rubber0_X,Rubber0_X_above_Tg,Rubber0_X_far_above_Tg,P0_S,P0_S_above_Tg,P0_S_far_above_Tg,T0_S,T0_S_above_Tg,T0_S_far_above_Tg,S0_S,S0_S_above_Tg,S0_S_far_above_Tg,Rubber0_S,Rubber0_S_above_Tg,Rubber0_S_far_above_Tg = SplitExperimental_X_Sw_Data(P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete,Far_Above_Data,**kwargs)
# v_0,alpha,B0,B1 = Tait_Parameters_of_Different_Polymers(**kwargs)

number_of_isotherm, result = Split_Isotherms(P0_X,T0_X,X0_X,'X')
P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5,P0_X_T6,T0_X_T6,X0_X_T6,P0_X_T7,T0_X_T7,X0_X_T7,P0_X_T8,T0_X_T8,X0_X_T8,P0_X_T9,T0_X_T9,X0_X_T9 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23],result[24],result[25],result[26]
# print P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5

number_of_isotherm_swelling, result = Split_Isotherms(P0_S,T0_S,S0_S,'S')
P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5,P0_S_T6,T0_S_T6,S0_S_T6,P0_S_T7,T0_S_T7,S0_S_T7,P0_S_T8,T0_S_T8,S0_S_T8,P0_S_T9,T0_S_T9,S0_S_T9 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23],result[24],result[25],result[26]
# print P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5

# P0_S_T2,T0_S_T2,S0_S_T2 = P0_S_T1,T0_S_T1,S0_S_T1
# P0_S_T1,T0_S_T1,S0_S_T1 =  P0_S_T5,T0_S_T5,S0_S_T5

Kier=False
Hassan=True  
Hassan_Var_Vol=False  
Condo=False  
Condo_Original=False 

kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight,'Kier':Kier,'Hassan':Hassan,'Hassan_Var_Vol':Hassan_Var_Vol,'Condo':Condo,'Condo_Original':True}

cepsilon_s,cepsilon_p,cz,czeta,epsilon_p,g,x,delta,zeta=Parameters_for_Mixtures_and_Tg(**kwargs)
cdelta=100.0

xS_infty=x*(Ppstar/(Tpstar*Rpstar))*(1+ln(1+g))
# print Ppstar,Tpstar,Rpstar,g,epsilon_p,x,xS_infty

#Paper12 and Paper13 Are Working Best Simultaneously
# zeta=	1.07136875	#0.95125453	#1.16681479	#1.09730204	#1.04532004	#1.16681479	#1.08884102	#1.04185590	#1.10678999	#1.03166800	#0.96584091	#1.07703406	#0.96584091	#1.04185590	#1.05128789	#0.93486423	#0.94730914
# delta=	1.23734185	#0.96557611	#1.46977534	#1.30622455	#1.19773798	#1.46977534	#1.29082731	#1.18464299	#1.33805847	#1.16318167	#1.04196138	#1.26576563	#1.04196138	#1.18464299	#1.19048236	#0.96152684	#0.98671535
# PS - CO2 Below:
# zeta = 1.07136995 #+/- 0.04821750 (4.50%) (init = 0.95)
# delta = 1.23734449 #+/- 0.10480090 (8.47%) (init = 1.344)
# PMMA - CO2 Below:
# zeta = 1.00191251 #+/- 0.03942875 (3.94%) (init = 1.1)
# delta = 1.00000000 #+/- 0.07766170 (7.77%) (init = 1.344)

# zeta = 1.10208611 #+/- 0.03942875 (3.94%) (init = 1.1)
# delta = 1.2 #+/- 0.07766170 (7.77%) (init = 1.344)
#===================================================================================
#Determining theoretical solubility/swelling.
#===================================================================================

print('DHV mixture parameters zeta = {} and delta = {}.'.format(zeta,delta))

gammas,vhs,epsilons = calculateNewMolecularParameters(Psstar,Tsstar,Rsstar,Ms)
vh = delta*vhs/NA
print('The hole volume is vh = {}.'.format(vh))

Pmin = min(P0_X)
Pmax = max(P0_X)
Tmin = min(T0_X)
Tmax = max(T0_X)

print('The pressure range is {}-{}MPa and the temperature range is {}-{} K.'.format(Pmin,Pmax,Tmin,Tmax))

# P0_X_T4 = npy.concatenate(([0.0001],P0_X_T4),axis=0)
# T0_X_T4 = npy.concatenate(([373.15],T0_X_T4),axis=0)
# X0_X_T4 = npy.concatenate(([0],X0_X_T4),axis=0)

# P0_S_T4 = npy.concatenate(([0.0001],P0_S_T4),axis=0)
# T0_S_T4 = npy.concatenate(([373.15],T0_S_T4),axis=0)
# S0_S_T4 = npy.concatenate(([1.0],S0_S_T4),axis=0)

P0 = npy.linspace(min(P0_X),max(P0_X),3)
# P0 = npy.linspace(2.0,17,15)
T1=0.0#T0_X_T1[0]	#403	#290
T2=T0_X_T2[0]	#423	#304
T3=T0_X_T3[0]	#463	#350
T4=T0_X_T4[0]	#423	#304
T5=0.0#T0_X_T5[0]	#463	#350
T6=0.0#T0_X_T6[0]	#423	#304
T7=0.0#T0_X_T7[0]	#463	#350
T8=0.0#T0_X_T8[0]	#463	#350
T9=0.0#T0_X_T9[0]	#463	#350

number_of_points = 10
P1 = npy.linspace(min(P0_X),max(P0_X_T1),number_of_points)
P2 = npy.linspace(min(P0_X),max(P0_X_T2),number_of_points)
P3 = npy.linspace(min(P0_X),max(P0_X_T3),number_of_points)
P4 = npy.linspace(min(P0_X),max(P0_X_T4),number_of_points)
# P5 = npy.linspace(min(P0_X),max(P0_X_T5),number_of_points)

verbose = False
if T1!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P1,T1,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T1_DHV = result[2]
	Sw_T1_DHV = result[3]
if T2!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P2,T2,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T2_DHV = result[2]
	Sw_T2_DHV = result[3]
if T3!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P3,T3,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T3_DHV = result[2]
	Sw_T3_DHV = result[3]
if T4!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P4,T4,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T4_DHV = result[2]	
	Sw_T4_DHV = result[3]
if T5!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P5,T5,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T5_DHV = result[2]	
	Sw_T5_DHV = result[3]
if T6!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P6,T6,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T6_DHV = result[2]	
	Sw_T6_DHV = result[3]
if T7!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P7,T7,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T7_DHV = result[2]	
	Sw_T7_DHV = result[3]
if T8!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P8,T8,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T8_DHV = result[2]	
	Sw_T8_DHV = result[3]
if T9!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P9,T9,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T9_DHV = result[2]	
	Sw_T9_DHV = result[3]

#===================================================================================
#Plotting the PMMA/CO2 mixture results.
#===================================================================================

#Setting saved image properties
img_extension = '.png'
img_dpi = None
output_folder = 'Post_Thesis'

#Checking for existence of output directory. If such a directory doesn't exist, one is created.
if not os.path.exists('./'+output_folder):
    os.makedirs('./'+output_folder)

#Setting font size
axis_size = 28  #Size  of x and y axis wordings (names)
title_size = 20 #We have no title
size = 18		#Size of legendre
label_size = 26	#Size of values of ticks on x and y axis

plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size

#Defining axes
solubility_axes = [7.5,42.5,0.0,0.14]
swelling_axes = [5,25,1.0,1.17]
# phi_axes = [5,25,0.0,1.0]
# TD_axes = [5,25,0.0,5.0]

#Markers
mark1 = 'o'
mark2 = 's'
mark3 = '>'
mark4 = '<'
mark5 = 'D'
mark6 = 'H'
mark7 = 'P'
mark8 = 'X'


linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

#Linestyles
ls1 = '-'
ls2 = '--'
ls3 = ':'
ls4 = '-.'
ls5 = linestyles['densely dashdotdotted']

#General line properties.
linewidth = 4
markersize = 12

#Plotting the solubility of the PMMA+CO2 mixture.
figX = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

if T1!=0.0:
	plt.plot(P0_X_T1,X0_X_T1,'k',marker=mark1,ls='',label='Experiment at {} K'.format(T0_X_T1[0]),ms=markersize)
if T2!=0.0:
	plt.plot(P0_X_T2,X0_X_T2,'k',marker=mark1,ls='',label='Experiment at {} K'.format(T0_X_T2[0]),ms=markersize)
if T3!=0.0:
	plt.plot(P0_X_T3,X0_X_T3,'k',marker=mark2,ls='',label='{} K'.format(T0_X_T3[0]),ms=markersize)
if T4!=0.0:
	plt.plot(P0_X_T4,X0_X_T4,'k',marker=mark3,ls='',label='{} K'.format(T0_X_T4[0]),ms=markersize)
if T5!=0.0:
	plt.plot(P0_X_T5,X0_X_T5,'k',marker=mark5,ls='',label='{} K'.format(T0_X_T5[0]),ms=markersize)
if T6!=0.0:
	plt.plot(P0_X_T6,X0_X_T6,'y',marker=mark6,ls='',label='{} K'.format(T0_X_T6[0]),ms=markersize)
if T7!=0.0:
	plt.plot(P0_X_T7,X0_X_T7,'r',marker=mark7,ls='',label='{} K'.format(T0_X_T7[0]),ms=markersize)
if T8!=0.0:
	plt.plot(P0_X_T8,X0_X_T8,'b',marker=mark8,ls='',label='{} K'.format(T0_X_T8[0]),ms=markersize)
if T9!=0.0:
	plt.plot(P0_X_T9,X0_X_T9,'g',marker=mark1,ls='',label='{} K'.format(T0_X_T9[0]),ms=markersize)
						
if T1!=0.0:
	plt.plot(P1,Xs_T1_DHV,'k',ls=ls1,label='Present theory {} K'.format(T1),lw=linewidth)
if T2!=0.0:
	plt.plot(P2,Xs_T2_DHV,'k',ls=ls1,label='Present theory {} K'.format(T2),lw=linewidth)
if T3!=0.0:
	plt.plot(P3,Xs_T3_DHV,'k',ls=ls2,label='{} K'.format(T3),lw=linewidth)
if T4!=0.0:
	plt.plot(P4,Xs_T4_DHV,'k',ls=ls3,label='{} K'.format(T4),lw=linewidth)
if T5!=0.0:
	plt.plot(P5,Xs_T5_DHV,'k',ls=ls5,label='{} K'.format(T5),lw=linewidth)
if T6!=0.0:
	plt.plot(P6,Xs_T6_DHV,'y',ls=ls3,label='{} K'.format(T6),lw=linewidth)
if T7!=0.0:
	plt.plot(P7,Xs_T7_DHV,'r',ls=ls3,label='{} K'.format(T7),lw=linewidth)
if T8!=0.0:
	plt.plot(P8,Xs_T8_DHV,'b',ls=ls3,label='{} K'.format(T8),lw=linewidth)
if T9!=0.0:
	plt.plot(P9,Xs_T9_DHV,'g',ls=ls3,label='{} K'.format(T9),lw=linewidth)

plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
plt.ylabel('Gas solubility $\chi_s$ (g/g)',fontsize=axis_size)
plt.legend(loc=2,fontsize=size,numpoints=1,frameon=False)
# plt.title(kwargs, fontdict=None, loc='center', pad=None)
# plt.axis(solubility_axes)
plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.10,wspace=0.30,hspace=0.25)
figX.savefig('./'+output_folder+r'\PMMA_CO2_Self_Grassia_02kilo_POST_THESIS_Paper15_Solubility_new'+img_extension,dpi=240)

#Plotting the swelling of the PMMA+CO2 mixture.
figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
if T1!=0.0:
	plt.plot(P0_S_T1,S0_S_T1,'k',marker=mark1,ls='',label='Experiment at {} K'.format(T0_S_T1[0]),ms=markersize)
if T2!=0.0:
	plt.plot(P0_S_T2,S0_S_T2,'k',marker=mark3,ls='',label='{} K'.format(T0_S_T2[0]),ms=markersize)
if T3!=0.0:
	plt.plot(P0_S_T3,S0_S_T3,'k',marker=mark4,ls='',label='{} K'.format(T0_S_T3[0]),ms=markersize)
if T4!=0.0:
	plt.plot(P0_S_T4,S0_S_T4,'k',marker=mark5,ls='',label='{} K'.format(T0_S_T4[0]),ms=markersize)
# if T5!=0.0:
# 	plt.plot(P0_S_T5,S0_S_T5,'k',marker=mark5,ls='',label='{} K'.format(T0_S_T5[0]),ms=markersize)
if T6!=0.0:
	plt.plot(P0_S_T6,S0_S_T6,'y',marker=mark6,ls='',label='{} K'.format(T0_S_T6[0]),ms=markersize)
if T7!=0.0:
	plt.plot(P0_S_T7,S0_S_T7,'r',marker=mark7,ls='',label='{} K'.format(T0_S_T7[0]),ms=markersize)
if T8!=0.0:
	plt.plot(P0_S_T8,S0_S_T8,'b',marker=mark8,ls='',label='{} K'.format(T0_S_T8[0]),ms=markersize)
if T9!=0.0:
	plt.plot(P0_S_T9,S0_S_T9,'g',marker=mark4,ls='',label='{} K'.format(T0_S_T9[0]),ms=markersize)

if T1!=0.0:
	plt.plot(P1,Sw_T1_DHV,'k',ls=ls1,label='Present theory {} K'.format(T1),lw=linewidth)
if T2!=0.0:
	plt.plot(P2,Sw_T2_DHV,'k',ls=ls2,label='{} K'.format(T2),lw=linewidth)
if T3!=0.0:
	plt.plot(P3,Sw_T3_DHV,'k',ls=ls3,label='{} K'.format(T3),lw=linewidth)
if T4!=0.0:
	plt.plot(P4,Sw_T4_DHV,'k',ls=ls4,label='{} K'.format(T4),lw=linewidth)
if T5!=0.0:
	plt.plot(P5,Sw_T5_DHV,'k',ls=ls5,label='{} K'.format(T5),lw=linewidth)
if T6!=0.0:
	plt.plot(P6,Sw_T6_DHV,'y',ls=ls3,label='{} K'.format(T6),lw=linewidth)
if T7!=0.0:
	plt.plot(P7,Sw_T7_DHV,'r',ls=ls3,label='{} K'.format(T7),lw=linewidth)
if T8!=0.0:
	plt.plot(P8,Sw_T8_DHV,'b',ls=ls3,label='{} K'.format(T8),lw=linewidth)
if T9!=0.0:
	plt.plot(P9,Sw_T9_DHV,'g',ls=ls3,label='{} K'.format(T9),lw=linewidth)

plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
plt.ylabel('Swelling ratio $S_W$ $(cm^3/cm^3)$',fontsize=axis_size)
plt.legend(loc=2,fontsize=size,numpoints=1)
# plt.title(kwargs, fontdict=None, loc='center', pad=None)
# plt.axis(swelling_axes)
plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.10,wspace=0.30,hspace=0.25)
# figS.savefig('./'+output_folder+r'\PS_CO2_Self_Grassia_02kilo_POST_THESIS_Paper4_11_12_Swelling_new'+img_extension,dpi=240)

#Show plot windows.
plt.show()