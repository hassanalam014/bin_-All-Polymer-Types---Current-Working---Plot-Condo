# Author: Kier von Konigslow
# Date: November 2017
#
import os,sys,math,csv,matplotlib.pyplot as plt,numpy as npy
from loadExperimentalData import *
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from loadPhysicalConstants import *
from calculatePureVariables import calculateNewMolecularParameters
from checkResults import *
from sympy import *
import warnings
import cmath
from scipy.optimize import bisect,fsolve
from scipy.interpolate import interp1d
from Parameters_of_Different_Polymers import *
from All_Functions import *
# from All_Functions_brentsolve import *
from Parameters_for_Mixtures_and_Tg import *
from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
from Tait_Parameters_of_Different_Polymers import *
from loadExperimentalDataCO2 import *
from CO2PVT_interpolation import *
from Split_Exp_Data_in_Isotherms import*
###########################################################################
###########################################################################
# PMMA/Grassia/02kilo/Paper9/zeta0.98/delta0.97
Polymer_Type='PS'
Solvent='CO2'
Parameters_Paper ='Self_Grassia'			# P*T*R* and g,epsilon_2,x (PVT-Tg Data Paper or Direct P*T*R* Values Reference)
Cp_Polymer_Weight = '02kilo_POST_THESIS'	# g,epsilon_2,x (Cp Paper Reference)
Paper_Number = 'Paper4_11_12'						# Solubility or Swelling Data Reference
#for PS: 'Paper4_11_12'
#for PMMA: 'Paper15'
kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight}

Ppstar,Tpstar,Rpstar,Mp,Psstar,Tsstar,Rsstar,Ms,P_exp,Tg_exp=Parameters_of_Different_Polymers(**kwargs)
P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete=loadExperimentSwXData(**kwargs)
Far_Above_Data=False
P0_X,P0_X_above_Tg,P0_X_far_above_Tg,T0_X,T0_X_above_Tg,T0_X_far_above_Tg,X0_X,X0_X_above_Tg,X0_X_far_above_Tg,Rubber0_X,Rubber0_X_above_Tg,Rubber0_X_far_above_Tg,P0_S,P0_S_above_Tg,P0_S_far_above_Tg,T0_S,T0_S_above_Tg,T0_S_far_above_Tg,S0_S,S0_S_above_Tg,S0_S_far_above_Tg,Rubber0_S,Rubber0_S_above_Tg,Rubber0_S_far_above_Tg = SplitExperimental_X_Sw_Data(P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete,Far_Above_Data,**kwargs)
v_0,alpha,B0,B1 = Tait_Parameters_of_Different_Polymers(**kwargs)

number_of_isotherm, result = Split_Isotherms(P0_X,T0_X,X0_X,'X')
P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5,P0_X_T6,T0_X_T6,X0_X_T6,P0_X_T7,T0_X_T7,X0_X_T7,P0_X_T8,T0_X_T8,X0_X_T8 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23]
# print P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5

number_of_isotherm_swelling, result = Split_Isotherms(P0_S,T0_S,S0_S,'S')
P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5,P0_S_T6,T0_S_T6,S0_S_T6,P0_S_T7,T0_S_T7,S0_S_T7,P0_S_T8,T0_S_T8,S0_S_T8 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23]
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
#1.087, 1.115, 1.100
# zeta=	0.85456862 			#0.98	#1.11882616398		#1.068				#1.08820786			#PS zeta= 1.124 and delta=0.97423316 at 2.0 bar gives retrograde  #1.100
# delta=	0.78246675			#1.0 	#0.766707907033	#0.97423316			#0.97423316

print Psstar,Tsstar,Rsstar,Ppstar,Tpstar,Rpstar,g,epsilon_p,x,xS_infty,zeta,delta

# vhp = kB*Tpstar/(Ppstar*NA)
# vhs = kB*Tsstar/(Psstar*NA)
# vhm = delta*vhs
# print vhm

X_Exp_Data_Empirically_Corrected = False
X_Exp_Data_Experimentally_Corrected = False
plot_fraction_isotherm = False
Isotherms=False
Plot_Solubility=Isotherms

# P0_X,T0_X,X0_X,Rubber0_X = loadBinaryData('Data/Pre-Thesis Data/353K_PMMA_CO2_X.csv')
# P0_X_T1,T0_X_T1,X0_X_T1,Rubber0_X_T1 = loadBinaryData('Data/Pre-Thesis Data/353K_PMMA_CO2_X.csv')

# P0_X_P1 = npy.concatenate((P0_X_T1 , P0_X_T2 , P0_X_T3),axis=0)
# T0_X_P1 = npy.concatenate((T0_X_T1 , T0_X_T2 , T0_X_T3),axis=0)
# X0_X_P1 = npy.concatenate((X0_X_T1 , X0_X_T2 , X0_X_T3 ),axis=0)

# P0_X = P0_X_P1
# T0_X = T0_X_P1
# X0_X = X0_X_P1

Isobars=True
Entropy=True #Isobars
Plot_Phi=False #Isobars

Plot_Swelling = False		#I have not corrected Swelling Error will come.
Plot_S2=False			

forward=False
backward=True 

if Isotherms:
	# P0 = npy.linspace(6,23,10)	
	if T0_X!=[]:
		# P0 = npy.linspace(min(P0_X),max(P0_X),7)
		P0 = npy.linspace(1.0,4.0,10)
		T1=290#T0_X_T1[0]	#403	#290
		T2=0.0#T0_X_T2[0]	#423	#304
		T3=0.0#T0_X_T3[0]	#463	#350
		T4=0.0#T0_X_T4[0]	#423	#304
		T5=0.0#T0_X_T5[0]	#463	#350
		T6=0.0#T0_X_T6[0]	#423	#304
		T7=0.0#T0_X_T7[0]	#463	#350
		T8=0.0#T0_X_T8[0]	#463	#350

	if T0_S!=[] and False:
		# P0 = npy.linspace(min(P0_S),max(P0_S),5)
		# P0 = npy.linspace(0.301325,30,20)
		T1=0.0#T0_S_T1[0]	#403	#290
		T2=0.0#T0_S_T2[0]	#423	#304
		T3=0.0#T0_S_T3[0]	#463	#350
		T4=0.0#T0_S_T4[0]	#423	#304
		T5=0.0#T0_S_T5[0]	#463	#350
		T6=0.0#T0_S_T6[0]	#423	#304
		T7=0.0#T0_S_T7[0]	#463	#350
		T8=0.0#T0_S_T8[0]	#463	#350

# zeta = 1.07
# czeta = 1.0875
if Isobars:
	number_of_isobar=3
	# T0 = npy.linspace(min(T0_X),max(T0_X),20)
	T0 = npy.linspace(290,330,3)		#max: 1400000  #Small pressure ==> entropy max reaches at smaller temperature
	P1=5.0	#P0_X_P1[0]#0.101325
	P2=7.15	#30.0
	P3=9.0	#50.0

###############################################################################################
###############################################################################################

#===================================================================================
#Determining theoretical solubility/swelling.
#===================================================================================
# print('DHV mixture parameters zeta = {} and delta = {}.'.format(zeta,delta))
# gammas,vhs,epsilons = calculateNewMolecularParameters(Psstar,Tsstar,Rsstar,Ms)
# vh = delta*vhs/NA
# print('The hole volume is vh = {}.'.format(vh))
if Isotherms and T0_X!=[]:
	Pmin = min(P0_X)
	Pmax = max(P0_X)
	Tmin = min(T0_X)
	Tmax = max(T0_X)
	print('The pressure range is {}-{}MPa and the temperature range is {}-{}K.'.format(Pmin,Pmax,Tmin,Tmax))
############################################################################################################
if Kier or Hassan or Condo or Hassan_Var_Vol:
	if Isotherms:
		isotherm_label=1
		for i in range(0,number_of_isotherm):
			source='''if T%s!=0.0:
				result = calculateBinarySolubilitySwelling('CHV',P0,T%s,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				Xs_T%s_DHV = npy.array(result[2])
				Sw_T%s_DHV = npy.array(result[3])
				phip_T%s_DHV = npy.array(result[4])
				phis_T%s_DHV = npy.array(result[5])
				Rtilde_T%s_DHV = npy.array(result[6])
				phip0_T%s_DHV = npy.array(result[7])
				phis0_T%s_DHV = npy.array(result[8])

				if plot_fraction_isotherm == True:
					Xs_T%s_DHV = Xs_T%s_DHV/(1-Xs_T%s_DHV)
				
				if Entropy:
					properties=calculateThermodynamicVariables(P0,T%s,phip_T%s_DHV,phis_T%s_DHV,phip0_T%s_DHV,phis0_T%s_DHV,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
					S_1_T%s_DHV = npy.array(properties[2])
					S_2_T%s_DHV = npy.array(properties[3])'''
			
			exec source %(isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label)

			isotherm_label=isotherm_label+1

			######################################

	if X_Exp_Data_Empirically_Corrected and Isotherms and True:
		isotherm_label=1
		for i in range(0,number_of_isotherm):
			source='''if T%s == T0_X_T%s[0]:

				result = calculateR_interCO2(P0_X_T%s,T0_X_T%s,P0_isobars,T0_isobars,R0_isobars)
				Rgas_T%s_DHV_Exp = npy.array(result[0])
				result = calculateTaitVolume(P0_X_T%s,T0_X_T%s,v_0,alpha,B0,B1)
				vppure_T%s_DHV_Exp = npy.array(result[0])

				result = calculateBinarySolubilitySwelling('CHV',P0_X_T%s,T0_X_T%s,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				Xs_T%s_DHV_Exp = npy.array(result[2])
				Sw_T%s_DHV_Exp = npy.array(result[3])
				phip_T%s_DHV_Exp = npy.array(result[4])
				phis_T%s_DHV_Exp = npy.array(result[5])
				#kwargs = {'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'zeta':zeta,'delta':delta,'method':'disparate','verbose':False,'Kier':Kier,'Hassan':Hassan,'Condo':Condo,'Hassan_Var_Vol':Hassan_Var_Vol,'forward':forward,'backward':backward}
				answer= calculateCorrectSolubilityExp(P0_X_T%s,T0_X_T%s,X0_X_T%s,Rgas_T%s_DHV_Exp,vppure_T%s_DHV_Exp,Xs_T%s_DHV_Exp,Sw_T%s_DHV_Exp,phip_T%s_DHV_Exp,phis_T%s_DHV_Exp,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				X_Exp_correct_T%s_DHV_Exp = npy.array(answer[0])

				if plot_fraction_isotherm == True:
					Xs_T%s_DHV_Exp = Xs_T%s_DHV_Exp/(1-Xs_T%s_DHV_Exp)
					X_Exp_correct_T%s_DHV_Exp = X_Exp_correct_T%s_DHV_Exp/(1-X_Exp_correct_T%s_DHV_Exp)'''
			
			exec source %(isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label)
			isotherm_label=isotherm_label+1

		# print 'kwargs2 =','zeta',zeta,'delta',delta,'Ppstar',Ppstar,'Tpstar',Tpstar,'Rpstar',Rpstar,'Psstar',Psstar,'Tsstar',Tsstar,'Rsstar',Rsstar,'method','disparate','Kier',Kier,'Hassan',Hassan,'Condo',Condo,'Hassan_Var_Vol',Hassan_Var_Vol,' forward',forward,'backward',backward
		# print 'P0_X =',P0_X_T1
		# print 'T0_X =',T0_X_T1
		# print 'X0_X =',X0_X_T1
		# print 'Rgas =',Rgas_T1_DHV_Exp
		# print 'vppure =',vppure_T1_DHV_Exp
		# print 'Mp =',Mp
		# print 'Ms =',Ms
		# print 'm_s =',Xs_T1_DHV_Exp
		# print 'phip =',phip_T1_DHV_Exp
		# print 'phis =',phis_T1_DHV_Exp
		# print 'm_s_corrected =',X_Exp_correct_T1_DHV_Exp

	if Isobars:
		isobar_label=1
		for i in range(0,number_of_isobar):
			source='''if P%s!=0.0:
				result = calculateBinarySolubilitySwelling('CHV',P%s,T0,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				Xs_P%s_DHV = result[2]
				Sw_P%s_DHV = result[3]
				phip_P%s_DHV = result[4]
				phis_P%s_DHV = result[5]
				Rtilde_P%s_DHV = result[6]
				phip0_P%s_DHV = result[7]
				phis0_P%s_DHV = result[8]
				
				if Entropy:
					properties=calculateThermodynamicVariables(P%s,T0,phip_P%s_DHV,phis_P%s_DHV,phip0_P%s_DHV,phis0_P%s_DHV,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
					S_1_P%s_DHV = properties[2]
					S_2_P%s_DHV = properties[3]'''
			
			exec source %(isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label)
			isobar_label=isobar_label+1
		
			########################################

	if X_Exp_Data_Empirically_Corrected and Isobars:
		Isobar_label=1
		for i in range(0,number_of_isobar):
			source='''if P%s == P0_X_P%s[0]:

				result = calculateR_interCO2(P0_X_P%s,T0_X_P%s,P0_isobars,T0_isobars,R0_isobars)
				Rgas_P%s_DHV_Exp = npy.array(result[0])
				result = calculateTaitVolume(P0_X_P%s,T0_X_P%s,v_0,alpha,B0,B1)
				vppure_P%s_DHV_Exp = npy.array(result[0])

				result = calculateBinarySolubilitySwelling('CHV',P0_X_P%s,T0_X_P%s,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				Xs_P%s_DHV_Exp = npy.array(result[2])
				Sw_P%s_DHV_Exp = npy.array(result[3])
				phip_P%s_DHV_Exp = npy.array(result[4])
				phis_P%s_DHV_Exp = npy.array(result[5])

				#kwargs = {'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'zeta':zeta,'delta':delta,'method':'disparate','verbose':False,'Kier':Kier,'Hassan':Hassan,'Condo':Condo,'Hassan_Var_Vol':Hassan_Var_Vol,'forward':forward,'backward':backward}
				answer= calculateCorrectSolubilityExp(P0_X_P%s,T0_X_P%s,X0_X_P%s,Rgas_P%s_DHV_Exp,vppure_P%s_DHV_Exp,Xs_P%s_DHV_Exp,Sw_P%s_DHV_Exp,phip_P%s_DHV_Exp,phis_P%s_DHV_Exp,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				X_Exp_correct_P%s_DHV_Exp = answer[0]'''
		
			exec source %(Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label,Isobar_label)
			Isobar_label=Isobar_label+1

		# print 'kwargs2 =','zeta',zeta,'delta',delta,'Ppstar',Ppstar,'Tpstar',Tpstar,'Rpstar',Rpstar,'Psstar',Psstar,'Tsstar',Tsstar,'Rsstar',Rsstar,'method','disparate','Kier',Kier,'Hassan',Hassan,'Condo',Condo,'Hassan_Var_Vol',Hassan_Var_Vol,' forward',forward,'backward',backward
		# print 'P0_X =',P0_X_P1
		# print 'T0_X =',T0_X_P1
		# print 'X0_X =',X0_X_P1
		# print 'Rgas =',Rgas_P1_DHV_Exp
		# print 'vppure =',vppure_P1_DHV_Exp
		# print 'Mp =',Mp
		# print 'Ms =',Ms
		# print 'm_s =',Xs_P1_DHV_Exp
		# print 'phip =',phip_P1_DHV_Exp
		# print 'phis =',phis_P1_DHV_Exp
		# print 'm_s_corrected =',X_Exp_correct_P1_DHV_Exp

################################################

if X_Exp_Data_Experimentally_Corrected and Isotherms:
	isotherm_label=1
	for i in range(0,number_of_isotherm):
		source='''if T%s == T0_X_T%s[0]:

			result = calculateR_interCO2(P0_X_T%s,T0_X_T%s,P0_isobars,T0_isobars,R0_isobars)
			Rgas_T%s_DHV_Exp = npy.array(result[0])
			result = calculateTaitVolume(P0_X_T%s,T0_X_T%s,v_0,alpha,B0,B1)
			vppure_T%s_DHV_Exp = npy.array(result[0])

			result = calculateExpSwellingDataCorrectedSolubility(P0_X_T%s,T0_X_T%s,X0_X_T%s,S0_S_T%s,Rgas_T%s_DHV_Exp,vppure_T%s_DHV_Exp)
			X0_total_corrected_hassan_T%s = npy.array(result[0])
			X0_total_corrected_park_T%s = npy.array(result[1])

			if plot_fraction_isotherm == True:
				X0_total_corrected_park_T%s = X0_total_corrected_park_T%s/(1-X0_total_corrected_park_T%s)'''

		exec source %(isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label)
		isotherm_label=isotherm_label+1

	print 'P0_X_T1 =',P0_X_T1
	print 'T0_X_T1 =',T0_X_T1
	print 'X0_X_apparent_T1 =',X0_X_T1
	print 'S0_S_T1 =',S0_S_T1
	print 'X0_X_corrected_T1 =',X0_total_corrected_park_T1

	# print 'Rgas =',Rgas_T1_DHV_Exp
	# print 'vppure =',vppure_T1_DHV_Exp
	# print 'X0_total_corrected_hassan_T1 =',X0_total_corrected_hassan_T1
	# print 'X0_total_corrected_park_T1 =',X0_total_corrected_park_T1

	print 'P0_X_T2 =',P0_X_T2
	print 'T0_X_T2 =',T0_X_T2
	print 'X0_X_apparent_T2 =',X0_X_T2
	print 'S0_S_T2 =',S0_S_T2
	print 'X0_X_corrected_T2 =',X0_total_corrected_park_T2

	print 'P0_X_T3 =',P0_X_T3
	print 'T0_X_T3 =',T0_X_T3
	print 'X0_X_apparent_T3 =',X0_X_T3
	print 'S0_S_T3 =',S0_S_T3
	print 'X0_X_corrected_T3 =',X0_total_corrected_park_T3

if X_Exp_Data_Experimentally_Corrected and Isobars:
	isobar_label=1
	for i in range(0,number_of_isobar):
		source='''if P%s == T0_X_P%s[0]:

			result = calculateR_interCO2(P0_X_P%s,T0_X_P%s,P0_isobars,T0_isobars,R0_isobars)
			Rgas_P%s_DHV_Exp = npy.array(result[0])
			result = calculateTaitVolume(P0_X_P%s,T0_X_P%s,v_0,alpha,B0,B1)
			vppure_P%s_DHV_Exp = npy.array(result[0])

			result = calculateExpSwellingDataCorrectedSolubility(P0_X_P%s,T0_X_P%s,X0_X_P%s,S0_S_P%s,Rgas_P%s_DHV_Exp,vppure_P%s_DHV_Exp)
			X0_total_corrected_hassan_P%s = result[0]
			X0_total_corrected_park_P%s = result[1]'''

		exec source %(isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label)
		isobar_label=isobar_label+1

	# print 'P0_X_P1 =',P0_X_P1
	# print 'T0_X_P1 =',T0_X_P1
	# print 'X0_X_P1 =',X0_X_P1
	# print 'S0_S_P1 =',S0_S_P1
	# print 'Rgas =',Rgas_P1_DHV_Exp
	# print 'vppure =',vppure_P1_DHV_Exp
	# print 'X0_total_corrected_hassan_P1 =',X0_total_corrected_hassan_P1
	# print 'X0_total_corrected_park_P1 =',X0_total_corrected_park_P1

if Isotherms and False:
	isotherm_label=1
	for i in range(0,number_of_isotherm):
		source='''if T%s == T0_X_T%s[0]:
			SSE_T%s=npy.sqrt(npy.sum(npy.power((X0_total_corrected_park_T%s-X_Exp_correct_T%s_DHV_Exp), 2))/len(X_Exp_correct_T%s_DHV_Exp))
			print 'SSE_T%s is:', SSE_T%s'''

		exec source %(isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label)
		isotherm_label=isotherm_label+1

################################################

if Condo_Original:
	if Isotherms:
		if T1!=0.0:
			result = calculateBinarySolubilitySwelling('Condo_Original',P0,T1,Mp,Ms,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,forward=forward,backward=backward)
			Xs_T1_Condo = result[2]
			Sw_T1_Condo = result[3]
			phip_T1_Condo = result[4]
			phis_T1_Condo = result[5]
			Rtilde_T1_Condo = result[6]
			phip0_T1_Condo = result[7]
			phis0_T1_Condo = result[8]
			
			if Entropy:
				properties=calculateCondoThermodynamicVariables_Original(P0,T1,phip_T1_Condo,phis_T1_Condo,phip0_T1_Condo,phis0_T1_Condo,Mp,Ms,z=cz,epsilon_s=cepsilon_s,epsilon_p=cepsilon_p,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
				S_1_T1_Condo = properties[2]
				S_2_T1_Condo = properties[3]
				######################################
		if T2!=0.0:
			result = calculateBinarySolubilitySwelling('Condo_Original',P0,T2,Mp,Ms,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,forward=forward,backward=backward)
			Xs_T2_Condo = result[2]
			Sw_T2_Condo = result[3]
			phip_T2_Condo = result[4]
			phis_T2_Condo = result[5]
			Rtilde_T2_Condo = result[6]
			phip0_T2_Condo = result[7]
			phis0_T2_Condo = result[8]
			
			if Entropy:
				properties=calculateCondoThermodynamicVariables_Original(P0,T2,phip_T2_Condo,phis_T2_Condo,phip0_T2_Condo,phis0_T2_Condo,Mp,Ms,z=cz,epsilon_s=cepsilon_s,epsilon_p=cepsilon_p,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
				S_1_T2_Condo = properties[2]
				S_2_T2_Condo = properties[3]
			######################################
		if T3!=0.0:
			result = calculateBinarySolubilitySwelling('Condo_Original',P0,T3,Mp,Ms,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,forward=forward,backward=backward)
			Xs_T3_Condo = result[2]
			Sw_T3_Condo = result[3]
			phip_T3_Condo = result[4]
			phis_T3_Condo = result[5]
			Rtilde_T3_Condo = result[6]
			phip0_T3_Condo = result[7]
			phis0_T3_Condo = result[8]
			
			if Entropy:
				properties=calculateCondoThermodynamicVariables_Original(P0,T3,phip_T3_Condo,phis_T3_Condo,phip0_T3_Condo,phis0_T3_Condo,Mp,Ms,z=cz,epsilon_s=cepsilon_s,epsilon_p=cepsilon_p,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
				S_1_T3_Condo = properties[2]
				S_2_T3_Condo = properties[3]
		#######################################

	if Isobars:

		if P1!=0.0:
			result = calculateBinarySolubilitySwelling('Condo_Original',P1,T0,Mp,Ms,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,forward=forward,backward=backward)
			Xs_P1_Condo = result[2]
			Sw_P1_Condo = result[3]
			phip_P1_Condo = result[4]
			phis_P1_Condo = result[5]
			Rtilde_P1_Condo = result[6]
			phip0_P1_Condo = result[7]
			phis0_P1_Condo = result[8]
			
			if Entropy:
				properties=calculateCondoThermodynamicVariables_Original(P1,T0,phip_P1_Condo,phis_P1_Condo,phip0_P1_Condo,phis0_P1_Condo,Mp,Ms,z=cz,epsilon_s=cepsilon_s,epsilon_p=cepsilon_p,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
				S_1_P1_Condo = properties[2]
				S_2_P1_Condo = properties[3]
			########################################
		if P2!=0.0:
			result = calculateBinarySolubilitySwelling('Condo_Original',P2,T0,Mp,Ms,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,forward=forward,backward=backward)
			Xs_P2_Condo = result[2]
			Sw_P2_Condo = result[3]
			phip_P2_Condo = result[4]
			phis_P2_Condo = result[5]
			Rtilde_P2_Condo = result[6]
			phip0_P2_Condo = result[7]
			phis0_P2_Condo = result[8]
			
			if Entropy:
				properties=calculateCondoThermodynamicVariables_Original(P2,T0,phip_P2_Condo,phis_P2_Condo,phip0_P2_Condo,phis0_P2_Condo,Mp,Ms,z=cz,epsilon_s=cepsilon_s,epsilon_p=cepsilon_p,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
				S_1_P2_Condo = properties[2]
				S_2_P2_Condo = properties[3]
			########################################
		if P3!=0.0:
			result = calculateBinarySolubilitySwelling('Condo_Original',P3,T0,Mp,Ms,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar, forward=forward,backward=backward)
			Xs_P3_Condo = result[2]
			Sw_P3_Condo = result[3]
			phip_P3_Condo = result[4]
			phis_P3_Condo = result[5]
			Rtilde_P3_Condo = result[6]
			phip0_P3_Condo = result[7]
			phis0_P3_Condo = result[8]
			
			if Entropy:
				properties=calculateCondoThermodynamicVariables_Original(P3,T0,phip_P3_Condo,phis_P3_Condo,phip0_P3_Condo,phis0_P3_Condo,Mp,Ms,z=cz,epsilon_s=cepsilon_s,epsilon_p=cepsilon_p,zeta=czeta,delta=cdelta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
				S_1_P3_Condo = properties[2]
				S_2_P3_Condo = properties[3]
			########################################

#===================================================================================
#Plotting the Poly/CO2 mixture results.
#===================================================================================

if Isotherms:
	isotherm_label=1
	for i in range(0,number_of_isotherm):
		source='''if plot_fraction_isotherm == True:
			X0_X_T%s = X0_X_T%s/(1-X0_X_T%s)'''

		exec source %(isotherm_label,isotherm_label,isotherm_label)
		isotherm_label=isotherm_label+1

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
phi_axes = [5,25,0.0,1.0]
TD_axes = [5,25,0.0,5.0]

#Markers
mark1 = 'o'
mark2 = 's'
mark3 = '>'
mark4 = '<'
mark5 = 'D'
mark6 = 'H'
mark7 = 'P'
mark8 = 'X'

#Linestyles
ls1 = '-'
ls2 = '--'
ls3 = ':'

#General line properties.
linewidth = 4
markersize = 12
if Kier or Hassan or Condo or Hassan_Var_Vol:
	if Isotherms:

		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			if T1!=0.0:
				plt.plot(P0,phip_T1_DHV,'r',ls=ls1,label='phi_p_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phip_T2_DHV,'r',ls=ls2,label='phi_p_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phip_T3_DHV,'r',ls=ls3,label='phi_p_{} K'.format(T3),lw=linewidth)
			
			
			if T1!=0.0:
				plt.plot(P0,phis_T1_DHV,'m',ls=ls1,label='phi_s_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phis_T2_DHV,'m',ls=ls2,label='phi_s_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phis_T3_DHV,'m',ls=ls3,label='phi_s_{} K'.format(T3),lw=linewidth)
			
			
			if T1!=0.0:
				plt.plot(P0,Rtilde_T1_DHV,'b',ls=ls1,label='Rtilde_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Rtilde_T2_DHV,'b',ls=ls2,label='Rtilde_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Rtilde_T3_DHV,'b',ls=ls3,label='Rtilde_{} K'.format(T3),lw=linewidth)
			
			if T1!=0.0:
				plt.plot(P0,phip0_T1_DHV,'k',ls=ls1,label='phi_p0_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phip0_T2_DHV,'k',ls=ls2,label='phi_p0_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phip0_T3_DHV,'k',ls=ls3,label='phi_p0_{} K'.format(T3),lw=linewidth)
			
			if T1!=0.0:
				plt.plot(P0,phis0_T1_DHV,'y',ls=ls1,label='phi_s0_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phis0_T2_DHV,'y',ls=ls2,label='phi_s0_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phis0_T3_DHV,'y',ls=ls3,label='phi_s0_{} K'.format(T3),lw=linewidth)

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('phi',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(phi_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Plot_Solubility:
			#Plotting the solubility of the PS+CO2 mixture.
			figX = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			if T1!=0.0:
				plt.plot(P0_X_T1,X0_X_T1,'r',marker=mark1,ls='',label='Exp Apparent X at {}K'.format(T0_X_T1[0]),ms=markersize)
			if T2!=0.0:
				plt.plot(P0_X_T2,X0_X_T2,'b',marker=mark1,ls='',label='{}K'.format(T0_X_T2[0]),ms=markersize)
			if T3!=0.0:
				plt.plot(P0_X_T3,X0_X_T3,'g',marker=mark1,ls='',label='Exp Apparent X at {}K'.format(T0_X_T3[0]),ms=markersize)
			if T4!=0.0:
				plt.plot(P0_X_T4,X0_X_T4,'m',marker=mark1,ls='',label='{}K'.format(T0_X_T4[0]),ms=markersize)
			if T5!=0.0:
				plt.plot(P0_X_T5,X0_X_T5,'k',marker=mark1,ls='',label='Exp Apparent X at {}K'.format(T0_X_T5[0]),ms=markersize)
			if T6!=0.0:
				plt.plot(P0_X_T6,X0_X_T6,'y',marker=mark1,ls='',label='{}K'.format(T0_X_T6[0]),ms=markersize)
			if T7!=0.0:
				plt.plot(P0_X_T7,X0_X_T7,'k',marker=mark1,ls='',label='{}K'.format(T0_X_T7[0]),ms=markersize)
			if T8!=0.0:
				plt.plot(P0_X_T8,X0_X_T8,'r',marker=mark1,ls='',label='{}K'.format(T0_X_T8[0]),ms=markersize)
			

			if X_Exp_Data_Empirically_Corrected and True:

				if T1!=0.0:
					plt.plot(P0_X_T1,X_Exp_correct_T1_DHV_Exp,'r',marker=mark4,ls='',label='Empir. Corrected X at {}K'.format(T0_X_T1[0]),ms=markersize)
				if T2!=0.0:
					plt.plot(P0_X_T2,X_Exp_correct_T2_DHV_Exp,'b',marker=mark4,ls='',label='{}K'.format(T0_X_T2[0]),ms=markersize)
				if T3!=0.0:
					plt.plot(P0_X_T3,X_Exp_correct_T3_DHV_Exp,'g',marker=mark4,ls='',label='Empir. Corrected X at {}K'.format(T0_X_T3[0]),ms=markersize)
				if T4!=0.0:
					plt.plot(P0_X_T4,X_Exp_correct_T4_DHV_Exp,'m',marker=mark4,ls='',label='{}K'.format(T0_X_T4[0]),ms=markersize)
				if T5!=0.0:
					plt.plot(P0_X_T5,X_Exp_correct_T5_DHV_Exp,'k',marker=mark4,ls='',label='Empir. Corrected X at {}K'.format(T0_X_T5[0]),ms=markersize)
				if T6!=0.0:
					plt.plot(P0_X_T6,X_Exp_correct_T6_DHV_Exp,'y',marker=mark4,ls='',label='{}K'.format(T0_X_T6[0]),ms=markersize)
				if T7!=0.0:
					plt.plot(P0_X_T7,X_Exp_correct_T7_DHV_Exp,'k',marker=mark4,ls='',label='{}K'.format(T0_X_T7[0]),ms=markersize)
				if T8!=0.0:
					plt.plot(P0_X_T8,X_Exp_correct_T8_DHV_Exp,'r',marker=mark4,ls='',label='{}K'.format(T0_X_T8[0]),ms=markersize)			

				if False:
					if T1!=0.0:
						plt.plot(P0_X_T1,Xs_T1_DHV_Exp,'r',marker=mark5,ls='',label='Present theory at Exp Points Only {} K'.format(T0_X_T1[0]),lw=linewidth)
					if T2!=0.0:
						plt.plot(P0_X_T2,Xs_T2_DHV_Exp,'b',marker=mark5,ls='',label='Present theory at Exp Points Only {} K'.format(T0_X_T2[0]),lw=linewidth)
					if T3!=0.0:
						plt.plot(P0_X_T3,Xs_T3_DHV_Exp,'g',marker=mark5,ls='',label='Present theory at Exp Points Only {} K'.format(T0_X_T3[0]),lw=linewidth)
					if T4!=0.0:
						plt.plot(P0_X_T4,Xs_T4_DHV_Exp,'m',marker=mark5,ls='',label='Present theory at Exp Points Only {} K'.format(T0_X_T4[0]),lw=linewidth)
					if T5!=0.0:
						plt.plot(P0_X_T5,Xs_T5_DHV_Exp,'k',marker=mark5,ls='',label='Present theory at Exp Points Only {} K'.format(T0_X_T5[0]),lw=linewidth)
					if T6!=0.0:
						plt.plot(P0_X_T6,Xs_T6_DHV_Exp,'y',marker=mark5,ls='',label='Present theory at Exp Points Only {} K'.format(T0_X_T6[0]),lw=linewidth)
					if T7!=0.0:
						plt.plot(P0_X_T7,Xs_T7_DHV_Exp,'k',marker=mark5,ls='',label='Present theory at Exp Points Only {} K'.format(T0_X_T7[0]),lw=linewidth)
					if T8!=0.0:
						plt.plot(P0_X_T8,Xs_T8_DHV_Exp,'r',marker=mark5,ls='',label='Present theory at Exp Points Only {} K'.format(T0_X_T8[0]),lw=linewidth)

			if X_Exp_Data_Experimentally_Corrected:
				
				if T1!=0.0:
						plt.plot(P0_X_T1,X0_total_corrected_park_T1,'r',marker=mark7,ls='',label='Experimentally Corrected X at {}K'.format(T0_X_T1[0]),ms=markersize)
				if T2!=0.0:
						plt.plot(P0_X_T2,X0_total_corrected_park_T2,'b',marker=mark7,ls='',label='{}K'.format(T0_X_T2[0]),ms=markersize)
				if T3!=0.0:
						plt.plot(P0_X_T3,X0_total_corrected_park_T3,'g',marker=mark7,ls='',label='Experimentally Corrected X at {}K'.format(T0_X_T3[0]),ms=markersize)
				if T4!=0.0:
						plt.plot(P0_X_T4,X0_total_corrected_park_T4,'m',marker=mark7,ls='',label='{}K'.format(T0_X_T4[0]),ms=markersize)
				if T5!=0.0:
						plt.plot(P0_X_T5,X0_total_corrected_park_T5,'k',marker=mark7,ls='',label='Experimentally Corrected X at {}K'.format(T0_X_T5[0]),ms=markersize)
				if T6!=0.0:
						plt.plot(P0_X_T6,X0_total_corrected_park_T6,'y',marker=mark7,ls='',label='{}K'.format(T0_X_T6[0]),ms=markersize)
				if T7!=0.0:
						plt.plot(P0_X_T7,X0_total_corrected_park_T7,'k',marker=mark7,ls='',label='{}K'.format(T0_X_T7[0]),ms=markersize)
				if T8!=0.0:
						plt.plot(P0_X_T8,X0_total_corrected_park_T8,'r',marker=mark7,ls='',label='{}K'.format(T0_X_T8[0]),ms=markersize)			

			#################################
			
			if T1!=0.0:
				plt.plot(P0,Xs_T1_DHV,'r',ls=ls1,label='Present theory {} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Xs_T2_DHV,'b',ls=ls2,label='{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Xs_T3_DHV,'g',ls=ls3,label='Present theory {} K'.format(T3),lw=linewidth)
			if T4!=0.0:
				plt.plot(P0,Xs_T4_DHV,'m',ls=ls3,label='{} K'.format(T4),lw=linewidth)
			if T5!=0.0:
				plt.plot(P0,Xs_T5_DHV,'k',ls=ls3,label='Present theory {} K'.format(T5),lw=linewidth)
			if T6!=0.0:
				plt.plot(P0,Xs_T6_DHV,'y',ls=ls3,label='{} K'.format(T6),lw=linewidth)
			if T7!=0.0:
				plt.plot(P0,Xs_T7_DHV,'k',ls=ls3,label='{} K'.format(T7),lw=linewidth)
			if T8!=0.0:
				plt.plot(P0,Xs_T8_DHV,'r',ls=ls3,label='{} K'.format(T8),lw=linewidth)

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Gas solubility $\chi_s$ (g/g)',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(solubility_axes)
			plt.subplots_adjust(bottom=0.3)
			# figX.savefig('./'+output_folder+r'\bin_PC_CO2_Solubility'+img_extension,dpi=240)

		if Plot_Swelling:
			#Plotting the swelling of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			if T1!=0.0:
				plt.plot(P0_S_T1,S0_S_T1,'r',marker=mark1,ls='',label='Saturated Poly/CO2 mixture at {}K'.format(T0_S_T1[0]),ms=markersize)
			if T2!=0.0:
				plt.plot(P0_S_T2,S0_S_T2,'b',marker=mark2,ls='',label='{}K'.format(T0_S_T2[0]),ms=markersize)
			if T3!=0.0:
				plt.plot(P0_S_T3,S0_S_T3,'g',marker=mark3,ls='',label='{}K'.format(T0_S_T3[0]),ms=markersize)
			if T4!=0.0:
				plt.plot(P0_S_T4,S0_S_T4,'m',marker=mark4,ls='',label='{}K'.format(T0_S_T4[0]),ms=markersize)

			if T1!=0.0:
				plt.plot(P0,Sw_T1_DHV,'r',ls=ls3,label='Present theory {}K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Sw_T2_DHV,'b',ls=ls3,label='Present theory {}K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Sw_T3_DHV,'g',ls=ls3,label='Present theory {}K'.format(T3),lw=linewidth)
			if T4!=0.0:
				plt.plot(P0,Sw_T4_DHV,'m',ls=ls3,label='Present theory {}K'.format(T4),lw=linewidth)

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Swelling ratio $S_W$ $(cm^3/cm^3)$',fontsize=axis_size)
			plt.legend(loc=2,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(swelling_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Entropy:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			
			if T1!=0.0:
				plt.plot(P0,S_1_T1_DHV,'r',ls=ls1,label='S_1_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,S_1_T2_DHV,'b',ls=ls2,label='S_1_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,S_1_T3_DHV,'g',ls=ls3,label='S_1_{} K'.format(T3),lw=linewidth)

			if Plot_S2:
				if T1!=0.0:
					plt.plot(P0,S_2_T1_DHV,'m',ls=ls1,label='S_2_{} K'.format(T1),lw=linewidth)
				if T2!=0.0:
					plt.plot(P0,S_2_T2_DHV,'m',ls=ls2,label='S_2_{} K'.format(T2),lw=linewidth)
				if T3!=0.0:
					plt.plot(P0,S_2_T3_DHV,'m',ls=ls3,label='S_2_{} K'.format(T3),lw=linewidth)

			# plt.axvline(x=378,lw=0.5,color='k', linestyle='-.')
			plt.axhline(y=0.0,lw=0.5,color='k', linestyle='-.')
			# S_max=npy.max(S_1_P1)
			# print 'S_max is:', S_max
			Tg_line=xS_infty#0.317*0.98268#0.310707*0.8708171#0.2361#2.38*0.271##0.2558 #S_max*kx #0.2361
			plt.axhline(y=Tg_line,lw=0.5,color='k', linestyle='-.')

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Entropy',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(TD_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

	if Isobars:

		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,phip_P1_DHV,'r',ls=ls1,label='phi_p_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phip_P2_DHV,'r',ls=ls2,label='phi_p_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phip_P3_DHV,'r',ls=ls3,label='phi_p_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phis_P1_DHV,'m',ls=ls1,label='phi_s_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phis_P2_DHV,'m',ls=ls2,label='phi_s_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phis_P3_DHV,'m',ls=ls3,label='phi_s_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,Rtilde_P1_DHV,'b',ls=ls1,label='Rtilde_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Rtilde_P2_DHV,'b',ls=ls2,label='Rtilde_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Rtilde_P3_DHV,'b',ls=ls3,label='Rtilde_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phip0_P1_DHV,'k',ls=ls1,label='phi_p0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phip0_P2_DHV,'k',ls=ls2,label='phi_p0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phip0_P3_DHV,'k',ls=ls3,label='phi_p0_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phis0_P1_DHV,'y',ls=ls1,label='phi_s0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phis0_P2_DHV,'y',ls=ls2,label='phi_s0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phis0_P3_DHV,'y',ls=ls3,label='phi_s0_{} MPa'.format(P3),lw=linewidth)

			plt.xlabel('Temperature T (K)',fontsize=axis_size)
			plt.ylabel('phi',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(phi_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Plot_Solubility:
			#Plotting the solubility of the PS+CO2 mixture.
			figX = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,Xs_P1_DHV,'r',ls=ls1,label='Present theory {} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Xs_P2_DHV,'k',ls=ls2,label='Present theory {} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Xs_P3_DHV,'k',ls=ls3,label='Present theory {} MPa'.format(P3),lw=linewidth)
			
			#Experimental Isobars
			plt.plot(T0_X_P1,X0_X_P1,'k',marker=mark1,ls='',label='Exp Apparent X at {}MPa'.format(P0_X_P1[0]),ms=markersize)

			if X_Exp_Data_Empirically_Corrected:
				plt.plot(T0_X_P1,X_Exp_correct_P1_DHV_Exp,'b',marker=mark1,ls='',label='Exp Corrected X at {}MPa'.format(P0_X_P1[0]),ms=markersize)
				plt.plot(T0_X_P2,Xs_P2_DHV_Exp,'r',marker=mark1,ls='',label='Present theory at Exp Points Only {} MPa'.format(P0_X_P2[0]),lw=linewidth)

			if X_Exp_Data_Experimentally_Corrected:
				# Wrong Hassan formula: plt.plot(T0_X_P1,X0_total_corrected_hassan_P1,'b',marker=mark6,ls='',label='ExpSwelling Corrected X at {}MPa -Hassan'.format(P0_X_P1[0]),ms=markersize)
				# Wrong Hassan formula: plt.plot(T0_X_P2,X0_total_corrected_hassan_P2,'r',marker=mark6,ls='',label='ExpSwelling X at {} MPa -Hassan'.format(P0_X_P2[0]),lw=linewidth)
				plt.plot(T0_X_P1,X0_total_corrected_park_P1,'b',marker=mark7,ls='',label='ExpSwelling Corrected X at {}MPa -Park'.format(P0_X_P1[0]),ms=markersize)
				plt.plot(T0_X_P2,X0_total_corrected_park_P2,'r',marker=mark7,ls='',label='ExpSwelling X at {} MPa -Park'.format(P0_X_P2[0]),lw=linewidth)

			plt.xlabel('Temperature T (K)',fontsize=axis_size)
			plt.ylabel('Gas solubility $\chi_s$ (g/g)',fontsize=axis_size)
			plt.legend(loc=1,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(solubility_axes)
			# figX.savefig('./'+output_folder+r'\bin_PS_CO2_Solubility'+img_extension,dpi=img_dpi)


		if Plot_Swelling:
			#Plotting the swelling of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			if P1!=0.0:
				plt.plot(T0,Sw_P1_DHV,'k',ls=ls1,label='Present theory {} K'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Sw_P2_DHV,'k',ls=ls2,label='Present theory {} K'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Sw_P3_DHV,'k',ls=ls3,label='Present theory {} K'.format(P3),lw=linewidth)

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Swelling ratio $S_W$ $(cm^3/cm^3)$',fontsize=axis_size)
			plt.legend(loc=2,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			plt.axis(swelling_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Entropy:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,S_1_P1_DHV,'k',ls=ls1,label='Present Theory at {} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,S_1_P2_DHV,'k',ls=ls2,label='{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,S_1_P3_DHV,'k',ls=ls3,label='{} MPa'.format(P3),lw=linewidth)

			if Plot_S2:
				
				plt.plot(T0,S_2_P1_DHV,'m',ls=ls1,label='S_2_{} MPa'.format(P1),lw=linewidth)
				if P2!=0.0:
					plt.plot(T0,S_2_P2_DHV,'m',ls=ls2,label='S_2_{} MPa'.format(P2),lw=linewidth)
				if P3!=0.0:
					plt.plot(T0,S_2_P3_DHV,'m',ls=ls3,label='S_2_{} MPa'.format(P3),lw=linewidth)
			
			# plt.axvline(x=378,lw=0.5,color='k', linestyle='-.')
			# plt.axhline(y=0.0,lw=0.5,color='k', linestyle='-.')
			# S_max=npy.max(S_1_P1)
			# print 'S_max is:', S_max
			Tg_line=xS_infty#0.317*0.98268 #0.310707*0.8708171#0.2361#2.38*0.271##0.2558 #S_max*kx #0.2361
			plt.axhline(y=Tg_line,lw=0.5,color='k', linestyle='-.')

			plt.xlabel('Temperature T (K)',fontsize=axis_size)
			plt.ylabel('Entropy $S$ $($ $J/g.K)$',fontsize=axis_size)
			plt.legend(loc=2,fontsize=size,numpoints=1,frameon=False)
			# plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(TD_axes)
			plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.10,wspace=0.30,hspace=0.25)
			figS.savefig('./'+output_folder+r'\PS_CO2_Self_Grassia_02kilo_POST_THESIS_Paper4_11_12_Entropy_final_new_new'+img_extension,dpi=240)


print T0
print S_1_P1_DHV
print S_1_P2_DHV
print S_1_P3_DHV


if Condo_Original:
	if Isotherms:

		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS_Condo = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(P0,phip_T1_Condo,'r',ls=ls1,label='phi_p_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phip_T2_Condo,'r',ls=ls2,label='phi_p_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phip_T3_Condo,'r',ls=ls3,label='phi_p_{} K'.format(T3),lw=linewidth)

			plt.plot(P0,phis_T1_Condo,'m',ls=ls1,label='phi_s_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phis_T2_Condo,'m',ls=ls2,label='phi_s_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phis_T3_Condo,'m',ls=ls3,label='phi_s_{} K'.format(T3),lw=linewidth)

			plt.plot(P0,Rtilde_T1_Condo,'b',ls=ls1,label='Rtilde_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Rtilde_T2_Condo,'b',ls=ls2,label='Rtilde_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Rtilde_T3_Condo,'b',ls=ls3,label='Rtilde_{} K'.format(T3),lw=linewidth)

			plt.plot(P0,phip0_T1_Condo,'k',ls=ls1,label='phi_p0_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phip0_T2_Condo,'k',ls=ls2,label='phi_p0_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phip0_T3_Condo,'k',ls=ls3,label='phi_p0_{} K'.format(T3),lw=linewidth)

			plt.plot(P0,phis0_T1_Condo,'y',ls=ls1,label='phi_s0_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phis0_T2_Condo,'y',ls=ls2,label='phi_s0_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phis0_T3_Condo,'y',ls=ls3,label='phi_s0_{} K'.format(T3),lw=linewidth)

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('phi',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			# plt.axis(phi_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Plot_Solubility:
			#Plotting the solubility of the PS+CO2 mixture.
			figX_Condo = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			plt.plot(P0_X_T1,X0_X_T1,'k',marker=mark1,ls='',label='Saturated mixture at {}K'.format(T0_X_T1[0]),ms=markersize)
			plt.plot(P0_X_T2,X0_X_T2,'k',marker=mark2,ls='',label='{}K'.format(T0_X_T2[0]),ms=markersize)
			plt.plot(P0_X_T3,X0_X_T3,'k',marker=mark3,ls='',label='{}K'.format(T0_X_T3[0]),ms=markersize)
			# plt.plot(P0_X_T4,X0_X_T4,'k',marker=mark4,ls='',label='{}K'.format(T0_X_T4[0]),ms=markersize)
			# plt.plot(P0_X_T5,X0_X_T5,'k',marker=mark5,ls='',label='{}K'.format(T0_X_T5[0]),ms=markersize)
			# plt.plot(P0_X_T6,X0_X_T6,'k',marker=mark6,ls='',label='{}K'.format(T0_X_T6[0]),ms=markersize)

			plt.plot(P0,Xs_T1_Condo,'k',ls=ls1,label='Present theory {} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Xs_T2_Condo,'k',ls=ls2,label='Present theory {} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Xs_T3_Condo,'k',ls=ls3,label='Present theory {} K'.format(T3),lw=linewidth)

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Gas solubility $\chi_s$ (g/g)',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			# plt.axis(solubility_axes)
			# figX.savefig('./'+output_folder+r'\bin_PS_CO2_Solubility'+img_extension,dpi=img_dpi)

		if Plot_Swelling:
			#Plotting the swelling of the PS+CO2 mixture.
			figS_Condo = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			plt.plot(P0_S_T1,S0_S_T1,'k',marker=mark1,ls='',label='Saturated mixture at {}K'.format(T0_X_T1[0]),ms=markersize)
			plt.plot(P0_S_T2,S0_S_T2,'k',marker=mark2,ls='',label='{}K'.format(T0_X_T2[0]),ms=markersize)
			plt.plot(P0_S_T3,S0_S_T3,'k',marker=mark3,ls='',label='{}K'.format(T0_X_T3[0]),ms=markersize)
			# plt.plot(P0_S_T4,S0_S_T4,'k',marker=mark1,ls='',label='{}K'.format(T0_X_T4[0]),ms=markersize)
			# plt.plot(P0_S_T5,S0_S_T5,'k',marker=mark2,ls='',label='{}K'.format(T0_X_T5[0]),ms=markersize)
			# plt.plot(P0_S_T6,S0_S_T6,'k',marker=mark3,ls='',label='{}K'.format(T0_X_T6[0]),ms=markersize)

			plt.plot(P0,Sw_T1_Condo,'k',ls=ls1,label='Present theory {} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Sw_T2_Condo,'k',ls=ls2,label='Present theory {} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Sw_T3_Condo,'k',ls=ls3,label='Present theory {} K'.format(T3),lw=linewidth)
			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Swelling ratio $S_W$ $(cm^3/cm^3)$',fontsize=axis_size)
			plt.legend(loc=2,fontsize=size,numpoints=1)
			plt.axis(swelling_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Entropy:
			#Plotting the phi's of the PS+CO2 mixture.
			figS_Condo = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(P0,S_1_T1_Condo,'r',ls=ls1,label='S_1_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,S_1_T2_Condo,'r',ls=ls2,label='S_1_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,S_1_T3_Condo,'r',ls=ls3,label='S_1_{} K'.format(T3),lw=linewidth)

			if Plot_S2:
				
				plt.plot(P0,S_2_T1_Condo,'m',ls=ls1,label='S_2_{} K'.format(T1),lw=linewidth)
				if T2!=0.0:
					plt.plot(P0,S_2_T2_Condo,'m',ls=ls2,label='S_2_{} K'.format(T2),lw=linewidth)
				if T3!=0.0:
					plt.plot(P0,S_2_T3_Condo,'m',ls=ls3,label='S_2_{} K'.format(T3),lw=linewidth)

			# plt.axvline(x=378,lw=0.5,color='k', linestyle='-.')
			plt.axhline(y=0.0,lw=0.5,color='k', linestyle='-.')
			# S_max=npy.max(S_1_P1)
			# print 'S_max is:', S_max
			Tg_line=xS_infty#0.310707*0.8708171#0.2361#2.38*0.271##0.2558 #S_max*kx #0.2361
			plt.axhline(y=Tg_line,lw=0.5,color='k', linestyle='-.')

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Entropy',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			# plt.axis(TD_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

	if Isobars:

		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS_Condo = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,phip_P1_Condo,'r',ls=ls1,label='phi_p_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phip_P2_Condo,'r',ls=ls2,label='phi_p_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phip_P3_Condo,'r',ls=ls3,label='phi_p_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phis_P1_Condo,'m',ls=ls1,label='phi_s_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phis_P2_Condo,'m',ls=ls2,label='phi_s_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phis_P3_Condo,'m',ls=ls3,label='phi_s_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,Rtilde_P1_Condo,'b',ls=ls1,label='Rtilde_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Rtilde_P2_Condo,'b',ls=ls2,label='Rtilde_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Rtilde_P3_Condo,'b',ls=ls3,label='Rtilde_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phip0_P1_Condo,'k',ls=ls1,label='phi_p0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phip0_P2_Condo,'k',ls=ls2,label='phi_p0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phip0_P3_Condo,'k',ls=ls3,label='phi_p0_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phis0_P1_Condo,'y',ls=ls1,label='phi_s0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phis0_P2_Condo,'y',ls=ls2,label='phi_s0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phis0_P3_Condo,'y',ls=ls3,label='phi_s0_{} MPa'.format(P3),lw=linewidth)

			plt.xlabel('Temperature T (K)',fontsize=axis_size)
			plt.ylabel('phi',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			# plt.axis(phi_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Plot_Solubility:
			#Plotting the solubility of the PS+CO2 mixture.
			figX_Condo = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,Xs_P1_Condo,'k',ls=ls1,label='Present theory {} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Xs_P2_Condo,'k',ls=ls2,label='Present theory {} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Xs_P3_Condo,'k',ls=ls3,label='Present theory {} MPa'.format(P3),lw=linewidth)
			plt.xlabel('Temperature T (K)',fontsize=axis_size)
			plt.ylabel('Gas solubility $\chi_s$ (g/g)',fontsize=axis_size)
			plt.legend(loc=2,fontsize=size,numpoints=1)
			# plt.axis(solubility_axes)
			# figX.savefig('./'+output_folder+r'\bin_PS_CO2_Solubility'+img_extension,dpi=img_dpi)

		if Plot_Swelling:
			#Plotting the swelling of the PS+CO2 mixture.
			figS_Condo = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			plt.plot(T0,Sw_P1_Condo,'k',ls=ls1,label='Present theory {} K'.format(T1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Sw_P2_Condo,'k',ls=ls2,label='Present theory {} K'.format(T2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Sw_P3_Condo,'k',ls=ls3,label='Present theory {} K'.format(T3),lw=linewidth)
			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Swelling ratio $S_W$ $(cm^3/cm^3)$',fontsize=axis_size)
			plt.legend(loc=2,fontsize=size,numpoints=1)
			plt.axis(swelling_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Entropy:
			#Plotting the phi's of the PS+CO2 mixture.
			figS_Condo = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,S_1_P1_Condo,'r',ls=ls1,label='S_1_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,S_1_P2_Condo,'r',ls=ls2,label='S_1_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,S_1_P3_Condo,'r',ls=ls3,label='S_1_{} MPa'.format(P3),lw=linewidth)

			if Plot_S2:
					
				plt.plot(T0,S_2_P1_Condo,'m',ls=ls1,label='S_2_{} MPa'.format(P1),lw=linewidth)
				if P2!=0.0:
					plt.plot(T0,S_2_P2_Condo,'m',ls=ls2,label='S_2_{} MPa'.format(P2),lw=linewidth)
				if P3!=0.0:
					plt.plot(T0,S_2_P3_Condo,'m',ls=ls3,label='S_2_{} MPa'.format(P3),lw=linewidth)
			
			# plt.axvline(x=378,lw=0.5,color='k', linestyle='-.')
			plt.axhline(y=0.0,lw=0.5,color='k', linestyle='-.')
			# S_max=npy.max(S_1_P1)
			# print 'S_max is:', S_max
			Tg_line=xS_infty#0.310707*0.8708171#0.2361#2.38*0.271##0.2558 #S_max*kx #0.2361
			plt.axhline(y=Tg_line,lw=0.5,color='k', linestyle='-.')

			plt.xlabel('Temperature T (K)',fontsize=axis_size)
			plt.ylabel('Entropy',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			# plt.axis(TD_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

#Show plot windows.
plt.show()