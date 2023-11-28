import os,sys,math,csv,numpy as npy
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from loadData import loadPVTData

# 311K_Rho_Hariharan1993.csv
# 320K_Rho_Duschek1990.csv
# 323K_Rho_Garg1994.csv
# 340K_Rho_Angus1976.csv
# 340K_Rho_Duschek1990.csv
# 355K_Rho_Gokmenoglu1996.csv
# 373K_Rho_Garg1994.csv
# 380K_Rho_Angus1976.csv
# 380K_Rho_Xiong1995_Bl.csv
# 400K_Rho_Xiong1995_Bl.csv
# 410K_Rho_Angus1976.csv
# 420K_Rho_Xiong1995_Bl.csv
# 430K_Rho_Angus1976.csv
# 490K_Rho_Angus1976.csv
# 520K_Rho_Angus1976.csv
# 660K_Rho_Angus1976.csv
# 1100K_Rho_Angus1976.csv
# 

#======================================================
#Critical Point and Molecular Weight
#======================================================

Pc0 = 7.38
Tc0 = 304.25
Rc0 = 0.468

#======================================================
#Isotherm PVT Data
#======================================================

with open('DataCO2/Isotherms/220K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_220 = range(0,numpts)
			T0_220 = range(0,numpts)
			R0_220 = range(0,numpts)
			M0_220 = range(0,numpts)
			I0_220 = range(0,numpts)
		if index1 >= 6:
			P0_220[index2] = float(row[0])
			T0_220[index2] = float(row[1])
			R0_220[index2] = float(row[2])
			M0_220[index2] = float(row[3])
			I0_220[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate(([],P0_220),axis=0)
T0_isotherms = npy.concatenate(([],T0_220),axis=0)
R0_isotherms = npy.concatenate(([],R0_220),axis=0)
M0_isotherms = npy.concatenate(([],M0_220),axis=0)
I0_isotherms = npy.concatenate(([],I0_220),axis=0)

with open('DataCO2/Isotherms/240K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_240 = range(0,numpts)
			T0_240 = range(0,numpts)
			R0_240 = range(0,numpts)
			M0_240 = range(0,numpts)
			I0_240 = range(0,numpts)
		if index1 >= 6:
			P0_240[index2] = float(row[0])
			T0_240[index2] = float(row[1])
			R0_240[index2] = float(row[2])
			M0_240[index2] = float(row[3])
			I0_240[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_240),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_240),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_240),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_240),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_240),axis=0)

with open('DataCO2/Isotherms/260K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_260 = range(0,numpts)
			T0_260 = range(0,numpts)
			R0_260 = range(0,numpts)
			M0_260 = range(0,numpts)
			I0_260 = range(0,numpts)
		if index1 >= 6:
			P0_260[index2] = float(row[0])
			T0_260[index2] = float(row[1])
			R0_260[index2] = float(row[2])
			M0_260[index2] = float(row[3])
			I0_260[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_260),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_260),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_260),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_260),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_260),axis=0)

with open('DataCO2/Isotherms/273K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_273 = range(0,numpts)
			T0_273 = range(0,numpts)
			R0_273 = range(0,numpts)
			M0_273 = range(0,numpts)
			I0_273 = range(0,numpts)
		if index1 >= 6:
			P0_273[index2] = float(row[0])
			T0_273[index2] = float(row[1])
			R0_273[index2] = float(row[2])
			M0_273[index2] = float(row[3])
			I0_273[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_273),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_273),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_273),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_273),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_273),axis=0)

with open('DataCO2/Isotherms/280K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_280 = range(0,numpts)
			T0_280 = range(0,numpts)
			R0_280 = range(0,numpts)
			M0_280 = range(0,numpts)
			I0_280 = range(0,numpts)
		if index1 >= 6:
			P0_280[index2] = float(row[0])
			T0_280[index2] = float(row[1])
			R0_280[index2] = float(row[2])
			M0_280[index2] = float(row[3])
			I0_280[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_280),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_280),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_280),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_280),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_280),axis=0)

with open('DataCO2/Isotherms/283K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_283 = range(0,numpts)
			T0_283 = range(0,numpts)
			R0_283 = range(0,numpts)
			M0_283 = range(0,numpts)
			I0_283 = range(0,numpts)
		if index1 >= 6:
			P0_283[index2] = float(row[0])
			T0_283[index2] = float(row[1])
			R0_283[index2] = float(row[2])
			M0_283[index2] = float(row[3])
			I0_283[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_283),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_283),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_283),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_283),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_283),axis=0)

with open('DataCO2/Isotherms/290K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_290 = range(0,numpts)
			T0_290 = range(0,numpts)
			R0_290 = range(0,numpts)
			M0_290 = range(0,numpts)
			I0_290 = range(0,numpts)
		if index1 >= 6:
			P0_290[index2] = float(row[0])
			T0_290[index2] = float(row[1])
			R0_290[index2] = float(row[2])
			M0_290[index2] = float(row[3])
			I0_290[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_290),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_290),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_290),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_290),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_290),axis=0)

with open('DataCO2/Isotherms/293K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_293 = range(0,numpts)
			T0_293 = range(0,numpts)
			R0_293 = range(0,numpts)
			M0_293 = range(0,numpts)
			I0_293 = range(0,numpts)
		if index1 >= 6:
			P0_293[index2] = float(row[0])
			T0_293[index2] = float(row[1])
			R0_293[index2] = float(row[2])
			M0_293[index2] = float(row[3])
			I0_293[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_293),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_293),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_293),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_293),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_293),axis=0)

with open('DataCO2/Isotherms/300K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_300 = range(0,numpts)
			T0_300 = range(0,numpts)
			R0_300 = range(0,numpts)
			M0_300 = range(0,numpts)
			I0_300 = range(0,numpts)
		if index1 >= 6:
			P0_300[index2] = float(row[0])
			T0_300[index2] = float(row[1])
			R0_300[index2] = float(row[2])
			M0_300[index2] = float(row[3])
			I0_300[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_300),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_300),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_300),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_300),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_300),axis=0)

with open('DataCO2/Isotherms/303K_Rho_Duschek1990_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_303 = range(0,numpts)
			T0_303 = range(0,numpts)
			R0_303 = range(0,numpts)
			M0_303 = range(0,numpts)
			I0_303 = range(0,numpts)
		if index1 >= 6:
			P0_303[index2] = float(row[0])
			T0_303[index2] = float(row[1])
			R0_303[index2] = float(row[2])
			M0_303[index2] = float(row[3])
			I0_303[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_303),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_303),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_303),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_303),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_303),axis=0)

with open('DataCO2/Isotherms/305K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_305 = range(0,numpts)
			T0_305 = range(0,numpts)
			R0_305 = range(0,numpts)
			M0_305 = range(0,numpts)
			I0_305 = range(0,numpts)
		if index1 >= 6:
			P0_305[index2] = float(row[0])
			T0_305[index2] = float(row[1])
			R0_305[index2] = float(row[2])
			M0_305[index2] = float(row[3])
			I0_305[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_305),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_305),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_305),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_305),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_305),axis=0)

with open('DataCO2/Isotherms/307K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_307 = range(0,numpts)
			T0_307 = range(0,numpts)
			R0_307 = range(0,numpts)
			M0_307 = range(0,numpts)
			I0_307 = range(0,numpts)
		if index1 >= 6:
			P0_307[index2] = float(row[0])
			T0_307[index2] = float(row[1])
			R0_307[index2] = float(row[2])
			M0_307[index2] = float(row[3])
			I0_307[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_307),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_307),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_307),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_307),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_307),axis=0)

with open('DataCO2/Isotherms/310K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_310 = range(0,numpts)
			T0_310 = range(0,numpts)
			R0_310 = range(0,numpts)
			M0_310 = range(0,numpts)
			I0_310 = range(0,numpts)
		if index1 >= 6:
			P0_310[index2] = float(row[0])
			T0_310[index2] = float(row[1])
			R0_310[index2] = float(row[2])
			M0_310[index2] = float(row[3])
			I0_310[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_310),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_310),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_310),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_310),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_310),axis=0)

with open('DataCO2/Isotherms/311K_Rho_Hariharan1993.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_311 = range(0,numpts)
			T0_311 = range(0,numpts)
			R0_311 = range(0,numpts)
			M0_311 = range(0,numpts)
			I0_311 = range(0,numpts)
		if index1 >= 6:
			P0_311[index2] = float(row[0])
			T0_311[index2] = float(row[1])
			R0_311[index2] = float(row[2])
			M0_311[index2] = float(row[3])
			I0_311[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_311),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_311),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_311),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_311),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_311),axis=0)

with open('DataCO2/Isotherms/313K_Rho_Duschek1990_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_313 = range(0,numpts)
			T0_313 = range(0,numpts)
			R0_313 = range(0,numpts)
			M0_313 = range(0,numpts)
			I0_313 = range(0,numpts)
		if index1 >= 6:
			P0_313[index2] = float(row[0])
			T0_313[index2] = float(row[1])
			R0_313[index2] = float(row[2])
			M0_313[index2] = float(row[3])
			I0_313[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_313),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_313),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_313),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_313),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_313),axis=0)

with open('DataCO2/Isotherms/320K_Rho_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_320 = range(0,numpts)
			T0_320 = range(0,numpts)
			R0_320 = range(0,numpts)
			M0_320 = range(0,numpts)
			I0_320 = range(0,numpts)
		if index1 >= 6:
			P0_320[index2] = float(row[0])
			T0_320[index2] = float(row[1])
			R0_320[index2] = float(row[2])
			M0_320[index2] = float(row[3])
			I0_320[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_320),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_320),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_320),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_320),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_320),axis=0)


P0_323,T0_323,R0_323,M0_323,I0_323 = loadPVTData('DataCO2/Isotherms/323K_Rho_Garg1994_Ohio.csv','isotherm')
P0_isotherms = npy.concatenate((P0_isotherms,P0_323),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_323),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_323),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_323),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_323),axis=0)

P0_333,T0_333,R0_333,M0_333,I0_333 = loadPVTData('DataCO2/Isotherms/333K_Rho_Ohio.csv','isotherm')
P0_isotherms = npy.concatenate((P0_isotherms,P0_333),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_333),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_333),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_333),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_333),axis=0)

with open('DataCO2/Isotherms/340K_Rho_Angus1976_Duschek1990.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_340A = range(0,numpts)
			T0_340A = range(0,numpts)
			R0_340A = range(0,numpts)
			M0_340A = range(0,numpts)
			I0_340A = range(0,numpts)
		if index1 >= 6:
			P0_340A[index2] = float(row[0])
			T0_340A[index2] = float(row[1])
			R0_340A[index2] = float(row[2])
			M0_340A[index2] = float(row[3])
			I0_340A[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_340A),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_340A),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_340A),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_340A),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_340A),axis=0)

with open('DataCO2/Isotherms/343K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_343A = range(0,numpts)
			T0_343A = range(0,numpts)
			R0_343A = range(0,numpts)
			M0_343A = range(0,numpts)
			I0_343A = range(0,numpts)
		if index1 >= 6:
			P0_343A[index2] = float(row[0])
			T0_343A[index2] = float(row[1])
			R0_343A[index2] = float(row[2])
			M0_343A[index2] = float(row[3])
			I0_343A[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_343A),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_343A),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_343A),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_343A),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_343A),axis=0)

with open('DataCO2/Isotherms/353K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_353A = range(0,numpts)
			T0_353A = range(0,numpts)
			R0_353A = range(0,numpts)
			M0_353A = range(0,numpts)
			I0_353A = range(0,numpts)
		if index1 >= 6:
			P0_353A[index2] = float(row[0])
			T0_353A[index2] = float(row[1])
			R0_353A[index2] = float(row[2])
			M0_353A[index2] = float(row[3])
			I0_353A[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_353A),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_353A),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_353A),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_353A),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_353A),axis=0)

with open('DataCO2/Isotherms/355K_Rho_Gokmenoglu1996.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_355 = range(0,numpts)
			T0_355 = range(0,numpts)
			R0_355 = range(0,numpts)
			M0_355 = range(0,numpts)
			I0_355 = range(0,numpts)
		if index1 >= 6:
			P0_355[index2] = float(row[0])
			T0_355[index2] = float(row[1])
			R0_355[index2] = float(row[2])
			M0_355[index2] = float(row[3])
			I0_355[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_355),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_355),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_355),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_355),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_355),axis=0)

with open('DataCO2/Isotherms/363K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_363 = range(0,numpts)
			T0_363 = range(0,numpts)
			R0_363 = range(0,numpts)
			M0_363 = range(0,numpts)
			I0_363 = range(0,numpts)
		if index1 >= 6:
			P0_363[index2] = float(row[0])
			T0_363[index2] = float(row[1])
			R0_363[index2] = float(row[2])
			M0_363[index2] = float(row[3])
			I0_363[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_363),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_363),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_363),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_363),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_363),axis=0)

with open('DataCO2/Isotherms/364K_Rho_Gokmenoglu1996.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_364 = range(0,numpts)
			T0_364 = range(0,numpts)
			R0_364 = range(0,numpts)
			M0_364 = range(0,numpts)
			I0_364 = range(0,numpts)
		if index1 >= 6:
			P0_364[index2] = float(row[0])
			T0_364[index2] = float(row[1])
			R0_364[index2] = float(row[2])
			M0_364[index2] = float(row[3])
			I0_364[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_364),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_364),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_364),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_364),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_364),axis=0)

with open('DataCO2/Isotherms/373K_Rho_Garg1994_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_373 = range(0,numpts)
			T0_373 = range(0,numpts)
			R0_373 = range(0,numpts)
			M0_373 = range(0,numpts)
			I0_373 = range(0,numpts)
		if index1 >= 6:
			P0_373[index2] = float(row[0])
			T0_373[index2] = float(row[1])
			R0_373[index2] = float(row[2])
			M0_373[index2] = float(row[3])
			I0_373[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_373),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_373),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_373),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_373),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_373),axis=0)

with open('DataCO2/Isotherms/380K_Rho_Angus1976_Xiong1995_Bl_Lit_Wh.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_380 = range(0,numpts)
			T0_380 = range(0,numpts)
			R0_380 = range(0,numpts)
			M0_380 = range(0,numpts)
			I0_380 = range(0,numpts)
		if index1 >= 6:
			P0_380[index2] = float(row[0])
			T0_380[index2] = float(row[1])
			R0_380[index2] = float(row[2])
			M0_380[index2] = float(row[3])
			I0_380[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_380),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_380),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_380),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_380),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_380),axis=0)

with open('DataCO2/Isotherms/381K_Rho_Gokmenoglu1996.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_381 = range(0,numpts)
			T0_381 = range(0,numpts)
			R0_381 = range(0,numpts)
			M0_381 = range(0,numpts)
			I0_381 = range(0,numpts)
		if index1 >= 6:
			P0_381[index2] = float(row[0])
			T0_381[index2] = float(row[1])
			R0_381[index2] = float(row[2])
			M0_381[index2] = float(row[3])
			I0_381[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_381),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_381),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_381),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_381),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_381),axis=0)

with open('DataCO2/Isotherms/383K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_383 = range(0,numpts)
			T0_383 = range(0,numpts)
			R0_383 = range(0,numpts)
			M0_383 = range(0,numpts)
			I0_383 = range(0,numpts)
		if index1 >= 6:
			P0_383[index2] = float(row[0])
			T0_383[index2] = float(row[1])
			R0_383[index2] = float(row[2])
			M0_383[index2] = float(row[3])
			I0_383[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_383),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_383),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_383),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_383),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_383),axis=0)

with open('DataCO2/Isotherms/388K_Rho_Gokmenoglu1996.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_388 = range(0,numpts)
			T0_388 = range(0,numpts)
			R0_388 = range(0,numpts)
			M0_388 = range(0,numpts)
			I0_388 = range(0,numpts)
		if index1 >= 6:
			P0_388[index2] = float(row[0])
			T0_388[index2] = float(row[1])
			R0_388[index2] = float(row[2])
			M0_388[index2] = float(row[3])
			I0_388[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_388),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_388),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_388),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_388),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_388),axis=0)

with open('DataCO2/Isotherms/393K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_393 = range(0,numpts)
			T0_393 = range(0,numpts)
			R0_393 = range(0,numpts)
			M0_393 = range(0,numpts)
			I0_393 = range(0,numpts)
		if index1 >= 6:
			P0_393[index2] = float(row[0])
			T0_393[index2] = float(row[1])
			R0_393[index2] = float(row[2])
			M0_393[index2] = float(row[3])
			I0_393[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_393),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_393),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_393),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_393),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_393),axis=0)

with open('DataCO2/Isotherms/398K_Rho_Gokmenoglu1996.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_398 = range(0,numpts)
			T0_398 = range(0,numpts)
			R0_398 = range(0,numpts)
			M0_398 = range(0,numpts)
			I0_398 = range(0,numpts)
		if index1 >= 6:
			P0_398[index2] = float(row[0])
			T0_398[index2] = float(row[1])
			R0_398[index2] = float(row[2])
			M0_398[index2] = float(row[3])
			I0_398[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_398),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_398),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_398),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_398),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_398),axis=0)

with open('DataCO2/Isotherms/400K_Rho_Xiong1995_Bl_Lit_Wh.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_400 = range(0,numpts)
			T0_400 = range(0,numpts)
			R0_400 = range(0,numpts)
			M0_400 = range(0,numpts)
			I0_400 = range(0,numpts)
		if index1 >= 6:
			P0_400[index2] = float(row[0])
			T0_400[index2] = float(row[1])
			R0_400[index2] = float(row[2])
			M0_400[index2] = float(row[3])
			I0_400[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_400),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_400),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_400),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_400),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_400),axis=0)

with open('DataCO2/Isotherms/403K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_403 = range(0,numpts)
			T0_403 = range(0,numpts)
			R0_403 = range(0,numpts)
			M0_403 = range(0,numpts)
			I0_403 = range(0,numpts)
		if index1 >= 6:
			P0_403[index2] = float(row[0])
			T0_403[index2] = float(row[1])
			R0_403[index2] = float(row[2])
			M0_403[index2] = float(row[3])
			I0_403[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_403),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_403),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_403),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_403),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_403),axis=0)

with open('DataCO2/Isotherms/406K_Rho_Gokmenoglu1996.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_406 = range(0,numpts)
			T0_406 = range(0,numpts)
			R0_406 = range(0,numpts)
			M0_406 = range(0,numpts)
			I0_406 = range(0,numpts)
		if index1 >= 6:
			P0_406[index2] = float(row[0])
			T0_406[index2] = float(row[1])
			R0_406[index2] = float(row[2])
			M0_406[index2] = float(row[3])
			I0_406[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_406),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_406),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_406),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_406),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_406),axis=0)

with open('DataCO2/Isotherms/410K_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_410 = range(0,numpts)
			T0_410 = range(0,numpts)
			R0_410 = range(0,numpts)
			M0_410 = range(0,numpts)
			I0_410 = range(0,numpts)
		if index1 >= 6:
			P0_410[index2] = float(row[0])
			T0_410[index2] = float(row[1])
			R0_410[index2] = float(row[2])
			M0_410[index2] = float(row[3])
			I0_410[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_410),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_410),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_410),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_410),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_410),axis=0)

with open('DataCO2/Isotherms/413K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_413 = range(0,numpts)
			T0_413 = range(0,numpts)
			R0_413 = range(0,numpts)
			M0_413 = range(0,numpts)
			I0_413 = range(0,numpts)
		if index1 >= 6:
			P0_413[index2] = float(row[0])
			T0_413[index2] = float(row[1])
			R0_413[index2] = float(row[2])
			M0_413[index2] = float(row[3])
			I0_413[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_413),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_413),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_413),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_413),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_413),axis=0)

with open('DataCO2/Isotherms/416K_Rho_Gokmenoglu1996.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_416 = range(0,numpts)
			T0_416 = range(0,numpts)
			R0_416 = range(0,numpts)
			M0_416 = range(0,numpts)
			I0_416 = range(0,numpts)
		if index1 >= 6:
			P0_416[index2] = float(row[0])
			T0_416[index2] = float(row[1])
			R0_416[index2] = float(row[2])
			M0_416[index2] = float(row[3])
			I0_416[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_416),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_416),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_416),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_416),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_416),axis=0)

with open('DataCO2/Isotherms/420K_Rho_Xiong1995_Bl_Lit_Wh.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_420 = range(0,numpts)
			T0_420 = range(0,numpts)
			R0_420 = range(0,numpts)
			M0_420 = range(0,numpts)
			I0_420 = range(0,numpts)
		if index1 >= 6:
			P0_420[index2] = float(row[0])
			T0_420[index2] = float(row[1])
			R0_420[index2] = float(row[2])
			M0_420[index2] = float(row[3])
			I0_420[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_420),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_420),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_420),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_420),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_420),axis=0)

with open('DataCO2/Isotherms/423K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_423 = range(0,numpts)
			T0_423 = range(0,numpts)
			R0_423 = range(0,numpts)
			M0_423 = range(0,numpts)
			I0_423 = range(0,numpts)
		if index1 >= 6:
			P0_423[index2] = float(row[0])
			T0_423[index2] = float(row[1])
			R0_423[index2] = float(row[2])
			M0_423[index2] = float(row[3])
			I0_423[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_423),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_423),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_423),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_423),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_423),axis=0)

with open('DataCO2/Isotherms/425K_Rho_Gokmenoglu1996.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_425 = range(0,numpts)
			T0_425 = range(0,numpts)
			R0_425 = range(0,numpts)
			M0_425 = range(0,numpts)
			I0_425 = range(0,numpts)
		if index1 >= 6:
			P0_425[index2] = float(row[0])
			T0_425[index2] = float(row[1])
			R0_425[index2] = float(row[2])
			M0_425[index2] = float(row[3])
			I0_425[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_425),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_425),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_425),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_425),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_425),axis=0)

with open('DataCO2/Isotherms/430K_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_430 = range(0,numpts)
			T0_430 = range(0,numpts)
			R0_430 = range(0,numpts)
			M0_430 = range(0,numpts)
			B0_430 = range(0,numpts)
			I0_430 = range(0,numpts)
		if index1 >= 6:
			P0_430[index2] = float(row[0])
			T0_430[index2] = float(row[1])
			R0_430[index2] = float(row[2])
			M0_430[index2] = float(row[3])
			B0_430[index2] = float(row[4])
			I0_430[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_430),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_430),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_430),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_430),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_430),axis=0)

with open('DataCO2/Isotherms/443K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_443 = range(0,numpts)
			T0_443 = range(0,numpts)
			R0_443 = range(0,numpts)
			M0_443 = range(0,numpts)
			I0_443 = range(0,numpts)
		if index1 >= 6:
			P0_443[index2] = float(row[0])
			T0_443[index2] = float(row[1])
			R0_443[index2] = float(row[2])
			M0_443[index2] = float(row[3])
			I0_443[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_443),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_443),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_443),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_443),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_443),axis=0)

with open('DataCO2/Isotherms/453K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_453 = range(0,numpts)
			T0_453 = range(0,numpts)
			R0_453 = range(0,numpts)
			M0_453 = range(0,numpts)
			I0_453 = range(0,numpts)
		if index1 >= 6:
			P0_453[index2] = float(row[0])
			T0_453[index2] = float(row[1])
			R0_453[index2] = float(row[2])
			M0_453[index2] = float(row[3])
			I0_453[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_453),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_453),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_453),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_453),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_453),axis=0)

with open('DataCO2/Isotherms/463K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_463 = range(0,numpts)
			T0_463 = range(0,numpts)
			R0_463 = range(0,numpts)
			M0_463 = range(0,numpts)
			I0_463 = range(0,numpts)
		if index1 >= 6:
			P0_463[index2] = float(row[0])
			T0_463[index2] = float(row[1])
			R0_463[index2] = float(row[2])
			M0_463[index2] = float(row[3])
			I0_463[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_463),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_463),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_463),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_463),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_463),axis=0)

with open('DataCO2/Isotherms/473K_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_473 = range(0,numpts)
			T0_473 = range(0,numpts)
			R0_473 = range(0,numpts)
			M0_473 = range(0,numpts)
			I0_473 = range(0,numpts)
		if index1 >= 6:
			P0_473[index2] = float(row[0])
			T0_473[index2] = float(row[1])
			R0_473[index2] = float(row[2])
			M0_473[index2] = float(row[3])
			I0_473[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_473),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_473),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_473),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_473),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_473),axis=0)

with open('DataCO2/Isotherms/490K_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_490 = range(0,numpts)
			T0_490 = range(0,numpts)
			R0_490 = range(0,numpts)
			M0_490 = range(0,numpts)
			B0_490 = range(0,numpts)
			I0_490 = range(0,numpts)
		if index1 >= 6:
			P0_490[index2] = float(row[0])
			T0_490[index2] = float(row[1])
			R0_490[index2] = float(row[2])
			M0_490[index2] = float(row[3])
			B0_490[index2] = float(row[4])
			I0_490[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_490),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_490),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_490),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_490),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_490),axis=0)

with open('DataCO2/Isotherms/520K_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_520 = range(0,numpts)
			T0_520 = range(0,numpts)
			R0_520 = range(0,numpts)
			M0_520 = range(0,numpts)
			B0_520 = range(0,numpts)
			I0_520 = range(0,numpts)
		if index1 >= 6:
			P0_520[index2] = float(row[0])
			T0_520[index2] = float(row[1])
			R0_520[index2] = float(row[2])
			M0_520[index2] = float(row[3])
			B0_520[index2] = float(row[4])
			I0_520[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_520),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_520),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_520),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_520),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_520),axis=0)

with open('DataCO2/Isotherms/660K_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_660 = range(0,numpts)
			T0_660 = range(0,numpts)
			R0_660 = range(0,numpts)
			M0_660 = range(0,numpts)
			B0_660 = range(0,numpts)
			I0_660 = range(0,numpts)
		if index1 >= 6:
			P0_660[index2] = float(row[0])
			T0_660[index2] = float(row[1])
			R0_660[index2] = float(row[2])
			M0_660[index2] = float(row[3])
			B0_660[index2] = float(row[4])
			I0_660[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_660),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_660),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_660),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_660),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_660),axis=0)

with open('DataCO2/Isotherms/1100K_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_1100 = range(0,numpts)
			T0_1100 = range(0,numpts)
			R0_1100 = range(0,numpts)
			M0_1100 = range(0,numpts)
			B0_1100 = range(0,numpts)
			I0_1100 = range(0,numpts)
		if index1 >= 6:
			P0_1100[index2] = float(row[0])
			T0_1100[index2] = float(row[1])
			R0_1100[index2] = float(row[2])
			M0_1100[index2] = float(row[3])
			B0_1100[index2] = float(row[4])
			I0_1100[index2] = 'isotherm'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isotherms = npy.concatenate((P0_isotherms,P0_1100),axis=0)
T0_isotherms = npy.concatenate((T0_isotherms,T0_1100),axis=0)
R0_isotherms = npy.concatenate((R0_isotherms,R0_1100),axis=0)
M0_isotherms = npy.concatenate((M0_isotherms,M0_1100),axis=0)
I0_isotherms = npy.concatenate((I0_isotherms,I0_1100),axis=0)

#======================================================
#Isobar PVT Data
#======================================================

with open('DataCO2/Isobars/Hassan_Best_CO2_EOS_Span1994.csv','rb') as csvfile:
	#CO2 EOS Fit Huge and Complete List of Data (BEST: Hassan)
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0 = range(0,numpts)
			T0 = range(0,numpts)
			R0 = range(0,numpts)
			M0 = range(0,numpts)
			I0 = range(0,numpts)
		if index1 >= 6:
			P0[index2] = float(row[0])
			T0[index2] = float(row[1])
			R0[index2] = float(row[2])
			M0[index2] = float(row[3])
			I0[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1
P0_isobars = npy.concatenate(([],P0),axis=0)
T0_isobars = npy.concatenate(([],T0),axis=0)
R0_isobars = npy.concatenate(([],R0),axis=0)
M0_isobars = npy.concatenate(([],M0),axis=0)
I0_isobars = npy.concatenate(([],I0),axis=0)

'''
with open('DataCO2/Isobars/0,5MPa_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_0pt5MPa = range(0,numpts)
			T0_0pt5MPa = range(0,numpts)
			R0_0pt5MPa = range(0,numpts)
			M0_0pt5MPa = range(0,numpts)
			B0_0pt5MPa = range(0,numpts)
			I0_0pt5MPa = range(0,numpts)
		if index1 >= 6:
			P0_0pt5MPa[index2] = float(row[0])
			T0_0pt5MPa[index2] = float(row[1])
			R0_0pt5MPa[index2] = float(row[2])
			M0_0pt5MPa[index2] = float(row[3])
			B0_0pt5MPa[index2] = float(row[4])
			I0_0pt5MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate(([],P0_0pt5MPa),axis=0)
T0_isobars = npy.concatenate(([],T0_0pt5MPa),axis=0)
R0_isobars = npy.concatenate(([],R0_0pt5MPa),axis=0)
M0_isobars = npy.concatenate(([],M0_0pt5MPa),axis=0)
I0_isobars = npy.concatenate(([],I0_0pt5MPa),axis=0)

with open('DataCO2/Isobars/1MPa_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_1MPa = range(0,numpts)
			T0_1MPa = range(0,numpts)
			R0_1MPa = range(0,numpts)
			M0_1MPa = range(0,numpts)
			I0_1MPa = range(0,numpts)
		if index1 >= 6:
			P0_1MPa[index2] = float(row[0])
			T0_1MPa[index2] = float(row[1])
			R0_1MPa[index2] = float(row[2])
			M0_1MPa[index2] = float(row[3])
			I0_1MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_1MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_1MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_1MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_1MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_1MPa),axis=0)

with open('DataCO2/Isobars/2MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_2MPa = range(0,numpts)
			T0_2MPa = range(0,numpts)
			R0_2MPa = range(0,numpts)
			M0_2MPa = range(0,numpts)
			I0_2MPa = range(0,numpts)
		if index1 >= 6:
			P0_2MPa[index2] = float(row[0])
			T0_2MPa[index2] = float(row[1])
			R0_2MPa[index2] = float(row[2])
			M0_2MPa[index2] = float(row[3])
			I0_2MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_2MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_2MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_2MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_2MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_2MPa),axis=0)

with open('DataCO2/Isobars/3MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_3MPa = range(0,numpts)
			T0_3MPa = range(0,numpts)
			R0_3MPa = range(0,numpts)
			M0_3MPa = range(0,numpts)
			I0_3MPa = range(0,numpts)
		if index1 >= 6:
			P0_3MPa[index2] = float(row[0])
			T0_3MPa[index2] = float(row[1])
			R0_3MPa[index2] = float(row[2])
			M0_3MPa[index2] = float(row[3])
			I0_3MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_3MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_3MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_3MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_3MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_3MPa),axis=0)

with open('DataCO2/Isobars/4MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_4MPa = range(0,numpts)
			T0_4MPa = range(0,numpts)
			R0_4MPa = range(0,numpts)
			M0_4MPa = range(0,numpts)
			I0_4MPa = range(0,numpts)
		if index1 >= 6:
			P0_4MPa[index2] = float(row[0])
			T0_4MPa[index2] = float(row[1])
			R0_4MPa[index2] = float(row[2])
			M0_4MPa[index2] = float(row[3])
			I0_4MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_4MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_4MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_4MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_4MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_4MPa),axis=0)

with open('DataCO2/Isobars/5MPa_Rho_Angus1976_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_5MPa = range(0,numpts)
			T0_5MPa = range(0,numpts)
			R0_5MPa = range(0,numpts)
			M0_5MPa = range(0,numpts)
			I0_5MPa = range(0,numpts)
		if index1 >= 6:
			P0_5MPa[index2] = float(row[0])
			T0_5MPa[index2] = float(row[1])
			R0_5MPa[index2] = float(row[2])
			M0_5MPa[index2] = float(row[3])
			I0_5MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_5MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_5MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_5MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_5MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_5MPa),axis=0)

with open('DataCO2/Isobars/6MPa_Rho_Vargaftik1975_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_6MPa = range(0,numpts)
			T0_6MPa = range(0,numpts)
			R0_6MPa = range(0,numpts)
			M0_6MPa = range(0,numpts)
			I0_6MPa = range(0,numpts)
		if index1 >= 6:
			P0_6MPa[index2] = float(row[0])
			T0_6MPa[index2] = float(row[1])
			R0_6MPa[index2] = float(row[2])
			M0_6MPa[index2] = float(row[3])
			I0_6MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_6MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_6MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_6MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_6MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_6MPa),axis=0)

with open('DataCO2/Isobars/7MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_7MPa = range(0,numpts)
			T0_7MPa = range(0,numpts)
			R0_7MPa = range(0,numpts)
			M0_7MPa = range(0,numpts)
			I0_7MPa = range(0,numpts)
		if index1 >= 6:
			P0_7MPa[index2] = float(row[0])
			T0_7MPa[index2] = float(row[1])
			R0_7MPa[index2] = float(row[2])
			M0_7MPa[index2] = float(row[3])
			I0_7MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_7MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_7MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_7MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_7MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_7MPa),axis=0)

with open('DataCO2/Isobars/8MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_8MPa = range(0,numpts)
			T0_8MPa = range(0,numpts)
			R0_8MPa = range(0,numpts)
			M0_8MPa = range(0,numpts)
			I0_8MPa = range(0,numpts)
		if index1 >= 6:
			P0_8MPa[index2] = float(row[0])
			T0_8MPa[index2] = float(row[1])
			R0_8MPa[index2] = float(row[2])
			M0_8MPa[index2] = float(row[3])
			I0_8MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_8MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_8MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_8MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_8MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_8MPa),axis=0)

with open('DataCO2/Isobars/9MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_9MPa = range(0,numpts)
			T0_9MPa = range(0,numpts)
			R0_9MPa = range(0,numpts)
			M0_9MPa = range(0,numpts)
			I0_9MPa = range(0,numpts)
		if index1 >= 6:
			P0_9MPa[index2] = float(row[0])
			T0_9MPa[index2] = float(row[1])
			R0_9MPa[index2] = float(row[2])
			M0_9MPa[index2] = float(row[3])
			I0_9MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_9MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_9MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_9MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_9MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_9MPa),axis=0)

with open('DataCO2/Isobars/10MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_10MPa = range(0,numpts)
			T0_10MPa = range(0,numpts)
			R0_10MPa = range(0,numpts)
			M0_10MPa = range(0,numpts)
			I0_10MPa = range(0,numpts)
		if index1 >= 6:
			P0_10MPa[index2] = float(row[0])
			T0_10MPa[index2] = float(row[1])
			R0_10MPa[index2] = float(row[2])
			M0_10MPa[index2] = float(row[3])
			I0_10MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_10MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_10MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_10MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_10MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_10MPa),axis=0)

with open('DataCO2/Isobars/11MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_11MPa = range(0,numpts)
			T0_11MPa = range(0,numpts)
			R0_11MPa = range(0,numpts)
			M0_11MPa = range(0,numpts)
			I0_11MPa = range(0,numpts)
		if index1 >= 6:
			P0_11MPa[index2] = float(row[0])
			T0_11MPa[index2] = float(row[1])
			R0_11MPa[index2] = float(row[2])
			M0_11MPa[index2] = float(row[3])
			I0_11MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_11MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_11MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_11MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_11MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_11MPa),axis=0)

with open('DataCO2/Isobars/12MPa_Rho_Vargaftik1975_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_12MPa = range(0,numpts)
			T0_12MPa = range(0,numpts)
			R0_12MPa = range(0,numpts)
			M0_12MPa = range(0,numpts)
			I0_12MPa = range(0,numpts)
		if index1 >= 6:
			P0_12MPa[index2] = float(row[0])
			T0_12MPa[index2] = float(row[1])
			R0_12MPa[index2] = float(row[2])
			M0_12MPa[index2] = float(row[3])
			I0_12MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_12MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_12MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_12MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_12MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_12MPa),axis=0)

with open('DataCO2/Isobars/13MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_13MPa = range(0,numpts)
			T0_13MPa = range(0,numpts)
			R0_13MPa = range(0,numpts)
			M0_13MPa = range(0,numpts)
			I0_13MPa = range(0,numpts)
		if index1 >= 6:
			P0_13MPa[index2] = float(row[0])
			T0_13MPa[index2] = float(row[1])
			R0_13MPa[index2] = float(row[2])
			M0_13MPa[index2] = float(row[3])
			I0_13MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_13MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_13MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_13MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_13MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_13MPa),axis=0)

with open('DataCO2/Isobars/14MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_14MPa = range(0,numpts)
			T0_14MPa = range(0,numpts)
			R0_14MPa = range(0,numpts)
			M0_14MPa = range(0,numpts)
			I0_14MPa = range(0,numpts)
		if index1 >= 6:
			P0_14MPa[index2] = float(row[0])
			T0_14MPa[index2] = float(row[1])
			R0_14MPa[index2] = float(row[2])
			M0_14MPa[index2] = float(row[3])
			I0_14MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_14MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_14MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_14MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_14MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_14MPa),axis=0)

with open('DataCO2/Isobars/15MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_15MPa = range(0,numpts)
			T0_15MPa = range(0,numpts)
			R0_15MPa = range(0,numpts)
			M0_15MPa = range(0,numpts)
			I0_15MPa = range(0,numpts)
		if index1 >= 6:
			P0_15MPa[index2] = float(row[0])
			T0_15MPa[index2] = float(row[1])
			R0_15MPa[index2] = float(row[2])
			M0_15MPa[index2] = float(row[3])
			I0_15MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_15MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_15MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_15MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_15MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_15MPa),axis=0)

with open('DataCO2/Isobars/16MPa_Rho_Vargaftik1975_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_16MPa = range(0,numpts)
			T0_16MPa = range(0,numpts)
			R0_16MPa = range(0,numpts)
			M0_16MPa = range(0,numpts)
			I0_16MPa = range(0,numpts)
		if index1 >= 6:
			P0_16MPa[index2] = float(row[0])
			T0_16MPa[index2] = float(row[1])
			R0_16MPa[index2] = float(row[2])
			M0_16MPa[index2] = float(row[3])
			I0_16MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_16MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_16MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_16MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_16MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_16MPa),axis=0)

with open('DataCO2/Isobars/17MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_17MPa = range(0,numpts)
			T0_17MPa = range(0,numpts)
			R0_17MPa = range(0,numpts)
			M0_17MPa = range(0,numpts)
			I0_17MPa = range(0,numpts)
		if index1 >= 6:
			P0_17MPa[index2] = float(row[0])
			T0_17MPa[index2] = float(row[1])
			R0_17MPa[index2] = float(row[2])
			M0_17MPa[index2] = float(row[3])
			I0_17MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_17MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_17MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_17MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_17MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_17MPa),axis=0)

with open('DataCO2/Isobars/18MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_18MPa = range(0,numpts)
			T0_18MPa = range(0,numpts)
			R0_18MPa = range(0,numpts)
			M0_18MPa = range(0,numpts)
			I0_18MPa = range(0,numpts)
		if index1 >= 6:
			P0_18MPa[index2] = float(row[0])
			T0_18MPa[index2] = float(row[1])
			R0_18MPa[index2] = float(row[2])
			M0_18MPa[index2] = float(row[3])
			I0_18MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_18MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_18MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_18MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_18MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_18MPa),axis=0)

with open('DataCO2/Isobars/19MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_19MPa = range(0,numpts)
			T0_19MPa = range(0,numpts)
			R0_19MPa = range(0,numpts)
			M0_19MPa = range(0,numpts)
			I0_19MPa = range(0,numpts)
		if index1 >= 6:
			P0_19MPa[index2] = float(row[0])
			T0_19MPa[index2] = float(row[1])
			R0_19MPa[index2] = float(row[2])
			M0_19MPa[index2] = float(row[3])
			I0_19MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_19MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_19MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_19MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_19MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_19MPa),axis=0)

with open('DataCO2/Isobars/20MPa_Rho_Ohio.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_20MPa = range(0,numpts)
			T0_20MPa = range(0,numpts)
			R0_20MPa = range(0,numpts)
			M0_20MPa = range(0,numpts)
			I0_20MPa = range(0,numpts)
		if index1 >= 6:
			P0_20MPa[index2] = float(row[0])
			T0_20MPa[index2] = float(row[1])
			R0_20MPa[index2] = float(row[2])
			M0_20MPa[index2] = float(row[3])
			I0_20MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_20MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_20MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_20MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_20MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_20MPa),axis=0)

with open('DataCO2/Isobars/30MPa_Rho_Vargaftik1975.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_30MPa = range(0,numpts)
			T0_30MPa = range(0,numpts)
			R0_30MPa = range(0,numpts)
			M0_30MPa = range(0,numpts)
			A0_30MPa = range(0,numpts)
			I0_30MPa = range(0,numpts)
		if index1 >= 6:
			P0_30MPa[index2] = float(row[0])
			T0_30MPa[index2] = float(row[1])
			R0_30MPa[index2] = float(row[2])
			M0_30MPa[index2] = float(row[3])
			A0_30MPa[index2] = float(row[4])
			I0_30MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_30MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_30MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_30MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_30MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_30MPa),axis=0)

with open('DataCO2/Isobars/50MPa_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_50MPa = range(0,numpts)
			T0_50MPa = range(0,numpts)
			R0_50MPa = range(0,numpts)
			M0_50MPa = range(0,numpts)
			A0_50MPa = range(0,numpts)
			CP_50MPa = range(0,numpts)
			S0_50MPa = range(0,numpts)
			I0_50MPa = range(0,numpts)
		if index1 >= 6:
			P0_50MPa[index2] = float(row[0])
			T0_50MPa[index2] = float(row[1])
			R0_50MPa[index2] = float(row[2])
			M0_50MPa[index2] = float(row[3])
			A0_50MPa[index2] = float(row[4])
			CP_50MPa[index2] = float(row[5])
			S0_50MPa[index2] = float(row[6])
			I0_50MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_50MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_50MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_50MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_50MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_50MPa),axis=0)

with open('DataCO2/Isobars/65MPa_Rho_Angus1976.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	#x0 = (float(column[1]) for column in datareader)
	index1 = 0
	index2 = 0
	for row in datareader:
		if index1 == 4:
			numpts = int(row[0])
			P0_65MPa = range(0,numpts)
			T0_65MPa = range(0,numpts)
			R0_65MPa = range(0,numpts)
			M0_65MPa = range(0,numpts)
			A0_65MPa = range(0,numpts)
			CP_65MPa = range(0,numpts)
			S0_65MPa = range(0,numpts)
			I0_65MPa = range(0,numpts)
		if index1 >= 6:
			P0_65MPa[index2] = float(row[0])
			T0_65MPa[index2] = float(row[1])
			R0_65MPa[index2] = float(row[2])
			M0_65MPa[index2] = float(row[3])
			A0_65MPa[index2] = float(row[4])
			CP_65MPa[index2] = float(row[5])
			S0_65MPa[index2] = float(row[6])
			I0_65MPa[index2] = 'isobar'
			index2 = index2 + 1
		index1 = index1 + 1

P0_isobars = npy.concatenate((P0_isobars,P0_65MPa),axis=0)
T0_isobars = npy.concatenate((T0_isobars,T0_65MPa),axis=0)
R0_isobars = npy.concatenate((R0_isobars,R0_65MPa),axis=0)
M0_isobars = npy.concatenate((M0_isobars,M0_65MPa),axis=0)
I0_isobars = npy.concatenate((I0_isobars,I0_65MPa),axis=0)
'''
#===
