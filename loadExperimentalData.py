# Author: Kier von Konigslow
# Date: May 2017
#
# Description	: The purpose of this module is to load the experimental solubility
#				  and swelling data for the PS/CO2 binary mixture.
#

import os,sys,math,numpy as npy
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from loadData import loadBinaryData

def loadExperimentSwXData(**kwargs):

    for key,value in kwargs.items():
		exec "%s='%s'" % (key,value)

    print 'Polymer Type', Polymer_Type, 'Solvent', Solvent, 'Parameters from', Parameters_Paper, 'Paper_Number', Paper_Number

    #======================================================
    #Post-Thesis Data Below:
    #======================================================

    #If Only Solubility Data is found:
    P0_S,T0_S,S0_S = [], [], []
    #If Only Swelling Data is found:
    P0_X,T0_X,X0_X = [], [], []
    #If Pre-Thesis: So not Rubber Column:
    Rubber0_X,Rubber0_S = [], []

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper9':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_323,T0_S_323,S0_S_323,Rubber0_S_323 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper9M1/323K_PMMA_CO2_Sw_Paper9.csv')
        P0_S = P0_S_323
        T0_S = T0_S_323
        S0_S = S0_S_323
        Rubber0_S = Rubber0_S_323

        P0_S_338,T0_S_338,S0_S_338,Rubber0_S_338 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper9M1/338K_PMMA_CO2_Sw_Paper9.csv')
        P0_S = npy.concatenate((P0_S,P0_S_338),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_338),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_338),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_338),axis=0)

        P0_S_353,T0_S_353,S0_S_353,Rubber0_S_353 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper9M1/353K_PMMA_CO2_Sw_Paper9.csv')
        P0_S = npy.concatenate((P0_S,P0_S_353),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_353),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_353),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_353),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper6':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_313,T0_S_313,S0_S_313,Rubber0_S_313 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/313K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = P0_S_313
        T0_S = T0_S_313
        S0_S = S0_S_313
        Rubber0_S = Rubber0_S_313

        P0_S_333,T0_S_333,S0_S_333,Rubber0_S_333 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/333K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = npy.concatenate((P0_S,P0_S_333),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_333),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_333),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_333),axis=0)

        P0_S_353,T0_S_353,S0_S_353,Rubber0_S_353 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/353K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = npy.concatenate((P0_S,P0_S_353),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_353),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_353),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_353),axis=0)

        P0_S_373,T0_S_373,S0_S_373,Rubber0_S_373 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/373K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = npy.concatenate((P0_S,P0_S_373),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_373),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_373),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_373),axis=0)

        P0_S_393,T0_S_393,S0_S_393,Rubber0_S_393 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/393K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = npy.concatenate((P0_S,P0_S_393),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_393),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_393),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_393),axis=0)

        P0_S_413,T0_S_413,S0_S_413,Rubber0_S_413 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/413K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = npy.concatenate((P0_S,P0_S_413),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_413),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_413),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_413),axis=0)

        P0_S_433,T0_S_433,S0_S_433,Rubber0_S_433 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/433K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = npy.concatenate((P0_S,P0_S_433),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_433),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_433),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_433),axis=0)

        P0_S_453,T0_S_453,S0_S_453,Rubber0_S_453 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/453K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = npy.concatenate((P0_S,P0_S_453),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_453),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_453),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_453),axis=0)

        P0_S_473,T0_S_473,S0_S_473,Rubber0_S_473 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/473K_PMMA_CO2_Sw_Paper6.csv')
        P0_S = npy.concatenate((P0_S,P0_S_473),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_473),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_473),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_473),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper12':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_308,T0_S_308,S0_S_308,Rubber0_S_308 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper12M2/308K_PMMA_CO2_Sw_Paper12.csv')
        P0_S = P0_S_308
        T0_S = T0_S_308
        S0_S = S0_S_308
        Rubber0_S = Rubber0_S_308


    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper11':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_295,T0_S_295,S0_S_295,Rubber0_S_295 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/295K_PMMA_CO2_Sw_Paper11.csv')
        P0_S = P0_S_295
        T0_S = T0_S_295
        S0_S = S0_S_295
        Rubber0_S = Rubber0_S_295

        P0_S_313,T0_S_313,S0_S_313,Rubber0_S_313 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/313K_PMMA_CO2_Sw_Paper11.csv')
        P0_S = npy.concatenate((P0_S,P0_S_313),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_313),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_313),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_313),axis=0)

        P0_S_333,T0_S_333,S0_S_333,Rubber0_S_333 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/333K_PMMA_CO2_Sw_Paper11.csv')
        P0_S = npy.concatenate((P0_S,P0_S_333),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_333),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_333),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_333),axis=0)

        P0_S_353,T0_S_353,S0_S_353,Rubber0_S_353 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/353K_PMMA_CO2_Sw_Paper11.csv')
        P0_S = npy.concatenate((P0_S,P0_S_353),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_353),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_353),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_353),axis=0)

        P0_S_373,T0_S_373,S0_S_373,Rubber0_S_373 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/373K_PMMA_CO2_Sw_Paper11.csv')
        P0_S = npy.concatenate((P0_S,P0_S_373),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_373),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_373),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_373),axis=0)

    # if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper15':
    #     #======================================================
    #     #Swelling Data
    #     #======================================================

    #     P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper15Mahmood/ALL_K_PMMA_CO2_SwEmail_Paper15Mahmood.csv')
    #     P0_S = P0_S_ALL
    #     T0_S = T0_S_ALL
    #     S0_S = S0_S_ALL
    #     Rubber0_S = Rubber0_S_ALL

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper15':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper15Mahmood/ALL_K_PMMA_CO2_SwManuscript_Paper15Mahmood.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper5':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_293,T0_S_293,S0_S_293,Rubber0_S_293 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/293K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = P0_S_293
        T0_S = T0_S_293
        S0_S = S0_S_293
        Rubber0_S = Rubber0_S_293

        P0_S_298,T0_S_298,S0_S_298,Rubber0_S_298 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/298K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_298),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_298),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_298),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_298),axis=0)

        P0_S_303,T0_S_303,S0_S_303,Rubber0_S_303 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/303K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_303),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_303),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_303),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_303),axis=0)

        P0_S_308,T0_S_308,S0_S_308,Rubber0_S_308 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/308K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_308),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_308),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_308),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_308),axis=0)

        P0_S_313,T0_S_313,S0_S_313,Rubber0_S_313 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/313K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_313),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_313),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_313),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_313),axis=0)

        P0_S_318,T0_S_318,S0_S_318,Rubber0_S_318 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/318K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_318),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_318),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_318),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_318),axis=0)

        P0_S_323,T0_S_323,S0_S_323,Rubber0_S_323 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/323K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_323),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_323),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_323),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_323),axis=0)

        P0_S_328,T0_S_328,S0_S_328,Rubber0_S_328 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/328K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_328),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_328),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_328),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_328),axis=0)

        P0_S_333,T0_S_333,S0_S_333,Rubber0_S_333 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/333K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_333),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_333),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_333),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_333),axis=0)

        P0_S_338,T0_S_338,S0_S_338,Rubber0_S_338 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/338K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_338),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_338),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_338),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_338),axis=0)

        P0_S_343,T0_S_343,S0_S_343,Rubber0_S_343 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/343K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_343),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_343),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_343),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_343),axis=0)

        P0_S_348,T0_S_348,S0_S_348,Rubber0_S_348 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/348K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_348),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_348),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_348),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_348),axis=0)

        P0_S_353,T0_S_353,S0_S_353,Rubber0_S_353 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/353K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_353),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_353),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_353),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_353),axis=0)

        P0_S_358,T0_S_358,S0_S_358,Rubber0_S_358 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/358K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_358),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_358),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_358),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_358),axis=0)

        P0_S_363,T0_S_363,S0_S_363,Rubber0_S_363 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/Sw Paper5/363K_PMMA_CO2_Sw_Paper5.csv')
        P0_S = npy.concatenate((P0_S,P0_S_363),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_363),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_363),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_363),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper6':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_303,T0_X_303,X0_X_303,Rubber0_X_303 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/303K_PMMA_CO2_X_Paper6M1.csv')
        P0_X = P0_X_303
        T0_X = T0_X_303
        X0_X = X0_X_303
        Rubber0_X = Rubber0_X_303

        P0_X_313,T0_X_313,X0_X_313,Rubber0_X_313 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/313K_PMMA_CO2_X_Paper6M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_313),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_313),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_313),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_313),axis=0)

        P0_X_323,T0_X_323,X0_X_323,Rubber0_X_323 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper6M1/323K_PMMA_CO2_X_Paper6M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_323),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_323),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_323),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_323),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper7':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper7M1/308K_PMMA_CO2_X_Paper7M1.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

        P0_X_324,T0_X_324,X0_X_324,Rubber0_X_324 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper7M1/324K_PMMA_CO2_X_Paper7M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_324),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_324),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_324),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_324),axis=0)

        P0_X_354,T0_X_354,X0_X_354,Rubber0_X_354 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper7M1/354K_PMMA_CO2_X_Paper7M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_354),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_354),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_354),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_354),axis=0)

        P0_X_373,T0_X_373,X0_X_373,Rubber0_X_373 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper7M1/373K_PMMA_CO2_X_Paper7M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_373),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_373),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_373),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_373),axis=0)

        P0_X_405,T0_X_405,X0_X_405,Rubber0_X_405 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper7M1/405K_PMMA_CO2_X_Paper7M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_405),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_405),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_405),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_405),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper8':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper8M1/308K_PMMA_CO2_X_Paper8M1.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

        P0_X_338,T0_X_338,X0_X_338,Rubber0_X_338 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper8M1/338K_PMMA_CO2_X_Paper8M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_338),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_338),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_338),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_338),axis=0)

        P0_X_348,T0_X_348,X0_X_348,Rubber0_X_348 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper8M1/348K_PMMA_CO2_X_Paper8M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_348),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_348),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_348),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_348),axis=0)

        P0_X_358,T0_X_358,X0_X_358,Rubber0_X_358 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper8M1/358K_PMMA_CO2_X_Paper8M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_358),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_358),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_358),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_358),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper9':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_323,T0_X_323,X0_X_323,Rubber0_X_323 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper9M1/323K_PMMA_CO2_X_Paper9M1.csv')
        P0_X = P0_X_323
        T0_X = T0_X_323
        X0_X = X0_X_323
        Rubber0_X = Rubber0_X_323

        P0_X_338,T0_X_338,X0_X_338,Rubber0_X_338 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper9M1/338K_PMMA_CO2_X_Paper9M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_338),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_338),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_338),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_338),axis=0)

        P0_X_353,T0_X_353,X0_X_353,Rubber0_X_353 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper9M1/353K_PMMA_CO2_X_Paper9M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_353),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_353),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_353),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_353),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper10':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper10M2/308K_PMMA_CO2_X_Paper10M2.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

        P0_X_323,T0_X_323,X0_X_323,Rubber0_X_323 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper10M2/323K_PMMA_CO2_X_Paper10M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_323),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_323),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_323),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_323),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper11':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_295,T0_X_295,X0_X_295,Rubber0_X_295 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/295K_PMMA_CO2_X_Paper11M2.csv')
        P0_X = P0_X_295
        T0_X = T0_X_295
        X0_X = X0_X_295
        Rubber0_X = Rubber0_X_295

        P0_X_313,T0_X_313,X0_X_313,Rubber0_X_313 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/313K_PMMA_CO2_X_Paper11M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_313),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_313),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_313),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_313),axis=0)

        P0_X_333,T0_X_333,X0_X_333,Rubber0_X_333 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/333K_PMMA_CO2_X_Paper11M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_333),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_333),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_333),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_333),axis=0)

        P0_X_353,T0_X_353,X0_X_353,Rubber0_X_353 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/353K_PMMA_CO2_X_Paper11M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_353),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_353),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_353),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_353),axis=0)

        P0_X_373,T0_X_373,X0_X_373,Rubber0_X_373 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper11M2/373K_PMMA_CO2_X_Paper11M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_373),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_373),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_373),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_373),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper12':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper12M2/308K_PMMA_CO2_X_Paper12M2.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper13':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_313,T0_X_313,X0_X_313,Rubber0_X_313 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper13/313K_PMMA_CO2_X_Paper13.csv')
        P0_X = P0_X_313
        T0_X = T0_X_313
        X0_X = X0_X_313
        Rubber0_X = Rubber0_X_313

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper14':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_373,T0_X_373,X0_X_373,Rubber0_X_373 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper14/373K_PMMA_CO2_X_Paper14.csv')
        P0_X = P0_X_373
        T0_X = T0_X_373
        X0_X = X0_X_373
        Rubber0_X = Rubber0_X_373

        P0_X_398,T0_X_398,X0_X_398,Rubber0_X_398 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper14/398K_PMMA_CO2_X_Paper14.csv')
        P0_X = npy.concatenate((P0_X,P0_X_398),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_398),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_398),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_398),axis=0)

        P0_X_423,T0_X_423,X0_X_423,Rubber0_X_423 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper14/423K_PMMA_CO2_X_Paper14.csv')
        P0_X = npy.concatenate((P0_X,P0_X_423),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_423),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_423),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_423),axis=0)

        P0_X_448,T0_X_448,X0_X_448,Rubber0_X_448 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper14/448K_PMMA_CO2_X_Paper14.csv')
        P0_X = npy.concatenate((P0_X,P0_X_448),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_448),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_448),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_448),axis=0)

        P0_X_473,T0_X_473,X0_X_473,Rubber0_X_473 = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/X Paper14/473K_PMMA_CO2_X_Paper14.csv')
        P0_X = npy.concatenate((P0_X,P0_X_473),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_473),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_473),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_473),axis=0)

    # if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper15':
    #     #======================================================
    #     #Solubility Data
    #     #======================================================

    #     P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper15Mahmood/ALL_K_PMMA_CO2_XApparent_Paper15Mahmood.csv')
    #     P0_X = P0_X_ALL
    #     T0_X = T0_X_ALL
    #     X0_X = X0_X_ALL
    #     Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PMMA' and Solvent=='CO2' and Paper_Number=='Paper15':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PMMA CO2/XSw Paper15Mahmood/ALL_K_PMMA_CO2_XCorrected_Paper15Mahmood.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL


    #############################################################
    # PC-CO2
    #############################################################

    if Polymer_Type=='PC' and Solvent=='CO2' and Paper_Number=='Paper1':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_348,T0_X_348,X0_X_348,Rubber0_X_348 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper1M1/348K_PC_CO2_X_Paper1M1.csv')
        P0_X = P0_X_348
        T0_X = T0_X_348
        X0_X = X0_X_348
        Rubber0_X = Rubber0_X_348

        P0_X_373,T0_X_373,X0_X_373,Rubber0_X_373 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper1M1/373K_PC_CO2_X_Paper1M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_373),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_373),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_373),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_373),axis=0)

        P0_X_398,T0_X_398,X0_X_398,Rubber0_X_398 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper1M1/398K_PC_CO2_X_Paper1M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_398),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_398),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_398),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_398),axis=0)

        P0_X_423,T0_X_423,X0_X_423,Rubber0_X_423 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper1M1/423K_PC_CO2_X_Paper1M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_423),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_423),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_423),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_423),axis=0)

        P0_X_448,T0_X_448,X0_X_448,Rubber0_X_448 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper1M1/448K_PC_CO2_X_Paper1M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_448),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_448),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_448),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_448),axis=0)

    if Polymer_Type=='PC' and Solvent=='CO2' and Paper_Number=='Paper2':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper2M1/308K_PC_CO2_X_Paper2M1.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

    if Polymer_Type=='PC' and Solvent=='CO2' and Paper_Number=='Paper3':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper3M1/308K_PC_CO2_X_Paper3M1.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

    if Polymer_Type=='PC' and Solvent=='CO2' and Paper_Number=='Paper4':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_293,T0_X_293,X0_X_293,Rubber0_X_293 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper4M1/293K_PC_CO2_X_Paper4M1.csv')
        P0_X = P0_X_293
        T0_X = T0_X_293
        X0_X = X0_X_293
        Rubber0_X = Rubber0_X_293

        P0_X_313,T0_X_313,X0_X_313,Rubber0_X_313 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper4M1/313K_PC_CO2_X_Paper4M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_313),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_313),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_313),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_313),axis=0)

        P0_X_333,T0_X_333,X0_X_333,Rubber0_X_333 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper4M1/333K_PC_CO2_X_Paper4M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_333),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_333),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_333),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_333),axis=0)

        P0_X_353,T0_X_353,X0_X_353,Rubber0_X_353 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper4M1/353K_PC_CO2_X_Paper4M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_353),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_353),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_353),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_353),axis=0)

        P0_X_373,T0_X_373,X0_X_373,Rubber0_X_373 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper4M1/373K_PC_CO2_X_Paper4M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_373),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_373),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_373),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_373),axis=0)

        P0_X_393,T0_X_393,X0_X_393,Rubber0_X_393 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper4M1/393K_PC_CO2_X_Paper4M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_393),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_393),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_393),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_393),axis=0)

    if Polymer_Type=='PC' and Solvent=='CO2' and Paper_Number=='Paper5':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper5/308K_PC_CO2_X_Paper5.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

    if Polymer_Type=='PC' and Solvent=='CO2' and Paper_Number=='Paper6':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_313,T0_X_313,X0_X_313,Rubber0_X_313 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper6/313K_PC_CO2_X_Paper6.csv')
        P0_X = P0_X_313
        T0_X = T0_X_313
        X0_X = X0_X_313
        Rubber0_X = Rubber0_X_313

    if Polymer_Type=='PC' and Solvent=='CO2' and Paper_Number=='Paper7':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PC CO2/X Paper7/308K_PC_CO2_X_Paper7.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

    ####################################
    # PS-CO2
    ####################################

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper9':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_308,T0_S_308,S0_S_308,Rubber0_S_308 = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper9/308K_PS_CO2_Sw_Paper9.csv')
        P0_S = P0_S_308
        T0_S = T0_S_308
        S0_S = S0_S_308
        Rubber0_S = Rubber0_S_308

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper2':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_308,T0_S_308,S0_S_308,Rubber0_S_308 = loadBinaryData('Data/Post-Thesis Data/PS CO2/Sw Paper2/308K_PS_CO2_Sw_Paper2.csv')
        P0_S = P0_S_308
        T0_S = T0_S_308
        S0_S = S0_S_308
        Rubber0_S = Rubber0_S_308

        P0_S_323,T0_S_323,S0_S_323,Rubber0_S_323 = loadBinaryData('Data/Post-Thesis Data/PS CO2/Sw Paper2/323K_PS_CO2_Sw_Paper2.csv')
        P0_S = npy.concatenate((P0_S,P0_S_323),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_323),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_323),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_323),axis=0)

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper3':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_423,T0_S_423,S0_S_423,Rubber0_S_423 = loadBinaryData('Data/Post-Thesis Data/PS CO2/Sw Paper3/423K_PS_CO2_Sw_Paper3.csv')
        P0_S = P0_S_423
        T0_S = T0_S_423
        S0_S = S0_S_423
        Rubber0_S = Rubber0_S_423

        P0_S_473,T0_S_473,S0_S_473,Rubber0_S_473 = loadBinaryData('Data/Post-Thesis Data/PS CO2/Sw Paper3/473K_PS_CO2_Sw_Paper3.csv')
        P0_S = npy.concatenate((P0_S,P0_S_473),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_473),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_473),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_473),axis=0)

        P0_S_523,T0_S_523,S0_S_523,Rubber0_S_523 = loadBinaryData('Data/Post-Thesis Data/PS CO2/Sw Paper3/523K_PS_CO2_Sw_Paper3.csv')
        P0_S = npy.concatenate((P0_S,P0_S_523),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_523),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_523),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_523),axis=0)

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper4':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_383,T0_X_383,X0_X_383,Rubber0_X_383 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper4M1/383K_PS_CO2_X_Paper4M1.csv')
        P0_X = P0_X_383
        T0_X = T0_X_383
        X0_X = X0_X_383
        Rubber0_X = Rubber0_X_383

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper5':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_373,T0_X_373,X0_X_373,Rubber0_X_373 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper5M1/373K_PS_CO2_X_Paper5M1.csv')
        P0_X = P0_X_373
        T0_X = T0_X_373
        X0_X = X0_X_373
        Rubber0_X = Rubber0_X_373

        P0_X_413,T0_X_413,X0_X_413,Rubber0_X_413 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper5M1/413K_PS_CO2_X_Paper5M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_413),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_413),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_413),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_413),axis=0)

        P0_X_453,T0_X_453,X0_X_453,Rubber0_X_453 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper5M1/453K_PS_CO2_X_Paper5M1.csv')
        P0_X = npy.concatenate((P0_X,P0_X_453),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_453),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_453),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_453),axis=0)

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper6':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper6M2/308K_PS_CO2_X_Paper6M2.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

        P0_X_324,T0_X_324,X0_X_324,Rubber0_X_324 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper6M2/324K_PS_CO2_X_Paper6M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_324),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_324),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_324),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_324),axis=0)

        P0_X_354,T0_X_354,X0_X_354,Rubber0_X_354 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper6M2/354K_PS_CO2_X_Paper6M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_354),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_354),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_354),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_354),axis=0)

        P0_X_373,T0_X_373,X0_X_373,Rubber0_X_373 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper6M2/373K_PS_CO2_X_Paper6M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_373),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_373),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_373),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_373),axis=0)

        P0_X_405,T0_X_405,X0_X_405,Rubber0_X_405 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper6M2/405K_PS_CO2_X_Paper6M2.csv')
        P0_X = npy.concatenate((P0_X,P0_X_405),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_405),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_405),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_405),axis=0)

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper7':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_313,T0_X_313,X0_X_313,Rubber0_X_313 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper7M2/313K_PS_CO2_X_Paper7M2.csv')
        P0_X = P0_X_313
        T0_X = T0_X_313
        X0_X = X0_X_313
        Rubber0_X = Rubber0_X_313

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper8':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_373,T0_X_373,X0_X_373,Rubber0_X_373 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper8/373K_PS_CO2_X_Paper8.csv')
        P0_X = P0_X_373
        T0_X = T0_X_373
        X0_X = X0_X_373
        Rubber0_X = Rubber0_X_373

        P0_X_473,T0_X_473,X0_X_473,Rubber0_X_473 = loadBinaryData('Data/Post-Thesis Data/PS CO2/X Paper8/473K_PS_CO2_X_Paper8.csv')
        P0_X = npy.concatenate((P0_X,P0_X_473),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_473),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_473),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_473),axis=0)

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper9':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_308,T0_X_308,X0_X_308,Rubber0_X_308 = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper9/308K_PS_CO2_X_Paper9.csv')
        P0_X = P0_X_308
        T0_X = T0_X_308
        X0_X = X0_X_308
        Rubber0_X = Rubber0_X_308

    # if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11':
    #     #======================================================
    #     #Solubility Data
    #     #======================================================

    #     P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11/ALL_K_PS_CO2_XApparent_Paper11.csv')
    #     P0_X = P0_X_ALL
    #     T0_X = T0_X_ALL
    #     X0_X = X0_X_ALL
    #     Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11/ALL_K_PS_CO2_XSelfCorrected_Paper11.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11/ALL_K_PS_CO2_Sw_Paper11.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL

    # if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper12':
    #     #======================================================
    #     #Solubility Data
    #     #======================================================

    #     P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper12/ALL_K_PS_CO2_XApparent_Paper12.csv')
    #     P0_X = P0_X_ALL
    #     T0_X = T0_X_ALL
    #     X0_X = X0_X_ALL
    #     Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper12':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper12/ALL_K_PS_CO2_XSelfCorrected_Paper12.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper12':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper12/ALL_K_PS_CO2_Sw_Paper12.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL


    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper13':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper13/ALL_K_PS_CO2_XCorrected_Paper13.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper13':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper13/ALL_K_PS_CO2_Sw_Paper13.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL


    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11_12_13':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11_12_13/ALL_K_PS_CO2_XCorrected_Paper11_12_13.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11_12_13':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11_12_13/ALL_K_PS_CO2_Sw_Paper11_12_13.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL


    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11_13':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11_13/ALL_K_PS_CO2_XCorrected_Paper11_13.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11_13':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11_13/ALL_K_PS_CO2_Sw_Paper11_13.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11_13_LowTemp':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11_13/ALL_K_PS_CO2_XCorrected_Paper11_13.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11_13_LowTemp':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11_13/ALL_K_PS_CO2_Sw_Paper11_13.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11_12':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11_12/ALL_K_PS_CO2_XCorrected_Paper11_12.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper11_12':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper11_12/ALL_K_PS_CO2_Sw_Paper11_12.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL


    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper12_13':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper12_13/ALL_K_PS_CO2_XCorrected_Paper12_13.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper12_13':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper12_13/ALL_K_PS_CO2_Sw_Paper12_13.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL


    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper4_11_12':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_ALL,T0_X_ALL,X0_X_ALL,Rubber0_X_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper4_11_12/ALL_K_PS_CO2_XCorrected_Paper4_11_12.csv')
        P0_X = P0_X_ALL
        T0_X = T0_X_ALL
        X0_X = X0_X_ALL
        Rubber0_X = Rubber0_X_ALL

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Paper4_11_12':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_ALL,T0_S_ALL,S0_S_ALL,Rubber0_S_ALL = loadBinaryData('Data/Post-Thesis Data/PS CO2/XSw Paper4_11_12/ALL_K_PS_CO2_Sw_Paper11_12.csv')
        P0_S = P0_S_ALL
        T0_S = T0_S_ALL
        S0_S = S0_S_ALL
        Rubber0_S = Rubber0_S_ALL


    #############################################################
    # PLA-CO2
    #############################################################

    if Polymer_Type=='PLA' and Solvent=='CO2' and Paper_Number=='Park':
        #======================================================
        #Swelling Data
        #======================================================

        P0_S_453,T0_S_453,S0_S_453,Rubber0_S_453 = loadBinaryData('Data/Post-Thesis Data/PLA CO2/Sw Park/453K_PLA_CO2_Sw_Park.csv')
        P0_S = P0_S_453
        T0_S = T0_S_453
        S0_S = S0_S_453
        Rubber0_S = Rubber0_S_453

        P0_S_463,T0_S_463,S0_S_463,Rubber0_S_463 = loadBinaryData('Data/Post-Thesis Data/PLA CO2/Sw Park/463K_PLA_CO2_Sw_Park.csv')
        P0_S = npy.concatenate((P0_S,P0_S_463),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_463),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_463),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_463),axis=0)

        P0_S_473,T0_S_473,S0_S_473,Rubber0_S_473 = loadBinaryData('Data/Post-Thesis Data/PLA CO2/Sw Park/473K_PLA_CO2_Sw_Park.csv')
        P0_S = npy.concatenate((P0_S,P0_S_473),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_473),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_473),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_473),axis=0)

    if Polymer_Type=='PLA' and Solvent=='CO2' and Paper_Number=='Park':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_453,T0_X_453,X0_X_453,Rubber0_X_453 = loadBinaryData('Data/Post-Thesis Data/PLA CO2/X Park/453K_PLA_CO2_X_Park.csv')
        P0_X = P0_X_453
        T0_X = T0_X_453
        X0_X = X0_X_453
        Rubber0_X = Rubber0_X_453

        P0_X_463,T0_X_463,X0_X_463,Rubber0_X_463 = loadBinaryData('Data/Post-Thesis Data/PLA CO2/X Park/463K_PLA_CO2_X_Park.csv')
        P0_X = npy.concatenate((P0_X,P0_X_463),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_463),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_463),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_463),axis=0)

        P0_X_473,T0_X_473,X0_X_473,Rubber0_X_473 = loadBinaryData('Data/Post-Thesis Data/PLA CO2/X Park/473K_PLA_CO2_X_Park.csv')
        P0_X = npy.concatenate((P0_X,P0_X_473),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_473),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_473),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_473),axis=0)











    #======================================================
    #Pre-Thesis Data Below:
    #======================================================

    if Polymer_Type=='PS' and Solvent=='CO2' and Paper_Number=='Pre-Thesis':
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_403,T0_X_403,X0_X_403,Rubber0_X_403 = loadBinaryData('Data/Pre-Thesis Data/403K_PS_CO2_X.csv')
        P0_X = P0_X_403
        T0_X = T0_X_403
        X0_X = X0_X_403
        Rubber0_X = Rubber0_X_403

        P0_X_423,T0_X_423,X0_X_423,Rubber0_X_423 = loadBinaryData('Data/Pre-Thesis Data/423K_PS_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_423),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_423),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_423),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_423),axis=0)

        P0_X_463,T0_X_463,X0_X_463,Rubber0_X_463 = loadBinaryData('Data/Pre-Thesis Data/463K_PS_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_463),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_463),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_463),axis=0)
        Rubber0_X = npy.concatenate((Rubber0_X,Rubber0_X_463),axis=0)

        #======================================================
        #Swelling Data
        #======================================================

        P0_S_403,T0_S_403,S0_S_403,Rubber0_S_403 = loadBinaryData('Data/Pre-Thesis Data/403K_PS_CO2_Sw.csv')
        P0_S = P0_S_403
        T0_S = T0_S_403
        S0_S = S0_S_403
        Rubber0_S = Rubber0_S_403

        P0_S_423,T0_S_423,S0_S_423,Rubber0_S_423 = loadBinaryData('Data/Pre-Thesis Data/423K_PS_CO2_Sw.csv')
        P0_S = npy.concatenate((P0_S,P0_S_423),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_423),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_423),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_423),axis=0)

        P0_S_463,T0_S_463,S0_S_463,Rubber0_S_463 = loadBinaryData('Data/Pre-Thesis Data/463K_PS_CO2_Sw.csv')
        P0_S = npy.concatenate((P0_S,P0_S_463),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_463),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_463),axis=0)
        Rubber0_S = npy.concatenate((Rubber0_S,Rubber0_S_463),axis=0)

    if Polymer_Type=='PMMA' and Solvent=='CO2' and True and Paper_Number=='Pre-Thesis':

        #======================================================
        #Solubility Data
        #======================================================

        P0_X_306,T0_X_306,X0_X_306 = loadBinaryData('Data/Pre-Thesis Data/306K_PMMA_CO2_X.csv')
        P0_X = P0_X_306
        T0_X = T0_X_306
        X0_X = X0_X_306

        P0_X_315,T0_X_315,X0_X_315 = loadBinaryData('Data/Pre-Thesis Data/315K_PMMA_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_315),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_315),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_315),axis=0)

        P0_X_332,T0_X_332,X0_X_332 = loadBinaryData('Data/Pre-Thesis Data/332K_PMMA_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_332),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_332),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_332),axis=0)

        #======================================================
        #Swelling Data
        #======================================================

        P0_S_306,T0_S_306,S0_S_306 = loadBinaryData('Data/Pre-Thesis Data/306K_PMMA_CO2_Sw.csv')
        P0_S = P0_S_306
        T0_S = T0_S_306
        S0_S = S0_S_306

        P0_S_315,T0_S_315,S0_S_315 = loadBinaryData('Data/Pre-Thesis Data/315K_PMMA_CO2_Sw.csv')
        P0_S = npy.concatenate((P0_S,P0_S_315),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_315),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_315),axis=0)

        P0_S_332,T0_S_332,S0_S_332 = loadBinaryData('Data/Pre-Thesis Data/332K_PMMA_CO2_Sw.csv')
        P0_S = npy.concatenate((P0_S,P0_S_332),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_332),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_332),axis=0)

    #New Paper of PMMA CO2, Paper Title: Simultaneous Measurement of Swelling and Sorption in a Supercritical CO2-Poly(methyl methacrylate) System

    if Polymer_Type=='PMMA' and Solvent=='CO2'and True and Paper_Number=='Pre-Thesis':

        #======================================================
        #Solubility Data
        #======================================================

        P0_X_323,T0_X_323,X0_X_323 = loadBinaryData('Data/Pre-Thesis Data/323K_PMMA_CO2_X.csv')
        
        P0_X = npy.concatenate((P0_X,P0_X_323),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_323),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_323),axis=0)

        # P0_X = P0_X_323
        # T0_X = T0_X_323
        # X0_X = X0_X_323

        P0_X_338,T0_X_338,X0_X_338 = loadBinaryData('Data/Pre-Thesis Data/338K_PMMA_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_338),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_338),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_338),axis=0)

        P0_X_353,T0_X_353,X0_X_353 = loadBinaryData('Data/Pre-Thesis Data/353K_PMMA_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_353),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_353),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_353),axis=0)

        #======================================================
        #Swelling Data
        #======================================================

        P0_S_323,T0_S_323,S0_S_323 = loadBinaryData('Data/Pre-Thesis Data/323K_PMMA_CO2_Sw.csv')
        P0_S = P0_S_323
        T0_S = T0_S_323
        S0_S = S0_S_323

        P0_S_338,T0_S_338,S0_S_338 = loadBinaryData('Data/Pre-Thesis Data/338K_PMMA_CO2_Sw.csv')
        P0_S = npy.concatenate((P0_S,P0_S_338),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_338),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_338),axis=0)

        P0_S_353,T0_S_353,S0_S_353 = loadBinaryData('Data/Pre-Thesis Data/353K_PMMA_CO2_Sw.csv')
        P0_S = npy.concatenate((P0_S,P0_S_353),axis=0)
        T0_S = npy.concatenate((T0_S,T0_S_353),axis=0)
        S0_S = npy.concatenate((S0_S,S0_S_353),axis=0)


    if Polymer_Type=='PC' and Solvent=='CO2' and Paper_Number=='Pre-Thesis':
        
        #======================================================
        #Solubility Data
        #======================================================

        P0_X_313,T0_X_313,X0_X_313 = loadBinaryData('Data/Pre-Thesis Data/313K_PC_CO2_X.csv')
        P0_X = P0_X_313
        T0_X = T0_X_313
        X0_X = X0_X_313

        P0_X_323,T0_X_323,X0_X_323 = loadBinaryData('Data/Pre-Thesis Data/323K_PC_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_323),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_323),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_323),axis=0)

        P0_X_333,T0_X_333,X0_X_333 = loadBinaryData('Data/Pre-Thesis Data/333K_PC_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_333),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_333),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_333),axis=0)

        P0_X_348,T0_X_348,X0_X_348 = loadBinaryData('Data/Pre-Thesis Data/348K_PC_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_348),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_348),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_348),axis=0)

        P0_X_373,T0_X_373,X0_X_373 = loadBinaryData('Data/Pre-Thesis Data/373K_PC_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_373),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_373),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_373),axis=0)

        # P0_X_373,T0_X_373,X0_X_373 = loadBinaryData('Data/Pre-Thesis Data/373K_PC_CO2_X.csv')
        # P0_X = P0_X_373
        # T0_X = T0_X_373
        # X0_X = X0_X_373

        P0_X_398,T0_X_398,X0_X_398 = loadBinaryData('Data/Pre-Thesis Data/398K_PC_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_398),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_398),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_398),axis=0)

        P0_X_423,T0_X_423,X0_X_423 = loadBinaryData('Data/Pre-Thesis Data/423K_PC_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_423),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_423),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_423),axis=0)

        P0_X_448,T0_X_448,X0_X_448 = loadBinaryData('Data/Pre-Thesis Data/448K_PC_CO2_X.csv')
        P0_X = npy.concatenate((P0_X,P0_X_448),axis=0)
        T0_X = npy.concatenate((T0_X,T0_X_448),axis=0)
        X0_X = npy.concatenate((X0_X,X0_X_448),axis=0)

        #======================================================
        #Swelling Data
        #======================================================

        P0_S_308,T0_S_308,S0_S_308 = loadBinaryData('Data/Pre-Thesis Data/308K_PC_CO2_Sw.csv')
        P0_S = P0_S_308
        T0_S = T0_S_308
        S0_S = S0_S_308


    return P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Rubber0_X,Rubber0_S

