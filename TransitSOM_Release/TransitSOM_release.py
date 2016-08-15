
import somtools
import utils
reload(utils)
import selfsom
import os
import numpy as np

#convenience functions for TransitSOM release

 #prepare lightcurve(s)
 #lightcurves in format time, flux, error
 #lightcurves which do not exist or which have no points in 3 duration window are skipped
 #empty bins will be interpolated
def PrepareLightcurves(filelist,periods,t0s,tdurs,nbins=50):

    
    SOMarray_bins = []
    som_ids = []
    SOMarray_binerrors = []
    for i,infile in enumerate(filelist):
        print i
        if os.path.exists(infile):
            #load lightcurve file  REMOVE SKIP HEADER AND FIRST ROW CUT IN PRACTICE
            lc = np.genfromtxt(infile,skip_header=35)
            lc = lc[:,(0,2,3)]
            lc = lc[~np.isnan(lc[:,1]),:]

            #get period,T0, Tdur for KOI data
            per = periods[i]
            t0 = t0s[i]
            tdur = tdurs[i]
            
            #phase fold (transit at 0.5)         
            phase = utils.phasefold(lc[:,0],per,t0-per*0.5)
            idx = np.argsort(phase)
            lc = lc[idx,:]
            phase = phase[idx]

            #cut to relevant region
            tdur_phase = tdur/per
            lowidx = np.searchsorted(phase,0.5-tdur_phase*1.5)
            highidx = np.searchsorted(phase,0.5+tdur_phase*1.5)
            lc = lc[lowidx:highidx,:]
            phase = phase[lowidx:highidx]

            #perform binning
            if len(lc[:,0]) != 0: 
                #bin
                binphases,SOMtransit_bin,binerrors = utils.GetBinnedVals(phase,lc[:,1],lc[:,2],lc[:,2],nbins,clip_outliers=5)    

            #append to SOMarray:
                SOMarray_bins.append(SOMtransit_bin)
                som_ids.append(i)
                SOMarray_binerrors.append(binerrors)
    
    #normalise arrays, and interpolate nans where necessary
    SOMarray_bins = np.array(SOMarray_bins)
    SOMarray_binerrors = np.array(SOMarray_binerrors)
    SOMarray_bins,SOMarray_binerrors = utils.PrepareArrays(SOMarray_bins,SOMarray_binerrors)
    
    return SOMarray_bins, SOMarray_binerrors, np.array(som_ids)
  
#missionflag= 0 for kepler, 1 for k2
def ClassifyPlanet(SOMarray,SOMerrors,n_mc=1000,som=None,groups=None,missionflag=0,case=1,map_all=np.zeros([5,5,5])-1):

    #INSERT INPUT CHECKS HERE, breakout if fail:
    	#missionflag 0 or 1
    	#n_mc an integer, at least 1
    	#som the appropriate object
    	#groups an array of ints, length to match SOMarray
    	#case 1 or 2, if case=1 groups array given

    #if no SOM, load our SOM (kepler or k2 depending on keplerflag)
    if not som:
        selfflag = 1
        if missionflag == 0:
            def Init(sample):
                return np.random.uniform(0,2,size=(20,20,50))
            som = selfsom.SimpleSOMMapper((20,20),1,initialization_func=Init,learning_rate=0.1)
            loadk = somtools.KohonenLoad('snrcut_30_lr01_300_20_20_bin50.txt')
            som.train(loadk) #tricks the som into thinking it's been trained
            som._K = loadk  #loads the actual Kohonen layer into place.
        else:
            def Init(sample):
                return np.random.uniform(0,2,size=(8,8,20))

            som = selfsom.SimpleSOMMapper((8,8),1,initialization_func=Init,learning_rate=0.1)
            loadk = somtools.KohonenLoad('/Users/davidarmstrong/Software/Python/TransitSOM/K2/k2all_lr01_500_8_8_bin20.txt')
            som.train(loadk) #tricks the som into thinking it's been trained
            som._K = loadk  #loads the actual Kohonen layer into place.        

    else:
        selfflag = 0

    #apply SOM
    mapped = som(SOMarray)


    #map_all results
    if (map_all<0).all():
        map_all = somtools.MapErrors_MC(som,SOMarray,SOMerrors,n_mc)
    

    #classify (depending on case)
    
    if case==1:
        
        if selfflag:  #load pre calculated proportions
            if missionflag==0:
                prop = somtools.KohonenLoad('prop_snrcut_all_kepler.txt')
                prop_weights = np.genfromtxt('prop_snrcut_all_weights_kepler.txt')
            else:
                prop = somtools.KohonenLoad('prop_all_k2.txt')
                prop_weights = np.genfromtxt('prop_all_weights_k2.txt')
                
        
        else:  #create new proportions
            prop ,prop_weights= somtools.Proportions(som.K,mapped,groups,2,som.K.shape[0],som.K.shape[1])
        class_probs = somtools.Classify(map_all,prop,2,prop_weights) 
        planet_prob = class_probs[:,0] / np.sum(class_probs,axis=1)
 
    else:

            if selfflag:
                if missionflag==0:
                    testdistances = somtools.KohonenLoad('testdistances_kepler.txt')
                else:
                    testdistances = somtools.KohonenLoad('testdistances_k2.txt')
                
            else:
                SOMarray_PDVM = np.genfromtxt('SOMarray_PDVM_perfect_norm.txt')
                groups_PDVM = np.genfromtxt('groups_PDVM_perfect_norm.txt')
                lowbound = np.floor(SOMarray.shape[1]/3).astype('int')
                testdistances = somtools.PixelClassifier(som.K,SOMarray_PDVM,groups_PDVM,6,lowbound=lowbound,highbound=2*lowbound)
            
            #apply classification
            planet_prob,class_power = somtools.Classify_Distances(map_all,testdistances)

    
    return planet_prob
    
    
#somshape = (int,int)
def CreateSOM(SOMarray,niter=500,learningrate=0.1,learningradius=None,somshape=(20,20),outfile=None):

    nbins = SOMarray.shape[1]
    
    if not learningradius:
        learningradius = np.max(somshape)
    
    def Init(sample):
        return np.random.uniform(0,2,size=(somshape[0],somshape[1],nbins))
    
    #initialise som
    som = selfsom.SimpleSOMMapper(somshape,niter,initialization_func=Init,learning_rate=learningrate,iradius=learningradius)
    #train som
    som.train(SOMarray)

    #save
    if outfile:
        somtools.KohonenSave(som.K,outfile)
    
    #return trained som
    return som
