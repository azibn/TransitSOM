#utils functions file
import numpy as np

def phasefold(time,per,t0):
    return np.mod(time-t0,per)/per

def WeightMeanClipped(vals):
    values = vals[:,0]
    weights = vals[:,1]
    mean = np.sum(values*weights)/(np.sum(weights))
    outliers = np.abs((values - mean)/vals[:,2]) > vals[0,3]
    mean = np.sum(values[~outliers]*weights[~outliers])/(np.sum(weights[~outliers]))
    return mean

def WeightMean(vals):
    values = vals[:,0]
    weights = vals[:,1]
    mean = np.sum(values*weights)/(np.sum(weights))
    return mean

def WeightMean_witherror(vals):
    values = vals[:,0]
    weights = vals[:,1]
    mean = np.sum(values*weights)/(np.sum(weights))
    error = np.sqrt(1./np.sum(weights))
    return mean,error

def GetBinnedVals(time,x,lowerr,higherr,bins,clip_outliers=0):
    #tobin = []
    #for i in range(len(x)):
    #    tobin.append([x[i],np.power(2./(higherr[i]+lowerr[i]),2),(higherr[i]+lowerr[i])/2.,clip_outliers])
    tobin = np.zeros([len(x),4])
    tobin[:,0] = x
    tobin[:,1] = np.power(2./(higherr+lowerr),2)  #inverse errors squared
    tobin[:,2] = (higherr+lowerr)/2.
    tobin[:,3] = clip_outliers

    bin_edges = np.linspace(time[0], time[-1], bins+1)
    bin_edges[-1]+=0.0001
    binnumber = np.digitize(time, bin_edges)
    timemeans = np.zeros(bins)
    if clip_outliers:
        for i in range(bins):
            segment = tobin[binnumber == i+1]
            if segment.shape[0] > 0:
                timemeans[i] = WeightMeanClipped(segment)
            else:
                timemeans[i] = np.nan
    
        #timemeans, bin_edges, binnumber = stats.binned_statistic(time,tobin,statistic=WeightMeanClipped,bins=bins)
    else:
        for i in range(bins):
            segment = tobin[binnumber == i+1]
            if segment.shape[0] > 0:
                timemeans[i] = WeightMean(segment)
            else:
                timemeans[i] = np.nan
        #timemeans, bin_edges, binnumber = stats.binned_statistic(time,tobin,statistic=WeightMean,bins=bins)
    binerrors = np.zeros(len(timemeans))

    for bin in range(len(timemeans)):

        weights = tobin[binnumber==bin+1,1]
        values = tobin[binnumber==bin+1,0]
        if clip_outliers:
            outliers = np.abs((values - timemeans[bin])/tobin[binnumber==bin+1,2]) > clip_outliers
            binerrors[bin] = np.sqrt(1./np.sum(weights[~outliers])) 
        else:
            binerrors[bin] = np.sqrt(1./np.sum(weights))
        #binerrors[bin] = np.std(Aps_offset[binnumber==bin+1])/np.sqrt(np.sum(binnumber==bin+1))
    
    binnedtimes = (bin_edges[1:]+bin_edges[:-1])/2.

    return binnedtimes,timemeans,binerrors

def GetBinnedVals_nooutliers(time,x,lowerr,higherr,bins):
    tobin = np.zeros([len(x),3])
    tobin[:,0] = x
    tobin[:,1] = np.power(2./(higherr+lowerr),2)  #inverse errors squared
    tobin[:,2] = (higherr+lowerr)/2.

    bin_edges = np.linspace(time[0], time[-1], bins+1)
    bin_edges[-1]+=0.0001
    binnumber = np.digitize(time, bin_edges)
    timemeans = np.zeros(bins)
    binerrors = np.zeros(len(timemeans))
    for i in range(bins):
        segment = tobin[binnumber == i+1]
        if segment.shape[0] > 0:
            timemeans[i],binerrors[i] = WeightMean_witherror(segment)
        else:
            timemeans[i] = np.nan
            binerrors[i] = np.nan
    
    binnedtimes = (bin_edges[1:]+bin_edges[:-1])/2.

    return binnedtimes,timemeans,binerrors

def PrepareArrays(SOMarray,errorarray):
      
    #replace nans, normalise transits, select high SNR objects
    for idx in range(SOMarray.shape[0]):
        mask = np.isnan(SOMarray[idx,:])
        SOMarray[idx,mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), SOMarray[idx,~mask])
        SOMarray[idx,:],errorarray[idx,:] = SOMNormalise(SOMarray[idx,:],errorarray[idx,:])

    return SOMarray,errorarray
    
    
def SOMNormalise(flux,errors):
    sorted = np.sort(flux)
    lowlevel = np.mean(sorted[:5])
    flux -= lowlevel
    flux *= 1./(np.mean(sorted[-30:])-lowlevel)
    errors = errors * 1./(np.mean(sorted[-30:])-lowlevel)
    return flux,errors


