#plotting functions for TransitSOM paper

import pylab as p
p.ion()
import numpy as np

def SOMLocPlot(groups,mapped,colours= ['m','r','g','k','b','c'],sizes= [3.0,3.0,3.0,3.0,3.0,3.0]):
    colours = ['m','r','g','k','b','c','k']
    sizes = [6.0,2.0,3.0,3.0,3.0,6.0,6.0]
    
    p.figure()
    p.clf()
    for obj in range(len(groups)):
        map = mapped_snrcut_all[obj]
        type = colours[groups[obj].astype('int')]
        size = sizes[groups[obj].astype('int')]
        p.plot(map[0]+np.random.uniform(-0.5,0.5),map[1]+np.random.uniform(-0.5,0.5),type+'.',markersize=size)
    p.xlim(np.min(mapped[:,0])-0.5,np.max(mapped[:,0])+0.5)
    p.ylim(np.min(mapped[:,1])-0.5,np.max(mapped[:,1])+0.5)
    p.xlabel('SOM Pixel Index (x)',fontsize=12)
    p.ylabel('SOM Pixel Index (y)',fontsize=12)


def TemplatesPlot(som_snrcut,loc0,loc1,loc2,loc3):
    f, ((ax1, ax2), (ax3, ax4)) = p.subplots(2, 2, sharex='col', sharey='row')
    xvals = np.arange(som_snrcut.K.shape[2])/(float(som_snrcut.K.shape[2])) * 3 - 1.5  #measured in Transit Durations
    
    ax1.plot(xvals,som_snrcut.K[loc0[0],loc0[1]],'b.-')
    ax2.plot(xvals,som_snrcut.K[loc1[0],loc1[1]],'b.-')
    ax3.plot(xvals,som_snrcut.K[loc2[0],loc2[1]],'g.-')
    ax4.plot(xvals,som_snrcut.K[loc3[0],loc3[1]],'g.-')
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0)
    p.setp(ax3.get_yticklabels()[-2:],visible=False)
    p.setp(ax4.get_xticklabels()[0],visible=False)
    f.text(0.5, 0.02, 'Phase (Transit Durations from centre)', ha='center',fontsize=12)
    f.text(0.04, 0.5, 'Normalised Amplitude', va='center', rotation='vertical',fontsize=12)

def ProportionsPlot(prop_snrcut_all):
    p.figure(3)
    p.clf()
    p.imshow(prop_snrcut_all[0,:,:].T,origin='lower',interpolation='nearest',cmap='viridis')
    cb = p.colorbar()
    cb.set_label('Planet Proportion',rotation=270,labelpad=15,fontsize=12)
    p.xlabel('SOM Pixel Index (x)',fontsize=12)
    p.ylabel('SOM Pixel Index (y)',fontsize=12)

def DistancesPlot(testdistances):
    p.figure(4)
    p.clf()
    avgdistances = np.mean(testdistances[:,:,(0,3)],axis=2)
    avgdistances[avgdistances>=3] = np.nan
    p.imshow(avgdistances.T,interpolation='nearest',origin='lower',cmap='viridis') 
    cb = p.colorbar()
    cb.set_label('Summed Distance',rotation=270,labelpad=15,fontsize=12)
    p.xlabel('SOM Pixel Index (x)',fontsize=12)
    p.ylabel('SOM Pixel Index (y)',fontsize=12)   
    


def Histograms_1(planet_prob,groups_test,planet_prob_snrcut,groups_testsnrcut):
    f, (ax1, ax2) = p.subplots(2, sharex=True)
    n1, bins1, patches1 = ax1.hist(planet_prob[groups_test==0],bins=40,color='#33FFFF')
    ax1.hist(planet_prob_snrcut[groups_testsnrcut==0],bins1,color='#0000FF')
    n2, bins2, patches2 = ax2.hist(planet_prob[groups_test==1],bins=40,color='#80FF00')
    ax2.hist(planet_prob_snrcut[groups_testsnrcut==1],bins2,color='#006600')
    ax1.plot([0.5,0.5],[0,300],'k--')
    ax2.plot([0.5,0.5],[0,900],'k--')
    p.setp(ax2.get_yticklabels()[-1],visible=False)
    ax2.set_xlim(0.,1.0)
    ax1.set_ylim(0,300)
    ax2.set_ylim(0.1,900)
    f.subplots_adjust(hspace=0)
    ax2.set_xlabel(r'$\theta_1$',fontsize=12)
    ax1.set_ylabel('N',fontsize=14)
    ax2.set_ylabel('N',fontsize=14)
    
def Histograms_2(planet_prob,groups_test,planet_prob_snrcut,groups_testsnrcut):
    f, (ax1, ax2) = p.subplots(2, sharex=True)
    n1, bins1, patches1 = ax1.hist(planet_prob[groups_test==0],bins=40,color='#33FFFF')
    ax1.hist(planet_prob_snrcut[groups_testsnrcut==0],bins1,color='#0000FF')
    n2, bins2, patches2 = ax2.hist(planet_prob[groups_test==1],bins=40,color='#80FF00')
    ax2.hist(planet_prob_snrcut[groups_testsnrcut==1],bins2,color='#006600')
    ax1.plot([0.5,0.5],[0,450],'k--')
    ax2.plot([0.5,0.5],[0,500],'k--')
    p.setp(ax2.get_yticklabels()[-1],visible=False)
    ax2.set_xlim(0.1,0.9)
    ax1.set_ylim(0,450)
    ax2.set_ylim(0.1,500)
    f.subplots_adjust(hspace=0)
    ax2.set_xlabel(r'$\theta_2$',fontsize=12)
    ax1.set_ylabel('N',fontsize=14)
    ax2.set_ylabel('N',fontsize=14)

def KeplerResultsTable(theta1,theta2,ids):
    out = np.zeros([len(theta1),3])
    theta1[np.isnan(theta1)] = -1
    theta2[np.isnan(theta2)] = -1
    out[:,0] = ids
    out[:,1] = theta1
    out[:,2] = theta2
    idx = np.argsort(theta1)[::-1]
    out = out[idx,:]
    
    np.savetxt('/Users/davidarmstrong/Documents/Papers/TransitSOM/kepresults.txt',out,delimiter=',',fmt=['%-.9u','%.3f','%.3f'],header='Kepler ID, Theta 1, Theta 2')

def K2ResultsTable(theta1,theta2,ids):
    out = np.zeros([len(theta1),3])
    theta1[np.isnan(theta1)] = -1
    theta2[np.isnan(theta2)] = -1
    out[:,0] = ids
    out[:,1] = theta1
    out[:,2] = theta2
    idx = np.argsort(theta1)[::-1]
    out = out[idx,:]
    np.savetxt('/Users/davidarmstrong/Documents/Papers/TransitSOM/k2results.txt',out,delimiter=',',fmt=['%-.9u','%.3f','%.3f'],header='Kepler ID, Theta 1, Theta 2')


def K2SOMLocPlot(mappedK2,groupsK2):
    colours = ['b','g','r','k','b','c','k']
    sizes = [6.0,6.0,6.0,3.0,3.0,6.0,6.0]
    
    p.figure(7)
    p.clf()
    for obj in range(len(groupsK2)):
        map = mappedK2[obj]
        type = colours[groupsK2[obj].astype('int')]
        size = sizes[groupsK2[obj].astype('int')]
        #if (groups_SNRcut[obj] != 4) & (groups_SNRcut[obj] != 2) :
        #if (groups[obj] != 1) :
        p.plot(map[0]+np.random.uniform(-0.5,0.5),map[1]+np.random.uniform(-0.5,0.5),type+'.',markersize=size)
    p.xlim(-0.5,7.5)
    p.ylim(-0.5,7.5)
    p.xlabel('SOM Pixel Index (x)',fontsize=14)
    p.ylabel('SOM Pixel Index (y)',fontsize=14)

def K2DistancesPlot(testdistances):
    p.figure(8)
    p.clf()
    avgdistances = np.mean(testdistances[:,:,(0,3)],axis=2)
    avgdistances[avgdistances>=3] = np.nan
    p.imshow(avgdistances.T,interpolation='nearest',origin='lower',cmap='viridis') 
    cb = p.colorbar()
    cb.set_label('Summed Distance',rotation=270,labelpad=15,fontsize=12)
    p.xlabel('SOM Pixel Index (x)',fontsize=12)
    p.ylabel('SOM Pixel Index (y)',fontsize=12)  
    
def K2Histograms_1(planet_prob,groups_test):
    f, (ax1, ax2) = p.subplots(2, sharex=True)
    n1, bins1, patches1 = ax1.hist(planet_prob[groups_test==0],bins=40,color='#33FFFF')
    #ax1.hist(planet_prob_snrcut[groups_testsnrcut==0],bins1,color='#0000FF')
    n2, bins2, patches2 = ax2.hist(planet_prob[groups_test==1],bins=40,color='#80FF00')
    #ax2.hist(planet_prob_snrcut[groups_testsnrcut==1],bins2,color='#006600')
    ax1.plot([0.5,0.5],[0,70],'k--')
    ax2.plot([0.5,0.5],[0,8],'k--')
    p.setp(ax2.get_yticklabels()[-1],visible=False)
    #ax2.set_xlim(0.,1.0)
    ax1.set_ylim(0,70)
    ax2.set_ylim(0.,8)
    f.subplots_adjust(hspace=0)
    ax2.set_xlabel(r'$\theta_1$',fontsize=12)
    ax1.set_ylabel('N',fontsize=14)
    ax2.set_ylabel('N',fontsize=14)
    
def K2Histograms_2(planet_prob,groups_test):
    f, (ax1, ax2) = p.subplots(2, sharex=True)
    n1, bins1, patches1 = ax1.hist(planet_prob[groups_test==0],bins=40,color='#33FFFF')
    #ax1.hist(planet_prob_snrcut[groups_testsnrcut==0],bins1,color='#0000FF')
    n2, bins2, patches2 = ax2.hist(planet_prob[groups_test==1],bins=40,color='#80FF00')
    #ax2.hist(planet_prob_snrcut[groups_testsnrcut==1],bins2,color='#006600')
    ax1.plot([0.5,0.5],[0,18],'k--')
    ax2.plot([0.5,0.5],[0,10],'k--')
    p.setp(ax2.get_yticklabels()[-1],visible=False)
    ax2.set_xlim(0.1,0.9)
    ax1.set_ylim(0,18)
    ax2.set_ylim(0.,10)
    f.subplots_adjust(hspace=0)
    ax2.set_xlabel(r'$\theta_2$',fontsize=12)
    ax1.set_ylabel('N',fontsize=14)
    ax2.set_ylabel('N',fontsize=14)
    
def PASTISSOMLocPlot(mappedPASTIS,groupsPASTIS):
    colours = ['m','r','g','k','b','c','k']
    sizes = [3.0,3.0,3.0,3.0,3.0,3.0,3.0]
    
    p.figure(11)
    p.clf()
    for obj in range(len(groupsPASTIS)):
        map = mappedPASTIS[obj]
        type = colours[groupsPASTIS[obj].astype('int')]
        size = sizes[groupsPASTIS[obj].astype('int')]
        #if (groups_SNRcut[obj] != 4) & (groups_SNRcut[obj] != 2) :
        #if (groups[obj] != 1) :
        p.plot(map[0]+np.random.uniform(-0.5,0.5),map[1]+np.random.uniform(-0.5,0.5),type+'.',markersize=size)
    p.xlim(-0.5,19.5)
    p.ylim(-0.5,19.5)
    p.xlabel('SOM Pixel Index (x)',fontsize=14)
    p.ylabel('SOM Pixel Index (y)',fontsize=14)

def PASTISConfMap(confmatrix):
    norms = np.sum(confmatrix,axis=1)

    for i in range(len(confmatrix[:,0])):
        confmatrix[i,:] /= norms[i]

    p.figure(12)
    p.clf()
    p.imshow(confmatrix,interpolation='nearest',origin='lower',cmap='YlOrRd')

    #box labels
    for x in range(len(confmatrix[:,0])):
        for y in range(len(confmatrix[:,0])):
            if confmatrix[y,x] > 0.05:
                if confmatrix[y,x]>0.7:
                    p.text(x,y,str(np.round(confmatrix[y,x],decimals=3)),va='center',ha='center',color='w')
                else:
                    p.text(x,y,str(np.round(confmatrix[y,x],decimals=3)),va='center',ha='center')

    #plot grid lines (using p.grid leads to unwanted offset)
    for x in [0.5,1.5,2.5,3.5,4.5]:
        p.plot([x,x],[-0.5,6.5],'k--')
    for y in [0.5,1.5,2.5,3.5,4.5]:
        p.plot([-0.5,6.5],[y,y],'k--')
    p.xlim(-0.5,5.5)
    p.ylim(-0.5,5.5)
    p.xlabel('Predicted Class')
    p.ylabel('True Class')

    #class labels
    p.xticks([0,1,2,3,4,5],['Planet', 'EB', 'ET', 'PSB','BEB', 'BTP'],rotation='vertical')
    p.yticks([0,1,2,3,4,5],['Planet', 'EB', 'ET', 'PSB','BEB', 'BTP'])

    