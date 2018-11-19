# hubblec.py    AJW, 9/21/13

import numpy as np
#import matplotlib 
import matplotlib.cm as cm
import matplotlib.pyplot as plt 

# compute the predicted SN magnitudes for a given cosmology
# http://en.wikipedia.org/wiki/Distance_measures_(cosmology)
def mzz(OmM,OmL,z):
    #Omk = 1.-OmM-OmL
    OI = OmM*(1+zz)**3+OmL
    #print ("IOmM,IOmL,OmM,OmL,Omk = ",IOmM,IOmL,OmM,OmL,Omk)
    # protect against sqrt(-1). Else, no Big Bang
    OI[OI<0.001] = 0.001
    # comoving distance
    dCz = np.cumsum(1./np.sqrt(OI))*dz
    # curvature cases, for transverse comoving distance:
    #if (Omk < 0.):
        #dMz = np.sin(np.sqrt(-Omk)*dCz)/np.sqrt(-Omk)
        #dMz[dMz < 0.0001] = 0.0001  # what if this goes negative?
    #elif (Omk > 0.):
        #dMz = np.sinh(np.sqrt(Omk)*dCz)/np.sqrt(Omk)
    #else:
        #dMz = dCz
    # convert luminosity distance into a magnitude: theory prediction, smooth curve
    mc = m0 + 5.0*np.log10(dH*dCz*(1+zz))
    ms = np.interp(z,zz,mc)
    return ms

# Data from SCP:
# http://supernova.lbl.gov/Union/
# http://arxiv.org/abs/1105.3470
# wget http://supernova.lbl.gov/Union/figures/SCPUnion2.1_mu_vs_z.txt
data = np.genfromtxt('SCPUnion2.1_mu_vs_z.txt')
zs = data.T[1]
mm = data.T[2]
dm = data.T[3]
dlabel = 'DATA'
ndof = mm.size

# vector of redshifts for smooth curves
dz = 0.001
zz = np.arange(dz,2.0,dz)

# estimate H0 and m0 from low-z SN:
c = 299792.   # km/s
i1 = zs<0.05
H0 = (c*1.e6/10.**(mm[i1]/5.+1)*zs[i1]).mean()
# horizon distance
dH = c/H0
ms = 5.0*np.log10(dH*zs)
# only use the nearby data with linear Hubble to determine m0:
m0 = np.sum((mm[i1]-ms[i1])/dm[i1]**2)/np.sum(1./dm[i1]**2)
print (dlabel,', redshifts ndof = ',zs.size, ', low redshifts ndof = ',zs[i1].size)
print (dlabel,', H0 = ', H0,', m0 = ', m0)

# construct a 2D grid in Om_M, Om_Lambda:
NOmM = 60
NOmL = 90
OmMs = np.linspace(0.0,1.0,NOmM)
OmLs = np.linspace(0.,1.5,NOmL)
X, Y = np.meshgrid(OmMs,OmLs)

# initialize a chisq surface in this space: 
Zmax = 800.
Z = np.ones((NOmL,NOmM))*Zmax

# loop over points (cosmologies) in this space
for IM in np.arange(NOmM):
  for IL in np.arange(NOmL):
    # get predicted magnitudes for this cosmology
    ms = mzz(OmMs[IM],OmLs[IL],zs)  
    # construct a chisq between the theory and the data
    chiv = (mm-ms)/dm
    chisq = np.sum(chiv[~i1]**2)
    # print OmM,OmL,Omk,chisq,ndof
    Z[IL,IM] = min(chisq,Zmax)

# find the minimum chisq
chimin = Z.min()
# and the values of the parameters at the minimum
am  = np.unravel_index(Z.argmin(), Z.shape)
OmM = X[am]
OmL = Y[am]
#Omk = 1.-OmM-OmL
print ('minimum at: ',OmM,OmL,chimin)
mc = mzz(OmM,OmL,zz)  

# find the contour of "no big bang"
b = []
for OmM in OmMs:
  for OmL in OmLs:
      if (np.sum((OmM*(1+zz)**3+OmL) < 0.)):
          b.append((OmM,OmL))
          break
          
b = np.array(b)

# at the minimum, plot the data and the prediction.
plt.figure()
plt.errorbar(zs,mm,xerr=dz,yerr=dm,fmt='+',label=dlabel)
plt.plot(zz,mc,'r',label='fit')
plt.xlim([0,2.0])
plt.xlabel('zs')
plt.ylabel('m_mod')
plt.grid(b=True,which='major')
plt.grid(b=True,which='minor')
plt.legend(loc='lower right')
plt.show()




# plot the chisq and contours
plt.figure()
im = plt.imshow(Z, interpolation='bilinear', origin='lower', \
                cmap=cm.jet, extent=(0.,1.0,0.,1.5))
plt.colorbar()
CS = plt.contour(X, Y, Z)
CS.levels = [chimin+1., chimin+2., chimin+3.]
plt.clabel(CS, CS.levels, inline=1, fontsize=10)
# plot the minimum
plt.plot(X[am],Y[am],'r*',markersize=20)
plt.plot([1.0, 0.0],[0.0,1.0],'k')
# plot the "no big bang" boundary
#plt.plot(b.T[0],b.T[1],linewidth=3)
plt.xlabel(r'$\Omega_M$',fontsize=18)
plt.ylabel(r'$\Omega_\Lambda$',fontsize=18)
plt.title('Countor plot in densities plane')
plt.show()


      
