import numpy
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('macOSX')

filename = '/Users/akim/Downloads/template_links.csv'
data = numpy.loadtxt(filename,delimiter = ",",skiprows=1)


# I think index L1[10] 9, L2[10] 19, f[x,x] 62 and m1[10:16] -1 cen be removed
data = data[:,2:]
data = numpy.delete(data,62,axis=1)
# data = numpy.delete(data,19,axis=1)
# data = numpy.delete(data,9,axis=1)

# Get rid of outliers

## Pass 1:  Get rid of outliers in all parameters

mn = numpy.mean(data,axis=0)
cov = numpy.cov(data,rowvar=False)
invcov = numpy.linalg.inv(cov)
insider = []
for i in range(data.shape[0]):
	d = data[i,:]-mn
	insider.append(d @ invcov @ d)

insider = numpy.array(insider)
# plt.hist(insider)
# plt.show()

cut1 = 400
w = insider < cut1
data = data[w,:]

## Pass 2: Get rid of outliers in the most important parameters
ndim = 50

mn = numpy.mean(data,axis=0)
cov = numpy.cov(data,rowvar=False)

evalu, evec = numpy.linalg.eig(cov)
data_red = (data-mn) @ evec[:,:ndim]

insider = []
for i in range(data.shape[0]):
	insider.append((data_red[i,:]**2 / evalu[:ndim]).sum())

insider = numpy.array(insider)
# plt.hist(insider)
# plt.show()
cut1 = 70
w = insider < cut1
data = data[w,:]

# Find the peak

## transform the data

mn = numpy.mean(data,axis=0)
cov = numpy.cov(data,rowvar=False)

evalu, evec = numpy.linalg.eig(cov)
data_red = ((data-mn) @ evec)/numpy.sqrt(evalu)

## determine centers
clustering = MeanShift().fit(data_red)

## transform back
ans = clustering.cluster_centers_[0,:]*numpy.sqrt(evalu)+mn

# ans=numpy.insert(ans,9,numpy.sqrt(1-(ans[:9]**2).sum()))
# ans=numpy.insert(ans,19,numpy.sqrt(1-(ans[10:19]**2).sum()))
ans=numpy.insert(ans,62,1)
# ans=numpy.append(ans,numpy.sqrt(160-ans[-159:].sum()))

print(ans)