import numpy
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('macOSX')

filename = '/Users/akim/Downloads/template_links.csv'
data = numpy.loadtxt(filename,delimiter = ",",skiprows=1)


# I think index L1[10] 9, L2[10] 19, f[x,x] 62 and m1[10:16] -1 cen be removed
data = data[:,2:-1]
data = numpy.delete(data,62,axis=1)
data = numpy.delete(data,19,axis=1)
data = numpy.delete(data,9,axis=1)

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
ndim = 25

mn = numpy.mean(data,axis=0)
cov = numpy.cov(data,rowvar=False)

# Looking at the eigenvalues of cov, seems like there is information in the first 50 or so
evalu, evec = numpy.linalg.eig(cov)
data_red = (data-mn) @ evec[:,:ndim]

insider = []
for i in range(data.shape[0]):
	insider.append((data_red[i,:]**2 / evalu[:ndim]).sum())

insider = numpy.array(insider)
# plt.hist(insider)
# plt.show()
cut1 = 40
w = insider < cut1
data = data[w,:]

# Gaussfit
## Fit the most important parameters first

mn = numpy.mean(data,axis=0)
cov = numpy.cov(data,rowvar=False)

evalu, evec = numpy.linalg.eig(cov)
data_red = (data-mn) @ evec[:,:ndim]

x0=numpy.zeros(2*data_red.shape[1])
x0[len(x0)//2:] = evalu[ndim]
res = minimize(lambda x: -(multivariate_normal.logpdf(data_red, mean= x[:len(x)//2], cov=numpy.abs(x[len(x)//2:]))).sum(),x0)


print(res)