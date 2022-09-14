import numpy
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')

filename = '/Users/akim/Downloads/template_links.csv'
data = numpy.loadtxt(filename,delimiter = ",",skiprows=1)


# Thien I think index L1[10] 9, L2[10] 19, f[x,x] 62 and m1[10:16] -1 cen be removed
data = data[:,2:-1]
data = numpy.delete(data,62,axis=1)
data = numpy.delete(data,19,axis=1)
data = numpy.delete(data,9,axis=1)

# iteration 1

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

# Look OK?

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

# Gaussfit
x0=numpy.zeros(2*data.shape[1])
x0[:len(x0)//2] = mn
x0[len(x0)//2:] = cov.diagonal()
res = minimize(lambda x: -(multivariate_normal.logpdf(data, mean= x[:len(x)//2], cov=numpy.abs(x[len(x)//2:]))).sum(),x0)



print(data)