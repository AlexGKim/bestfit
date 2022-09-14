import numpy
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')

filename = '/Users/akim/Downloads/template_links.csv'

data = numpy.loadtxt(filename,delimiter = ",",skiprows=1)
data = data[:,2:]


# iteration 1

mn = numpy.mean(data,axis=0)
cov = numpy.cov(data,rowvar=False)
invcov = numpy.linalg.pinv(cov)
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
invcov = numpy.linalg.pinv(cov)
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
x0[len(x0)//2:] = cov.diagonal()+0.1
res = minimize(lambda x: -(multivariate_normal.logpdf(data, mean= x[:len(x)//2], cov=x[len(x)//2:])).sum(),x0)



print(data)