n = 10000
k = 10000
Mn_sample = np.zeros((k,1))
for i in range(k):
    Mn = np.max(np.random.exponential(size = n))
    Mn_sample[i] = Mn
plt.hist(Mn_sample, cumulative=True, normed=True,\
 bins = 80, label='Histogram of Mn samples')
x = np.arange(0,25,0.1)
G = lambda x : np.exp(x)
y = G(-G(-x + np.log(n)))
plt.plot(x,y, label = 'Analytic distribution')
