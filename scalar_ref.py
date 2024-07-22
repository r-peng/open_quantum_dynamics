import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,6.4)})

nsamples = 500
every = 50

N = 300 
epsilons = [0.001,0.002,0.004,0.008,0.01,0.015,0.02]#,0.04,0.08] 
#epsilons = [0.001,0.01,0.02,0.04,0.08] 
colors = 'r','g','b','y','c','orange','pink','grey','olive'
fig1,ax1 = plt.subplots(nrows=1,ncols=1)
fig2,ax2 = plt.subplots(nrows=1,ncols=1)
fig3,ax3 = plt.subplots(nrows=1,ncols=1)
fig4,ax4 = plt.subplots(nrows=1,ncols=1)
for eps,color in zip(epsilons,colors):
    epsilon = eps 
    print('epsilon=',epsilon)
    errs = []
    rhos = []
    for isample in range(nsamples):
        if isample%every==0:
            print('isample=',isample)
        U_N = numpy.random.normal(loc=1.0, scale=epsilon, size=N + 2)
        K_N = numpy.zeros(N + 1)
        K_N[0] = U_N[1] - 1
        for i in range(1, N + 1):
            K_N[i] = U_N[i + 1] - U_N[i]
            for m in range(2, i + 2):
                K_N[i] -= K_N[i + 1 - m] * U_N[m - 1]
        errs.append(K_N)

        rho = numpy.ones(N+1)
        for i in range(1,N+1):
            rho[i] = rho[i-1]
            for m in range(1,i+1):
                rho[i] += K_N[i-m] * rho[m-1] 
        rhos.append(rho-numpy.ones(N+1))
    
    mean = numpy.log10(numpy.fabs(numpy.mean(errs,axis=0)))
    #std = numpy.log10(numpy.std(errs,axis=0))
    std = numpy.std(errs,axis=0)
    x = range(N+1) 
    ax1.plot(x,mean,linestyle='-',color=color,label=f'eps={eps}')
    ax2.plot(x,std,linestyle='-',color=color,label=f'eps={eps}')

    mean = numpy.log10(numpy.fabs(numpy.mean(rhos,axis=0)))
    std = numpy.std(rhos,axis=0)
    ax3.plot(x,mean,linestyle='-',color=color,label=f'eps={eps}')
    ax4.plot(x,std,linestyle='-',color=color,label=f'eps={eps}')
ax1.set_xlabel('N')
ax1.set_ylabel(f'log10(abs(mean))')
ax2.set_xlabel('N')
#ax2.set_ylabel(f'log10(std)')
ax2.set_ylabel(f'std')
ax3.set_xlabel('N')
ax3.set_ylabel(f'log10(abs(mean))')
ax4.set_xlabel('N')
ax4.set_ylabel(f'std')
plt.subplots_adjust(left=0.2, bottom=0.1, right=0.99, top=0.95)
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
fig1.savefig(f'scalar_ref_mean.png')
fig2.savefig(f'scalar_ref_std.png')
fig3.savefig(f'scalar_rho_mean.png')
fig4.savefig(f'scalar_rho_std.png')

