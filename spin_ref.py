import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,6.4)})

nsamples = 300
every = 50

N = 100 
sys_size = 2
plot_rho = False
plot_rho = True 
if plot_rho:
    epsilons = [0.001,0.002,0.004,0.008,0.01,0.015,0.02]#,0.04,0.08] 
else:
    epsilons = [0.001,0.01,0.02,0.04,0.08] 
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
        U_N = numpy.random.normal(scale=epsilon, size=(N + 2,sys_size,sys_size))
        U_N += numpy.array([numpy.eye(sys_size) for _ in range(N+2)])
        K_N = numpy.zeros((N + 1,sys_size,sys_size))
        K_N[0] = U_N[1] - numpy.eye(sys_size) 
        for i in range(1, N + 1):
            K_N[i] = U_N[i + 1] - U_N[i]
            for m in range(2, i + 2):
                K_N[i] -= numpy.dot(K_N[i + 1 - m],U_N[m - 1])
        errs.append(K_N)

        rho = numpy.ones((N+1,sys_size,sys_size))
        for i in range(1,N+1):
            rho[i] = rho[i-1]
            for m in range(1,i+1):
                rho[i] += numpy.dot(K_N[i-m],rho[m-1])
        rhos.append(rho-numpy.ones((N+1,sys_size,sys_size)))
    
    x = range(N+1) 
    mean_err = numpy.mean(errs,axis=0)
    std_err = numpy.std(errs,axis=0)
    mean_rho = numpy.mean(rhos,axis=0)
    std_rho = numpy.std(rhos,axis=0)
    if plot_rho:
        for dat,ax in zip([mean_err,mean_rho],(ax1,ax3)):
            dat = numpy.log10(numpy.array([numpy.linalg.norm(di) for di in dat]))
            ax.plot(x,dat,linestyle='-',color=color,label=f'eps={eps}')
        for dat,ax in zip([std_err,std_rho],(ax2,ax4)):
            dat = numpy.array([numpy.linalg.norm(di) for di in dat])
            ax.plot(x,dat,linestyle='-',color=color,label=f'eps={eps}')
    else:
        for dat,ax in zip([mean_err,std_err,mean_rho,std_rho],(ax1,ax2,ax3,ax4)):
            dat = numpy.log10(numpy.array([numpy.linalg.norm(di) for di in dat]))
            ax.plot(x,dat,linestyle='-',color=color,label=f'eps={eps}')
    
ax1.set_xlabel('N')
ax1.set_ylabel(f'log10(abs(mean))')
ax2.set_xlabel('N')
if plot_rho:
    ax2.set_ylabel(f'std')
else:
    ax2.set_ylabel(f'log10(std)')
ax3.set_xlabel('N')
ax3.set_ylabel(f'log10(abs(mean))')
ax4.set_xlabel('N')
ax4.set_ylabel(f'std')
plt.subplots_adjust(left=0.25, bottom=0.1, right=0.99, top=0.95)
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
if plot_rho: 
    fig1.savefig(f'spin_ref_mean.png')
    fig2.savefig(f'spin_ref_std.png')
    fig3.savefig(f'spin_rho_mean.png')
    fig4.savefig(f'spin_rho_std.png')
else:
    fig1.savefig(f'spin_ref_mean_.png')
    fig2.savefig(f'spin_ref_std_.png')

