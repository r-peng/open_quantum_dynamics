import numpy as np
import scipy.special
import itertools
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
colors = 'r','g','b','y','c','orange','pink','grey','olive'
# assume error distribute like gaussian

class Kernel:
    def __init__(self,N,max_order=None):
        self.N = N
        self.max_order = N+1 if max_order is None else max_order
        self.orders = [0]

    def add(self,order,coeff):
        if order > self.max_order:
            return
        if order<len(self.orders):
            self.orders[order] += coeff
        else:
            assert order==len(self.orders)
            self.orders.append(coeff)
class Kernels:
    def __init__(self,max_order=2):
        self.max_order = max_order
        self.rng = np.random.default_rng()
        self.eps = [0,self.rng.normal()]

        K0 = Kernel(0,max_order=self.max_order)
        K0.add(1,self.eps[1])
        self.Ks = [K0]
    def get_next(self):
        # K_{N} = U_{N+1} - U_{N} - \sum_{m=2}^{N+1}K_{N+1-m}U_{m-1}
        # U_m = 1+eps_m

        N = len(self.Ks) 
        self.eps.append(self.rng.normal())

        KN = Kernel(N,max_order=self.max_order) 
        KN.add(1,self.eps[N+1])
        KN.add(1,-self.eps[N])

        for m in range(2,N+2):
            eps_m = self.eps[m-1]
            for p,Kp in enumerate(self.Ks[N+1-m].orders):
                KN.add(p,-Kp)
                KN.add(p+1,-Kp*eps_m)
                
        self.Ks.append(KN) 
    def get_O1(self,N):
        return self.eps[N+1] - 2*self.eps[N] + self.eps[N-1]
    def _get_O2(self,N):
        err = 0
        if N%2==0:
            k = N//2
            if k-1>0:
                err += self.eps[k-1]**2
            if k>0:
                err += 3 * self.eps[k]**2
            for i in range(1,k+1):
                if i<2*k-i+1: 
                    err -= 2 * self.eps[i] * self.eps[2*k-i+1]
                if i<2*k-i:
                    err += 6 * self.eps[i] * self.eps[2*k-i] 
                if i<2*k-i-1:
                    err -= 6 * self.eps[i] * self.eps[2*k-i-1]
                if i<2*k-i-2:
                    err += 2 * self.eps[i] * self.eps[2*k-i-2]
        else:
            k = (N+1)//2
            if k-1>0:
                err -= 3 * self.eps[k-1] ** 2 
            if k>0:
                err -= self.eps[k] **2
            for i in range(1,k+1):
                if i<2*k-i:
                    err -= 2 * self.eps[i] * self.eps[2*k-i]
                if i<2*k-i-1:
                    err += 6 * self.eps[i] * self.eps[2*k-i-1] 
                if i<2*k-i-2:
                    err -= 6 * self.eps[i] * self.eps[2*k-i-2]
                if i<2*k-i-3:
                    err += 2 * self.eps[i] * self.eps[2*k-i-3]
        return err
    def get_O2(self,N):
        coeffs = -1,3,-3,1
        err = 0
        for idx,cix in enumerate(coeffs):
            for m in range(1,N-idx+1):
                err += self.eps[N+1-idx-m] * self.eps[m] * cix
        return err
    def get_O3(self,N):
        coeffs = 1,-4,6,-4,1
        err = 0
        for idx,cix in enumerate(coeffs):
            eix = 0
            for m in range(1,N-idx):
                epsm = self.eps[m]
                #for n in range(1,N-idx+1-m):
                for n in range(1,N-idx):
                    if m+n>N-idx:
                        continue
                    epsn = self.eps[n]
                    eix += self.eps[N+1-idx-n-m] * epsm * epsn
            err += cix * eix
        return err
    def get_O4(self,N):
        coeffs = -1,5,-10,10,-5,1
        err = 0
        for idx,cix in enumerate(coeffs):
            eix = 0
            for m in range(1,N-idx):
                epsm = self.eps[m]
                #for n in range(1,N-idx+1-m):
                for n in range(1,N-idx):
                    epsn = self.eps[n]
                    #for k in range(1,N-idx+1-m-n):
                    for k in range(1,N-idx):
                        if m+n+k>N-idx:
                            continue
                        epsk = self.eps[k]
                        eix += self.eps[N+1-idx-n-m-k] * epsm * epsn * epsk
            err += cix * eix
        return err
    def get_order(self,p,N):
        #if p==1:
        #    return self.get_O1(N)
        #if p==2:
        #    return self.get_O2(N)
        #if p==3:
        #    return self.get_O3(N)
        #if p==4:
        #    return self.get_O4(N)
        #raise NotImplementedError
        err = 0
        kmax = min(p+1,N+1-p)
        ct = [0 for _ in range(kmax+1)]  
        for k in range(kmax+1):
            coeff = scipy.special.binom(p+1,k) * (-1)**(p+k+1)

            Smax = N-k
            mrange = range(1,Smax-(p-2)+1)
            ek = 0
            for m in itertools.product(mrange,repeat=p-1):
                _sum = sum(m)
                if _sum>Smax:
                    continue
                ct[k] += 1
                pd = self.eps[N+1-k-_sum] 
                for mix in m:
                    pd *= self.eps[mix]
                ek += pd
            err += coeff * ek
        return err,np.array(ct)
def count1(N,p):
    if p==1:
        return 3,None
    err = 0
    kmax = min(p+1,N+1-p)
    ct = [0 for _ in range(kmax+1)]  
    for k in range(kmax+1):
        coeff = scipy.special.binom(p+1,k) 
        for s in range(p-1,N-k+1):
            ct[k] += scipy.special.binom(s-1,p-2)
        err += coeff * ct[k]
        ct_ = scipy.special.binom(N-k+1,p)
        assert ct[k] <= ct_
        assert ct_ <= ct[k] * (N-k)/(p-1)
    return err,np.array(ct)
def count2(N,p):
    if p==1:
        return 3,None
    err = 0
    kmax = min(p+1,N+1-p)
    ct = [0 for _ in range(kmax+1)]  
    for k in range(kmax+1):
        coeff = scipy.special.binom(p+1,k) 
        ct[k] = scipy.special.binom(N-k+1,p)
        err += coeff * ct[k]
    return err,np.array(ct)
def count3(N,p):
    return scipy.special.binom(N+1,p)*2**p
    
def approx(N,p):
    return 2 * (2*np.e*(N+1)/p)**p
          
check = True
if check:
    max_order = 10 
    Nmax = 10

    kn = Kernels(max_order=max_order)
    for N in range(1,Nmax+1):
        kn.get_next()
    errs = {order:[] for order in range(1,max_order+1)} 
    for N in range(1,Nmax+1): 
        K = kn.Ks[N]
        for order in range(1,max_order+1):
            if order>=len(K.orders):
                continue
            err1 = K.orders[order] #* eps**order
            err2,ct2 = kn.get_order(order,N) 
            _,ct3 = count1(N,order)
            if abs(err1)<1e-6:
                assert abs(err2)<1e-6
            else:
                assert abs(err1-err2)/abs(err1)<1e-6
            if ct3 is None:
                continue
            if np.linalg.norm(ct2)<1e-6:
                assert np.linalg.norm(ct3)<1e-6
            else:
                assert np.linalg.norm(ct2-ct3)/np.linalg.norm(ct2)<1e-6
            #err3 = count(N,order)
            #err4 = approx(N,order)
            #print('order=',order,err1,err2,err3,err4)

eps = 0.001
max_order = 8 
Nmax = 500
plot_eps = True
#plot_eps = False 
if plot_eps:
    runs = range(300)
    ls = []
    for run in runs: 
        print('\n run=',run)
        kn = Kernels(max_order=max_order)
        for N in range(Nmax+1):
            kn.get_next()
        
        err = {order:[] for order in range(1,max_order+1)} 
        for N in range(1,Nmax+1):
            K = kn.Ks[N]
            for order in range(1,max_order+1):
                if order>=len(K.orders):
                    continue
                err[order].append(K.orders[order])
        ls.append(err)

    plt.rcParams.update({'figure.figsize':(6.4,6.4)})
    fig1,ax1 = plt.subplots(nrows=1,ncols=1)
    fig2,ax2 = plt.subplots(nrows=1,ncols=1)
    for order in range(1,max_order+1):
        y = []
        for run in runs:
            y.append(ls[run][order])
        y = np.array(y)
        mean = np.log10(np.fabs(np.mean(y,axis=0)))
        std = np.log10(np.std(y,axis=0))
        n = y.shape[1]
        x = np.log10(range(Nmax+1-n,Nmax+1))
        ax1.plot(x,mean,linestyle='-',color=colors[order],label=f'O{order}')
        ax2.plot(x,std,linestyle='-',color=colors[order],label=f'O{order}')
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.95)
    ax1.set_xlabel('log10(N)')
    ax2.set_xlabel('log10(N)')
    ax1.set_ylabel('log10(abs(mean))')
    ax2.set_ylabel('log10(std)')
    ax1.legend()
    ax2.legend()
    fig1.savefig(f'scalar_gaussian_mean.png')
    fig2.savefig(f'scalar_gaussian_std.png')
    exit()

fig,ax = plt.subplots(nrows=1,ncols=1)
count_ = 4 
for order in range(1,max_order+1):
    print('order=',order)
    Ns = np.array(range(order,Nmax+1))
    x = np.log10(Ns)
    if count_==1:
        y = np.log10(np.array([count1(Ni,order)[0] for Ni in Ns]))
    if count_==2:
        y = np.log10(np.array([count2(Ni,order)[0] for Ni in Ns]))
    if count_==3:
        y = np.log10(np.array([count3(Ni,order) for Ni in Ns]))
    if count_==4:
        y = np.log10((Ns + 1) * 2 * np.e / order) * order
    ax.plot(x,y,linestyle='-',color=colors[order],label=f'O{order}')
plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.95)
ax.set_xlabel('log10(N)')
ax.set_ylabel(f'log10(count{count_})')
ax.legend()
fig.savefig(f'scalar_count{count_}.png')
