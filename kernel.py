import numpy as np
import scipy.special
import scipy.linalg
import itertools
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
colors = 'r','g','b','y','c','orange','pink','grey','olive'

def get_density_matrix(size,check=True):
    rho = np.random.rand(size)
    rho /= rho.sum() # eigenvalues
    D = np.random.rand(size,size)
    D -= D.T
    D = scipy.linalg.expm(D)
    rho = np.dot(D*rho,D.T)
    check_property(rho)
    return rho 
def check_property(rho,thresh=1e-6):
    print('check hermiticity...')
    assert np.linalg.norm(rho-rho.T.conj())/np.linalg.norm(rho)<thresh
    print('check positivity...')
    w = np.linalg.eigvalsh(rho) 
    try:
        assert len(w[w>0])==len(w)
    except:
        print(w)
    print('check trace...')
    assert abs(w.sum()-1)<thresh
def get_full_H(ls):
    H = ls[0]
    n = H.shape[0]
    for Hi in ls[1:]:
        H = np.einsum('...,pq->...pq',H,Hi)
        n *= Hi.shape[0]
    ax = [2*i for i in range(len(ls))] + [2*i+1 for i in range(len(ls))]
    return H.transpose(*ax).reshape(n,n)
class Hamiltonian:
    def __init__(self,size=(2,3)):
        self.ns,self.nb = size[0],np.prod(np.array(size[1:]))
        self.size = size
        self.U = None
        self.D = None
        self.Hb = [None] * (len(size) - 1)
        for i,ni in enumerate(size):
            if i==0:
                ls = [np.eye(nj) for nj in size]
                ls[0] = self.get_Hs()
                self.H = get_full_H(ls) 
                continue

            ls = [np.eye(nj) for nj in size]
            ls[i] = self.get_Hb(i) 
            self.H += get_full_H(ls) 

            ls = [np.eye(ni) for ni in size]
            ls[0] = self.get_S(i) 
            ls[i] = self.get_B(i) 
            self.H += get_full_H(ls) 

        self.Ls = np.einsum('pr,qs->pqrs',self.Hs,np.eye(self.ns))
        self.Ls -=np.einsum('pr,qs->pqrs',np.eye(self.ns),self.Hs)
        self.Ls = self.Ls.reshape(self.ns**2,self.ns**2)
    def get_Hs(self):
        self.Hs = np.random.rand(self.ns,self.ns) * 2 - 1
        self.Hs += self.Hs.T
        return self.Hs
    def get_Hb(self,i):
        Hb = np.random.rand(self.size[i],self.size[i]) * 2 - 1
        Hb += Hb.T
        self.Hb[i-1] = Hb
        return Hb
    def get_S(self,i):
        if self.D is None:
            self.D = np.random.rand(self.ns,self.ns)
            self.D -= self.D.T
            self.D = scipy.linalg.expm(self.D)
        S = np.random.rand(self.ns)
        return np.dot(self.D*S,self.D.T)
    def get_B(self,i):
        B = np.random.rand(self.size[i],self.size[i]) * 2 - 1
        B += B.T
        return B
    def get_U(self,delta):
        H = self.H
        if len(self.H.shape)>2:
            n = self.ns * self.nb
            H = H.reshape(n,n)
        self.U = scipy.linalg.expm(-H*1j*delta)
    def get_rho_bath(self,beta):
        ls = [scipy.linalg.expm(-beta*Hi) for Hi in self.Hb]
        ls = [rho/np.trace(rho) for rho in ls]
        return get_full_H(ls) 
class SpinBoson(Hamiltonian):
    def __init__(self,hx=1,hz=1,**kwargs):
        self.hx,self.hz = hx,hz
        super().__init__(**kwargs)
    def get_Hs(self):
        self.Hs = self.hx * np.array([[0,1.],[1,0]]) 
        self.Hs += self.hz * np.array([[1.,0],[0,-1]])
        self.Hs *= .5 
        return self.Hs
    def get_Hb(self,i):
        self.Hb[i-1] = np.diag(np.random.rand(self.size[i]))
        return self.Hb[i-1]
    def get_S(self,i):
        return .5 * np.array([[1,0],[0,-1]])
    def get_B(self,i):
        B = np.zeros((self.size[i],self.size[i]))
        for i in range(self.size[i]-1):
            B[i,i+1] = B[i+1,i] = np.random.rand() 
        return B
class Propagator:
    def __init__(self,ham=None,size=(2,3),rho_bath0=None,delta=1,beta=1):
        self.delta = delta
        if ham is None:
            ham = Hamiltonian(size=size)
        self.ham = ham
        self.ham.get_U(delta)
        self.ns,self.nb = ham.ns,ham.nb
        if rho_bath0 is None:
            self.rho_bath0 = ham.get_rho_bath(beta) 

        self.Ufull = {0:np.eye(self.ns*self.nb)}
        self.U = {0:self.get_Us(self.Ufull[0])}
        self.K = dict()
    def get_Us(self,U):
        if len(U.shape)==2:
            U = U.reshape(self.ns,self.nb,self.ns,self.nb)
        U = np.einsum('pqrs,PqRS,sS->pPrR',U,U.conj(),self.rho_bath0)
        return U.reshape(self.ns**2,self.ns**2)
    def _L(self,U):
        return U - self.delta * 1j * np.dot(self.ham.Ls,U)
    def propagate(self,N):
        if N in self.K:
            return
        self.Ufull[N+1] = np.dot(self.ham.U,self.Ufull[N])
        self.U[N+1] = self.get_Us(self.Ufull[N+1])

        self.K[N] = self.U[N+1] - self._L(self.U[N])
        for m in range(2,N+2):
            self.K[N] -= self.delta**2*np.dot(self.K[N+1-m],self.U[m-1]) 
        self.K[N] /= self.delta**2
class MeasuredPropagator(Propagator):
    def __init__(self,eps,**kwargs):
        super().__init__(**kwargs)
        self.eps = eps

        self.E = {0:np.zeros((self.ns**2,)*2)}
        self.U_appr = {0:self.U[0]}
        self.K_appr = dict() 
    def propagate(self,N):
        super().propagate(N)
        if N in self.K_appr:
            return
        self.E[N+1] = np.random.normal(size=(self.ns**2,)*2)
        self.E[N+1] = self.E[N+1] + 1j*np.random.normal(size=(self.ns**2,)*2)
        self.U_appr[N+1] = self.U[N+1]+self.eps*self.E[N+1]

        self.K_appr[N] = self.U_appr[N+1] - self._L(self.U_appr[N])
        for m in range(2,N+2):
            self.K_appr[N] -= self.delta**2*np.dot(self.K_appr[N+1-m],self.U_appr[m-1]) 
        self.K_appr[N] /= self.delta**2
class PropagatedError(MeasuredPropagator):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.F = dict() 

        self.X = dict()
        self.Y = dict()
        self.Z = dict()
        self.C = dict() 
        self.D = dict()
    def get_X(self,N):
        XN = self.E[N+1] - self.E[N] - self._L(self.E[N])
        if N>0:
            XN += self._L(self.E[N-1])
        return XN
    def get_Y(self,N):
        YN = -self.E[N].copy()
        if N>0:
            YN += self.E[N-1]
        return YN
    def get_Z(self,N):
        if N==0:
            return 0
        return 1j * np.dot(self.ham.Ls,self.U[N-1])
    def get_C(self,N):
        CN = np.zeros((self.ns**2,)*2,dtype=complex)
        if N>0:
            CN -= np.dot(self.K[N-1],self.E[1])
        for m in range(2,N+1):
            CN -= np.dot(self.K[N-m],self.E[m]-self.E[m-1])
        return CN 
    def get_D(self,N):
        DN = np.zeros((self.ns**2,)*2,dtype=complex)
        for n in range(1,N+1):
            DN -= np.dot(self.K[N-n],self.U[n-1])
        return DN
    def get_matrix(self,typ,N):
        if typ=='U':
            return self.U[N]
        if typ=='K':
            return self.K[N]
        saved = {'X':self.X,'Y':self.Y,'Z':self.Z,'C':self.C,'D':self.D}[typ]
        try:
            return saved[N]
        except:
            pass
        fxn = {'X':self.get_X,'Y':self.get_Y,'Z':self.get_Z,'C':self.get_C,'D':self.get_D}[typ]
        saved[N] = fxn(N)
        return saved[N] 
    def propagate(self,N):
        super().propagate(N)
        if N in self.F:
            return
        XN = self.get_X(N)
        CN = self.get_C(N)
        self.F[N] = self.eps * (XN + self.delta**2 * CN)
        for m in range(1,N+1):
            Ym = self.get_Y(m)
            Zm = self.get_Z(m)
            Dm = self.get_D(m)

            tmp = self.eps * Ym + self.delta * Zm + self.delta**2 * Dm
            self.F[N] += self.delta**2 * np.dot(self.F[N-m],tmp)
        self.F[N] /= self.delta**2
        assert np.linalg.norm(self.K_appr[N]-self.K[N]-self.F[N])/np.linalg.norm(self.F[N])<1e-6
class Error:
    def __init__(self,N,max_order=None):
        self.N = N
        self.max_order = max_order
        self.orders = dict() 
    def add(self,order,coeff):
        if self.max_order is not None:
            if sum(order) > self.max_order:
                return
        if np.linalg.norm(coeff)<1e-10:
            return
        if order in self.orders:
            self.orders[order] += coeff
        else:
            self.orders[order] = coeff.copy()
class OrderedPropagatedError(PropagatedError):
    def __init__(self,max_order=None,**kwargs):
        super().__init__(**kwargs)
        self.max_order = max_order
        self.ordered_F = dict() 
    def propagate(self,N):
        super().propagate(N)
        if N in self.ordered_F:
            return
        FN = Error(N,max_order=self.max_order)
        XN = self.get_X(N)
        CN = self.get_C(N)
        FN.add((0,1),XN)
        FN.add((2,1),CN)
        for m in range(1,N+1):
            Ym = self.get_Y(m)
            Zm = self.get_Z(m)
            Dm = self.get_D(m)
            for (p,q),Fpq in self.ordered_F[N-m].orders.items():
                FN.add((p,q+1),np.dot(Fpq,Ym))
                FN.add((p+1,q),np.dot(Fpq,Zm)) 
                FN.add((p+2,q),np.dot(Fpq,Dm)) 
        self.ordered_F[N] = FN
        err = np.zeros((self.ns**2,)*2,dtype=complex)
        for (p,q),Fpq in FN.orders.items():
            err += self.delta**p * self.eps**q * Fpq
        assert np.linalg.norm(err/self.delta**2-self.F[N])/np.linalg.norm(self.F[N])<1e-6
    def _permute(self,typs,lens,m):
        string = ''
        for typ,l in zip(typs,lens):
            string += ''.join([typ] * l)
        ls = set(itertools.permutations(string,len(string)))
        _sum = np.zeros((self.ns**2,)*2,dtype=complex)
        for p in ls:
            pd = np.eye(self.ns**2)
            for typ,mi in zip(p,m):
                pd = np.dot(pd,self.get_matrix(typ,mi))
            _sum += pd 
        return _sum
    def get_sum_permute(self,N,typ1,typs,lens):
        l = sum(lens)
        FN = np.zeros((self.ns**2,)*2,dtype=complex)
        mrange = range(1,N-(l-1)+1)
        ct = 0
        for m in itertools.product(mrange,repeat=l):
            _sum = sum(m)
            if _sum>N:
                continue
            ct += 1
            FN += np.dot(self.get_matrix(typ1,N-_sum),self._permute(typs,lens,m))
        return FN,ct
    def get_order(self,p,q,N):
        k,r = p//2, p%2
        FN = np.zeros((self.ns**2,)*2,dtype=complex)
        ct = 0
        for i in range(k+1):
            nz = 2*(k-i) if r==0 else 2*(k-i)+1
            FNi,cti = self.get_sum_permute(N,'X',['D','Z','Y'],[i,nz,q-1])
            FN += FNi
            ct += cti
        for i in range(k):
            nz = 2*(k-1-i) if r==0 else 2*(k-1-i)+1
            FNi,cti = self.get_sum_permute(N,'C',['D','Z','Y'],[i,nz,q-1])
            FN += FNi
            ct += cti
        return FN,ct

size = 2,3,2
check_phys = False
check_phys = True 
if check_phys:
    #ham = Hamiltonian(size=size)
    ham = SpinBoson(size=size)
    prop = Propagator(ham=ham)
    rho = get_density_matrix(size[0])
    Us = prop.get_Us(ham.U)
    rho = np.dot(Us,rho.flatten())
    check_property(rho.reshape(size[0],size[0]))
    rho -= prop.delta * np.dot(-1j*ham.Ls,rho)
    check_property(rho.reshape(size[0],size[0]))
    #exit()

check = True
check = False
if check:
    max_order = None 
    Nmax = 8 

    #ham = Hamiltonian(size=size)
    ham = SpinBoson(size=size)
    kn = OrderedPropagatedError(ham=ham,eps=0.13,delta=.27)
    for N in range(Nmax+1):
        print('N=',N)
        kn.propagate(N)
    for N in range(Nmax+1): 
        print('\nN=',N)
        for (p,q),F1 in kn.ordered_F[N].orders.items():
            print('p,q=',p,q)
            F2,ct = kn.get_order(p,q,N) 
            if np.linalg.norm(F1)<1e-6:
                assert np.linalg.norm(F2)<1e-6
            else:
                assert np.linalg.norm(F1-F2)/np.linalg.norm(F1)<1e-6
    exit()

plot_XYZ = True
plot_XYZ = False
if plot_XYZ:
    Nmax = 100
    Ns = range(Nmax+1)
    typs = 'X','Y','Z','K','C','D'
    runs = range(200)
    every = 1 
    data = np.zeros((len(runs),len(typs),Nmax+1))
    for run in runs:
        if run%every==0:
            print('run=',run)
        #ham = Hamiltonian(size=size)
        ham = SpinBoson(size=size)
        kn = PropagatedError(ham=ham,eps=0.13,delta=0.27)
        for N in Ns:
            kn.propagate(N)
        for ix,typ in enumerate(typs): 
            for N in Ns:
                data[run,ix,N] = np.linalg.norm(kn.get_matrix(typ,N))
    data = np.quantile(data,(0.25,0.5,0.75),axis=0)
    for ix,typ in enumerate(typs):
        y = data[:,ix,:]
        fig,ax = plt.subplots(nrows=1,ncols=1)
        ax.plot(Ns,y[1],linestyle='-')
        ax.fill_between(Ns,y[0],y[2],alpha=0.2)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.95)
        ax.set_xlabel('N')
        ax.set_ylabel(typ+'norm')
        if typ=='K':
            ax.set_yscale('log')
        fig.savefig(typ+'.png')
    exit()


#plot_eps = False 
if plot_eps:
    Nmax = 50
    Ns = range(Nmax+1)
    runs = range(20)
    every = 1 
    param = (0.0001,0.00001),
    fig,ax = plt.subplots(nrows=1,ncols=1)
    for (deli,epsi),color in zip(eps,colors):
        print('epsilon=',eps)
        ls = []
        for run in runs: 
            if run%every==0:
                print('run=',run)
            kn = Kernels(sys_size=sys_size,max_order=max_order,delta=eps)
            for N in range(Nmax+1):
                kn.get_next_full()
            ls.append([np.linalg.norm(Ki) for Ki in kn.K_full])
            print(kn.K_full)
            exit()

        plt.rcParams.update({'figure.figsize':(6.4,6.4)})
        y = np.log10(np.quantile(np.array(ls),(0.25,0.5,0.75),axis=0))
        x = range(y.shape[1])
        ax.plot(x,y[1],linestyle='-',color=color,label=f'eps={epsi}')
        ax.fill_between(x,y[0],y[2],color=color,alpha=0.2)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.95)
    ax.set_xlabel('N')
    ax.set_ylabel('log10(K_N)')
    ax.legend()
    fig.savefig(f'dynamic_kernel.png')

