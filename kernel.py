import numpy as np
import scipy.special
import scipy.linalg
import itertools
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
colors = 'r','g','b','y','c','orange','pink','grey','olive'

class Kernel:
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
class Kernels:
    def __init__(self,sys_size=2,max_order=None,eps=1,delta=1,decay=None):
        self.max_order = max_order
        self.sys_size = sys_size
        self.delta = delta
        self.eps = eps
        self.decay = decay
        self.rng = np.random.default_rng()
        
        self.Hs = self.rng.random(size=(self.sys_size,)*2) * 2 - 1
        self.Hs = (self.Hs + self.Hs.T) / 2

        self.U = [np.eye(self.sys_size,dtype=complex)]
        self.E = [np.zeros((self.sys_size,)*2)]
        self.U_appr = [self.U[0]]
        self.K = []
        self.K_appr = []
        self.F = []

        self.ordered_F = []
        self.X = dict()
        self.Y = dict()
        self.Z = dict()
        self.C = dict() 
        self.D = dict()
        # suppose know the set of true K's (which will be randomly generated)
        # then can get the set of true U's. 
        # from the true U's can get the measured U's
        # then can get approximate K and F
    def _decay(self,N):
        if self.decay is None:
            return 1
        raise NotImplementedError
    def _Ls(self,U):
        return 1j * (np.dot(self.Hs,U)-np.dot(U,self.Hs))
    def _L(self,U):
        return U - self.delta * self._Ls(U)
    def get_trueKnext(self):
        N = len(self.K)
        KN = self.rng.random(size=(self.sys_size,)*2) * 2 - 1
        self.K.append(self._decay(N)*KN)
    def get_trueUnext(self):
        N = len(self.U)
        UN = self._L(self.U[N-1]) 
        for m in range(1,N+1):
            UN += self.delta**2 * np.dot(self.K[N-m],self.U[m-1])
        self.U.append(UN)
        EN = self.rng.normal(size=(self.sys_size,)*2)
        EN = EN + 1j*self.rng.normal(size=(self.sys_size,)*2)
        self.U_appr.append(UN+self.eps*EN)
        self.E.append(EN)
    def get_approxKnext(self):
        N = len(self.K_appr)
        KN = self.U_appr[N+1] - self._L(self.U_appr[N])
        for m in range(2,N+2):
            KN -= self.delta**2 * np.dot(self.K_appr[N+1-m],self.U_appr[m-1]) 
        self.K_appr.append(KN/self.delta**2)
    def get_X(self,N):
        try:
            return self.X[N]
        except:
            pass
        XN = self.E[N+1] - self.E[N] - self._L(self.E[N])
        if N>0:
            XN += self._L(self.E[N-1])
        self.X[N] = XN
        return XN
    def get_C(self,N):
        try:
            return self.C[N]
        except:
            pass
        CN = np.zeros((self.sys_size,)*2,dtype=complex)
        if N>0:
            CN -= np.dot(self.K[N-1],self.E[1])
        for m in range(2,N+1):
            CN -= np.dot(self.K[N-m],self.E[m]-self.E[m-1])
        self.C[N] = CN
        return CN 
    def get_Y(self,N):
        try:
            return self.Y[N]
        except:
            pass
        YN = -self.E[N]
        if N>0:
            YN += self.E[N-1]
        self.Y[N] = YN
        return YN
    def get_Z(self,N):
        try:
            return self.Z[N]
        except:
            pass
        ZN = self._Ls(self.U[N-1])
        self.Z[N] = ZN
        return ZN
    def get_D(self,N):
        try:
            return self.D[N]
        except:
            pass
        DN = np.zeros((self.sys_size,2),dtype=complex)
        for n in range(1,N+1):
            DN -= np.dot(self.K[N-n],self.U[n-1])
        self.D[N] = DN
        return DN
    def get_matrix(self,typ,N):
        if typ=='X':
            return self.get_X(N)
        if typ=='Y':
            return self.get_Y(N)
        if typ=='Z':
            return self.get_Z(N)
        if typ=='C':
            return self.get_C(N)
        if typ=='D':
            return self.get_D(N)
    def get_next(self,full=True,order=False):
        self.get_trueKnext()
        self.get_trueUnext()
        self.get_approxKnext()

        N = len(self.F)
        if order:
            FN_ = Kernel(N,max_order=self.max_order)

        XN = self.get_X(N)
        CN = self.get_C(N)
        if full:
            FN = self.eps * (XN + self.delta**2 * CN)
        if order:
            FN_.add((0,1),XN)
            FN_.add((2,1),CN)
        for m in range(1,N+1):
            Ym = self.get_Y(m)
            Zm = self.get_Z(m)
            Dm = self.get_D(m)
            if full:
                tmp = self.eps * Ym + self.delta * Zm + self.delta**2 * Dm
                FN += np.dot(self.F[N-m],tmp)
            if not order:
                continue
            for (p,q),Fpq in self.ordered_F[N-m].orders.items():
                FN_.add((p,q+1),np.dot(Fpq,Ym))
                FN_.add((p+1,q),np.dot(Fpq,Zm)) 
                FN_.add((p+2,q),np.dot(Fpq,Dm)) 
        if full:
            self.F.append(FN)
            assert np.linalg.norm(self.K_appr[N]-self.K[N]-self.F[N]/self.delta**2)/np.linalg.norm(self.F[N])<1e-6
        if not order:
            return
        self.ordered_F.append(FN_)
        if self.max_order is not None:
            return
        try: 
            FN = self.F[N]
        except:
            return 
        tmp = np.zeros((self.sys_size,)*2,dtype=complex)
        for (p,q),Fpq in FN_.orders.items():
            tmp += self.delta**p * self.eps**q * Fpq
        assert np.linalg.norm(tmp-self.F[N])/np.linalg.norm(self.F[N])<1e-6
    def get_order(self,p,q,N):
        if p==0:
            return self.get_O0x(q,N)
        if p==1:
            return self.get_O1x(q,N)
            #if q==1:
            #    return self.get_O11(N)
            #if q==2:
            #    return self.get_O12(N)
        if p==2:
            return self.get_O2x(q,N)
            #if q==2:
            #    self.get_O22(N)
            #if q==1:
            #    return self.get_O21(N)
        if p==3:
            return self.get_O3x(q,N)
            #if q==1:
            #    return self.get_O31(N)
            #if q==2:
            #    return self.get_O32(N)
        if p==4:
            return self.get_O4x(q,N)
            if q==1:
                return self.get_O41(N)
            if q==2:
                return self.get_O42(N)

        #FN,ct = self.get_sum_permute(N,'C',['Z','Y'],[p-2,q-1])
        #FN_,ct_ = self.get_sum_permute(N,'X',['Z','Y'],[p,q-1])
        #FN += FN_
        #ct += ct_
        #FN_,ct_ = self.get_sum_permute(N,'X',['D','Z','Y'],[1,p-2,q-1])
        #FN += FN_
        #ct += ct_
        #return FN,ct
    def _get_product(self,typ,m):
        pd = np.eye(self.sys_size) 
        for mix in m:
            pd = np.dot(pd,self.get_matrix(typ,mix))
        return pd 
    def get_sum_product(self,N,l,typ1,typ2):
        ct = 0
        if l<0:
            return 0,ct
        if l==0:
            return self.get_matrix(typ1,N).copy(),ct
        FN = np.zeros((self.sys_size,)*2,dtype=complex) 
        mrange = range(1,N-(l-1)+1)
        for m in itertools.product(mrange,repeat=l):
            _sum = sum(m)
            if _sum>N:
                continue
            ct += 1
            FN += np.dot(self.get_matrix(typ1,N-_sum),self._get_product(typ2,m))
        return FN,ct
    def get_O0x(self,q,N):
        return self.get_sum_product(N,q-1,'X','Y')
    def _permute1(self,typ1,typ2,m):
        A = self.get_matrix(typ1,m[0]) 
        if len(m)==1:
            return A.copy() 
        Bs = [self.get_matrix(typ2,mix) for mix in m[1:]]
        L = [np.eye(self.sys_size)] 
        for B in Bs:
            L.append(np.dot(L[-1],B))
        R = [np.eye(self.sys_size)]
        for B in Bs[::-1]:
            R.append(np.dot(B,R[-1])) 
        _sum = np.zeros((self.sys_size,)*2,dtype=complex)
        l = len(m)
        for nl in range(l):
            nr = max(l - nl - 1,0)
            _sum += np.dot(np.dot(L[nl],A),R[nr])
        return _sum 
    def get_sum_permute1(self,N,l,typ1,typ2,typ3):
        FN = np.zeros((self.sys_size,)*2,dtype=complex)
        mrange = range(1,N-(l-1)+1)
        ct = 0
        for m in itertools.product(mrange,repeat=l):
            _sum = sum(m)
            if _sum>N:
                continue
            ct += 1
            FN += np.dot(self.get_matrix(typ1,N-_sum),self._permute1(typ2,typ3,m))
        return FN,ct
    def get_O1x(self,q,N):
        return self.get_sum_permute1(N,q,'X','Z','Y')
    def get_O11(self,N):
        FN = np.zeros((self.sys_size,)*2,dtype=complex)
        for m in range(1,N+1):
            FN += np.dot(self.get_X(N-m),self.get_Z(m))
        return FN,None 
    def get_O12(self,N):
        FN = np.zeros((self.sys_size,)*2,dtype=complex)
        for m in range(1,N):
            Ym = self.get_Y(m) 
            for n in range(1,N):
                if m+n>N:
                    continue
                Zn = self.get_Z(n) 
                ac = np.dot(Ym,Zn)+np.dot(Zn,Ym)
                FN += np.dot(self.get_X(N-m-n),ac)
        return FN,None 
    def get_O21(self,N):
        FN = self.get_C(N).copy()
        for m in range(1,N):
            Zm = self.get_Z(m)
            for n in range(1,N):
                if m+n>N:
                    continue
                Zn = self.get_Z(n)
                X = self.get_X(N-m-n)
                FN += np.dot(X,np.dot(Zm,Zn))
        for m in range(1,N+1):
            FN += np.dot(self.get_X(N-m),self.get_D(m))
        return FN,None 
    def get_O22(self,N):
        FN = np.zeros((self.sys_size,)*2,dtype=complex)
        for m in range(1,N+1):
            FN += np.dot(self.get_C(N-m),self.get_Y(m))
        for m in range(1,N):
            Dm = self.get_D(m)
            for n in range(1,N):
                if m+n>N:
                    continue
                Yn = self.get_Y(n)
                X = self.get_X(N-m-n)
                tmp = np.dot(Dm,Yn)+np.dot(Yn,Dm)
                FN += np.dot(X,tmp)
        mrange = range(1,N-1)
        for m in itertools.product(mrange,repeat=3):
            _sum = sum(m)
            if _sum>N:
                continue
            X = self.get_X(N-_sum)
            Z0 = self.get_Z(m[0])
            Z1 = self.get_Z(m[1])
            Y = self.get_Y(m[2])
            tmp = np.dot(np.dot(Z0,Z1),Y)
            tmp += np.dot(np.dot(Z0,Y),Z1)
            tmp += np.dot(np.dot(Y,Z0),Z1)
            FN += np.dot(X,tmp)
        return FN,None
    def _permute(self,typs,lens,m):
        string = ''
        for typ,l in zip(typs,lens):
            string += ''.join([typ] * l)
        ls = set(itertools.permutations(string,len(string)))
        _sum = np.zeros((self.sys_size,)*2,dtype=complex)
        for p in ls:
            pd = np.eye(self.sys_size)
            for typ,mi in zip(p,m):
                pd = np.dot(pd,self.get_matrix(typ,mi))
            _sum += pd 
        return _sum
    def get_sum_permute(self,N,typ1,typs,lens):
        l = sum(lens)
        FN = np.zeros((self.sys_size,)*2,dtype=complex)
        mrange = range(1,N-(l-1)+1)
        ct = 0
        for m in itertools.product(mrange,repeat=l):
            _sum = sum(m)
            if _sum>N:
                continue
            ct += 1
            FN += np.dot(self.get_matrix(typ1,N-_sum),self._permute(typs,lens,m))
        return FN,ct
    def get_O2x(self,q,N):
        FN,ct = self.get_sum_product(N,q-1,'C','Y')
        FN_,ct_ = self.get_sum_permute1(N,q,'X','D','Y')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute(N,'X',['Z','Y'],[2,q-2+1])
        FN += FN_
        ct += ct_
        return FN,ct
    def get_O31(self,N):
        FN,ct = self.get_sum_product(N,1,'C','Z')
        FN_,ct_ = self.get_sum_product(N,3,'X','Z')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute1(N,2,'X','D','Z')
        FN += FN_
        ct += ct_
        return FN,ct
    def get_O32(self,N):
        FN,ct = self.get_sum_permute1(N,2,'C','Z','Y')
        FN_,ct_ = self.get_sum_permute1(N,4,'X','Y','Z')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute(N,'X',['D','Z','Y'],[1,1,1])
        FN += FN_
        ct += ct_
        return FN,ct
    def get_O3x(self,q,N):
        FN,ct = self.get_sum_permute1(N,q,'C','Z','Y')
        FN_,ct_ = self.get_sum_permute(N,'X',['Z','Y'],[3,q-1])
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute(N,'X',['D','Z','Y'],[1,1,q-1])
        FN += FN_
        ct += ct_
        return FN,ct
    def get_O41(self,N):
        FN,ct = self.get_sum_product(N,2,'C','Z')
        FN_,ct_ = self.get_sum_product(N,4,'X','Z')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_product(N,1,'C','D')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_product(N,2,'X','D')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute1(N,3,'X','D','Z')
        FN += FN_
        ct += ct_
        return FN,ct
    def get_O42(self,N):
        FN,ct = self.get_sum_permute1(N,3,'C','Y','Z')
        FN_,ct_ = self.get_sum_permute1(N,5,'X','Y','Z')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute1(N,2,'C','D','Y')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute1(N,3,'X','Y','D')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute(N,'X',['D','Z','Y'],[1,2,1])
        FN += FN_
        ct += ct_
        return FN,ct
    def get_O4x(self,q,N):
        FN,ct = self.get_sum_permute(N,'C',['Y','Z'],[q-1,2])
        FN_,ct_ = self.get_sum_permute(N,'X',['Y','Z'],[q-1,4])
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute1(N,q,'C','D','Y')
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute(N,'X',['Y','D'],[q-1,2])
        FN += FN_
        ct += ct_
        FN_,ct_ = self.get_sum_permute(N,'X',['D','Z','Y'],[1,2,q-1])
        FN += FN_
        ct += ct_
        return FN,ct

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
    max_order = None 
    Nmax = 10 

    kn = Kernels(sys_size=2,max_order=max_order,eps=0.13,delta=0.27)
    for N in range(Nmax+1):
        print('\nN=',N)
        kn.get_next(order=True)
    for N in range(Nmax+1): 
        print('\nN=',N)
        #for order in range(1,max_order+1):
        for (p,q),F1 in kn.ordered_F[N].orders.items():
            if p>4:
                continue
            #if p==4:
            #    if q>2:
            #        continue
            print('p,q=',p,q)
            F2,ct2 = kn.get_order(p,q,N) 
            #_,ct3 = count1(N,order)
            if np.linalg.norm(F1)<1e-6:
                assert np.linalg.norm(F2)<1e-6
            else:
                assert np.linalg.norm(F1-F2)/np.linalg.norm(F1)<1e-6
            #if ct3 is None:
            #    continue
            #if np.linalg.norm(ct2)<1e-6:
            #    assert np.linalg.norm(ct3)<1e-6
            #else:
            #    assert np.linalg.norm(ct2-ct3)/np.linalg.norm(ct2)<1e-6
            #err3 = count(N,order)
            #err4 = approx(N,order)
            #print('order=',order,err1,err2,err3,err4)

string = 'AAB'
ls = itertools.permutations(string,len(string))
print(set(ls))
exit()

eps = [0.001]
max_order = 8 
Nmax = 50
plot_eps = True
sys_size = 128 
#plot_eps = False 
if plot_eps:
    fig,ax = plt.subplots(nrows=1,ncols=1)
    runs = range(20)
    every = 20
    for epsi,color in zip(eps,colors):
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

