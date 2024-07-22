import numpy as np
import itertools
import quimb.tensor as qtn
class SumMPS:
    def __init__(self,eta):
        self.eta = eta
        self.N = len(eta)
        
        self.hL = np.zeros((2,2))
        for s,sc in itertools.product((0,1),(0,1)):
            self.hL[s,sc] = s-sc

        self.hR = np.zeros((self.N,2,2),dtype=complex) 
        for i in range(N):
            for s,sc in itertools.product((0,1),repeat=2):
                self.hR[i,s,sc] = s * eta[i] - eta[i].conj() * sc
        self.hLR = self.hL * self.hR[0]

        self.compute_mps()
    def compute_mps(self):
        term = self.get_term(0,0) 
        mps = [None] * N
        for i in range(self.N):
            if i==0 or i==self.N-1:
                mps[i] = term[i].reshape(1,2,2)
            else:
                mps[i] = term[i].reshape(1,1,2,2)
        for i in range(self.N):
            for j in range(i,self.N):
            #for j in range(i+1,min(i+2,self.N)):
                if i==0 and j==0:
                    continue
                term = self.get_term(i,j) 
                mps = self.add_term(mps,term) 
        self.mps = mps
        return mps
    def get_term(self,i,j):
        M = np.ones((self.N,2,2),dtype=complex)
        if j==i:
            M[i] = self.hLR.copy()
        else:
            M[i] = self.hL.copy()
            M[j] = self.hR[j-i].copy()
        return M
    def add_term(self,mps,term):
        for i,Mi in enumerate(mps):
            if i==0 or i==self.N-1:
                mps[i] = np.concatenate([Mi,term[i].reshape(1,2,2)],axis=0)
            else:
                dim1,dim2 = Mi.shape[:2]
                Minew = np.zeros((dim1+1,dim2+1,2,2),dtype=complex)
                Minew[:dim1,:dim2] = Mi
                Minew[-1,-1] = term[i]
                mps[i] = Minew
        return mps
    def compute_mps_amplitude(self,s):
        for i in range(self.N-1,-1,-1):
            s1,s2 = s[i*2],s[2*i+1]
            v1 = np.zeros(2)
            v1[s1] = 1
            v2 = np.zeros(2)
            v2[s2] = 1
            if i==self.N-1:
                M = np.einsum('aij,i,j->a',self.mps[i],v1,v2)
            elif i==0:
                M = np.einsum('aij,i,j,a->',self.mps[i],v1,v2,M)
            else:
                M = np.einsum('abij,i,j,b->a',self.mps[i],v1,v2,M)
        return M
    def compute_S(self,s):
        S = 0
        for i in range(self.N):
            si,sic = s[2*i],s[2*i+1]
            for j in range(i,self.N):
                sj,sjc = s[2*j],s[2*j+1]
                S += (si-sic)*(sj*self.eta[j-i]-sjc*self.eta[j-i].conj())
        return S 
    def compress(self,nsweep):
        mps = qtn.TensorNetwork([])
        for i,data in enumerate(self.mps):
            if i==0:
                inds = (f'L{i},{i+1}',f's{i}+',f's{i}-',)
            elif i==self.N-1:
                inds = (f'L{i-1},{i}',f's{i}+',f's{i}-',)
            else:
                inds = (f'L{i-1},{i}',f'L{i},{i+1}',f's{i}+',f's{i}-',)
            mps.add_tensor(qtn.Tensor(data=data,inds=inds,tags=f'L{i}'))
        print(mps.max_bond())

        # canonize to the left 
        for i in range(self.N-1,0,-1):
            T1,T2 = mps[f'L{i-1}'],mps[f'L{i}']
            qtn.tensor_canonize_bond(T1,T2,absorb='left') 
        for n in range(nsweep):
            print('sweep=',n)
            if n%2==0:
                for i in range(self.N-1):
                    T1,T2 = mps[f'L{i}'],mps[f'L{i+1}']
                    qtn.tensor_compress_bond(T1,T2,absorb='right',cutoff=1e-10) 
            else:
                for i in range(self.N-1,0,-1):
                    T1,T2 = mps[f'L{i-1}'],mps[f'L{i}']
                    qtn.tensor_compress_bond(T1,T2,absorb='left',cutoff=1e-10) 
            print(mps.max_bond())
            
check = True 
check = False 
if check:
    for N in range(2,8): 
        print(f'check length N={N}...')
        eta = np.random.rand(N) + 1j * np.random.rand(N)
        mps = SumMPS(eta)
        err = 0
        for s in itertools.product((0,1),repeat=N*2):
            S = mps.compute_S(s)
            amp = mps.compute_mps_amplitude(s)
        err += np.linalg.norm(S-amp)
        print('err=',err)
    exit()

N = 20
eta = np.random.rand(N) + 1j * np.random.rand(N)
#eta = np.array([np.exp(-i) for i in range(N)])
#eta = np.ones(N)
mps = SumMPS(eta)
mps.compress(5)

