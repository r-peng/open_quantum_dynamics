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
        self.bond_dict = [None] * N
        self.mps = [None] * N
        for i in range(self.N-1,-1,-1):
            self.get_mps_site(i) 
    def get_mps_final(self):
        M = np.zeros((self.N+1,2,2),dtype=complex)
        M[0] = self.hLR.copy() 
        M[1:self.N] = self.hR[1:].copy()
        M[self.N] = np.ones((2,2))
        self.mps[self.N-1] = M

        self.bond_dict[self.N-1] = {(f'h{i}',):i for i in range(1,self.N)}
    def get_mps_site(self,i):
        if i==self.N-1:
            return self.get_mps_final()
        bond_dict_prev = self.bond_dict[i+1]
        size_prev = self.mps[i+1].shape[0]
        bond_dict = dict()
        size = 0
        M = []
        nspin = self.N-1 - i + 1 # number of current spin
    
        # S
        Mi = np.zeros((size_prev,2,2),dtype=complex)
        Mi[0] = np.ones((2,2))
        for j in range(1,nspin):
            key = [1] * nspin
            key[j] = f'h{j}'
            idx = bond_dict_prev[tuple(key)[1:]] 
            Mi[idx] = self.hL.copy()
        Mi[-1] = self.hLR.copy()     
        M.append(Mi)
        size += 1 
        if i==0:
            self.mps[i] = np.stack(M,axis=0)
            return
        
        for d in range(1,self.N): # dist of h
            for j in range(min(nspin,d)):
                Mi = np.zeros((size_prev,2,2),dtype=complex)
                key = [1] * nspin
                key[j] = f'h{d}'
                key = tuple(key)
                if key[1:] in bond_dict_prev:
                    idx = bond_dict_prev[key[1:]]
                    Mi[idx] = np.ones((2,2))
                else:
                    Mi[-1] = self.hR[d].copy()
                M.append(Mi)
                bond_dict[key] = size
                size += 1

    
        Mi = np.zeros((size_prev,2,2),dtype=complex)
        Mi[-1] = np.ones((2,2)) 
        M.append(Mi)

        self.mps[i] = np.stack(M,axis=0)
        self.bond_dict[i] = bond_dict
    def compute_mps_amplitude(self,s):
        nsite = len(s) // 2
        mps = self.mps[-nsite:]
        for i in range(nsite-1,-1,-1):
            s1,s2 = s[i*2],s[2*i+1]
            v1 = np.zeros(2)
            v1[s1] = 1
            v2 = np.zeros(2)
            v2[s2] = 1
            if i==nsite-1:
                M = np.einsum('aij,i,j->a',mps[i],v1,v2)
            else:
                M = np.einsum('abij,i,j,b->a',mps[i],v1,v2,M)
        return M[0]
    def compute_S(self,s):
        nsite = len(s)//2
        S = 0
        for i in range(nsite):
            si,sic = s[2*i],s[2*i+1]
            for j in range(i,nsite):
                sj,sjc = s[2*j],s[2*j+1]
                S += (si-sic)*(sj*self.eta[j-i]-sjc*self.eta[j-i].conj())
        return S 
    def compress(self,nsweep):
        mps = qtn.TensorNetwork([])
        for i,tsr in enumerate(self.mps):
            data = tsr[0] if i==0 else tsr
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
    N = 7 
    eta = np.random.rand(N) + 1j * np.random.rand(N)
    mps = SumMPS(eta)
    for i in range(N-1,-1,-1):
        print(f'check site = {i}...')
        err = 0
        nsite = N-1-i+1
        for s in itertools.product((0,1),repeat=nsite*2):
            S = mps.compute_S(s)
            amp = mps.compute_mps_amplitude(s)
        err += np.linalg.norm(S-amp)
        print('err=',err)

N = 20
eta = np.random.rand(N) + 1j * np.random.rand(N)
mps = SumMPS(eta)
mps.compress(5)

