import numpy as np
import itertools
import quimb.tensor as qtn
class SumMPS:
    def __init__(self,omega,lambd,N):
        self.N = N 
        self.eta = np.zeros(N,dtype=complex)
        for i in range(N):
            self.eta[i] = sum([lk**2*np.exp(-i*ok) for lk,ok in zip(lambd,omega)])
        K = len(omega)
        
        # construct MPS tensors
        M = np.zeros((2*K+2,2*K+2,2,2),dtype=complex)
        M[0,0] = np.ones((2,2))
        M[0,-1] = np.array([[0,self.eta[0].conj()],[self.eta[0],0]])
        M[-1,-1] = np.ones((2,2)) 

        tmp = np.array([[0,-1],[1,0]])
        for k,(lk,ok) in enumerate(zip(lambd,omega)):
            M[0,1+k] = lk * tmp 
            M[0,K+1+k] = - lk.conj() * tmp 

            exp = np.exp(-ok)
            M[1+k,-1,1] = lk*exp
            M[K+1+k,-1,:,1] = lk.conj()*exp.conj()

            M[1+k,1+k] = exp 
            M[K+1+k,K+1+k] = exp.conj() 

        self.mps = [None] * N
        for i in range(self.N):
            if i==0:
                self.mps[i] = M[0].reshape(1,2*K+2,2,2)
            elif i==self.N-1:
                self.mps[i] = M[:,-1,:,:]
            else:
                self.mps[i] = M.copy()
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
        return  S 
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
        #print(mps.max_bond())

        # canonize to the left 
        for i in range(self.N-1,0,-1):
            T1,T2 = mps[f'L{i-1}'],mps[f'L{i}']
            qtn.tensor_canonize_bond(T1,T2,absorb='left') 
        for n in range(nsweep):
            #print('sweep=',n)
            if n%2==0:
                for i in range(self.N-1):
                    T1,T2 = mps[f'L{i}'],mps[f'L{i+1}']
                    qtn.tensor_compress_bond(T1,T2,absorb='right',cutoff=1e-10) 
            else:
                for i in range(self.N-1,0,-1):
                    T1,T2 = mps[f'L{i-1}'],mps[f'L{i}']
                    qtn.tensor_compress_bond(T1,T2,absorb='left',cutoff=1e-10) 
            #print(mps.max_bond())
        return mps
            
check = True 
#check = False 
if check:
    N = 7 
    K = 10
    lambd = np.random.rand(K)*2-1 + 1j * (np.random.rand(K)*2-1)
    omega = np.random.rand(K)*2-1 + 1j * (np.random.rand(K)*2-1)
    mps = SumMPS(lambd,omega,N)
    for i in range(N-1,-1,-1):
        print(f'check site = {i}...')
        err = 0
        Strace = 0
        nsite = N-1-i+1
        for s in itertools.product((0,1),repeat=nsite*2):
            S = mps.compute_S(s)
            amp = mps.compute_mps_amplitude(s)
            err += np.linalg.norm(S-amp)
            Strace += np.linalg.norm(S)
        print('err=',err,Strace)
    exit()

N = 20
for N in range(5,50,5):
     eta = np.random.rand(N) + 1j * np.random.rand(N)
     mps = SumMPS(eta)
     cmps = mps.compress(3)
     for i in range(N):
         data = cmps[f'L{i}'].data
         print(data.shape)
     #exit()
     print(N,cmps.max_bond())

