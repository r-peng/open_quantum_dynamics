import numpy as np
# worst case scenario
class Kernel:
    def __init__(self,N,max_order=None):
        self.N = N
        self.max_order = N+1 if max_order is None else max_order
        self.orders = [dict() for _ in range(self.max_order+1)]
    def add(self,eps,power,coeff):
        order = sum(power)
        if order > self.max_order:
            return
        expr = self.orders[order]
        key = tuple(eps),tuple(power)
        if key in expr:
            expr[key] += coeff
        else:
            expr[key] = coeff
        if expr[key]==0:
            expr.pop(key)
        self.orders[order] = expr
    def multiply(self,eps,power,eps_new):
        eps = list(eps)
        power = list(power)
        if eps_new in eps:
            idx = eps.index(eps_new)
            power[idx] += 1
            return eps,power
        eps = np.array(eps)
        nless = len(eps[eps<eps_new])
        eps = list(eps)
        eps = eps[:nless] + [eps_new] + eps[nless:]
        power = power[:nless] + [1] + power[nless:]
        return eps,power
class Kernels:
    def __init__(self,max_order=2):
        self.max_order = max_order

        K0 = Kernel(0,max_order=self.max_order)
        K0.add((1,),(1,),1)

        self.Ks = [K0]
    def get_next(self):
        # K_{N} = U_{N+1} - U_{N} - \sum_{m=2}^{N+1}K_{N+1-m}U_{m-1}
        # U_m = 1+eps_m

        N = len(self.Ks) 
        KN = Kernel(N,max_order=self.max_order) 

        KN.add((N+1,),(1,),1)
        KN.add((N,),(1,),-1)

        for m in range(2,N+2):
            for expr in self.Ks[N+1-m].orders:
                for key,coeff in expr.items():

                    eps,power = key
                    KN.add(eps,power,-coeff)

                    eps,power = KN.multiply(eps,power,m-1)
                    KN.add(eps,power,-coeff)

        self.Ks.append(KN) 
def get_O2(N):
    expr = dict()
    def _add(eps1,eps2,coeff):
        if eps1<=0:
            return
        if eps2<=0:
            return
        if eps1==eps2:
            key = (eps1,),(2,)
        elif eps1<eps2:
            key = (eps1,eps2),(1,1)
        else:
            key = (eps2,eps1),(1,1)
        if key in expr:
            expr[key] += coeff
        else:
            expr[key] = coeff

    for m in range(1,N+1):
        _add(m,N+1-m,-1)
    for m in range(1,N):
        _add(m,N-m,3)
    for m in range(1,N-1):
        _add(m,N-1-m,-3)
    for m in range(1,N-2):
        _add(m,N-2-m,1)
    return expr
def _get_O2(N):
    expr = dict()
    if N%2==0:
        k = N//2
        if k-1>0:
            expr[(k-1,),(2,)] = 1
        if k>0:
            expr[(k,),(2,)] = 3
        for i in range(1,k+1):
            if i<2*k-i+1:
                expr[(i,2*k-i+1),(1,1)] = -2
            if i<2*k-i:
                expr[(i,2*k-i),(1,1)] = 6 
            if i<2*k-i-1:
                expr[(i,2*k-i-1),(1,1)] = -6
            if i<2*k-i-2:
                expr[(i,2*k-i-2),(1,1)] = 2
    else:
        k = (N+1)//2
        if k-1>0:
            expr[(k-1,),(2,)] = -3 
        if k>0:
            expr[(k,),(2,)] = -1
        for i in range(1,k):
            if i<2*k-i:
                expr[(i,2*k-i),(1,1)] = -2
            if i<2*k-i-1:
                expr[(i,2*k-i-1),(1,1)] = 6 
            if i<2*k-i-2:
                expr[(i,2*k-i-2),(1,1)] = -6
            if i<2*k-i-3:
                expr[(i,2*k-i-3),(1,1)] = 2
    return expr

max_order = 3 
Ns = np.array(range(1,41))
kn = Kernels(max_order=max_order)
sum_coeff = {order:[] for order in range(1,max_order+1)} 
nterms = {order:[] for order in range(1,max_order+1)} 
for N in Ns:
    kn.get_next()
    K = kn.Ks[-1]

    print('\nN=',N)
    for order in range(1,max_order+1):
        expr = K.orders[order]
        if order==2:
            ls = get_O2(N)
            for key,val in ls.items():
                assert val==expr[key]
            for key,val in expr.items():
                assert val==ls[key]
        if len(expr)==0:
            continue
        sum_coeff[order].append(sum([abs(val) for val in expr.values()]))
        nterms[order].append(len(expr))
        print('order,nterms,sum_coeff=',order,nterms[order][-1],sum_coeff[order][-1])
        #if order==3:
        #    print(expr)
exit()

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4*2,6.4)})
fig,ax = plt.subplots(nrows=1,ncols=2)
colors = 'r','g','b','y','c','orange','pink','grey','olive'
for order in range(1,max_order+1):
    y1 = np.log10(np.array(nterms[order]))
    print(np.array(nterms[order]))
    n = len(y1)
    x = np.log10(Ns[-n:] - Ns[-n] + 1)
    ax[0].plot(x,y1,linestyle='-',color=colors[order],label=f'O{order}')
    y2 = np.log10(np.array(sum_coeff[order]))
    print('order,slope=',order,(y2[-1]-y2[0])/(x[-1]-x[0]))
    ax[1].plot(x,y2,linestyle='-',color=colors[order],label=f'O{order}')
    #ax[2].plot(x,y2-y1,linestyle='-',color=colors[order],label=f'O{order}')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.95)
ax[0].set_xlabel('log10(N-N0)')
ax[1].set_xlabel('log10(N-N0)')
#ax[2].set_xlabel('log10(N-N0)')
ax[0].set_ylabel('log10(nterms)')
ax[1].set_ylabel('log10(sum coeff)')
#ax[2].set_ylabel('log10(scalar)')
ax[0].legend()
fig.savefig(f'scalar.png')
