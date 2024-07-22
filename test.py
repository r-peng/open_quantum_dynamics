import numpy as np
import scipy.linalg
np.set_printoptions(suppress=True,precision=6)
N = 6
A = np.zeros((N,N))
for i in range(N-1):
    A[i,i+1] = 1
An = A
for i in range(2,10):
    An = np.dot(A,An)
    print(i)
    print(An)
exit()


p,l,u = scipy.linalg.lu(A)
print(p)
print(l)
print(u)
exit()
#u,s,v = np.linalg.svd(A)
#print(A)
#print(u.T)
##print(s)
#print(v)
#print(np.linalg.norm(A-np.dot(u[:,:-1],v[:-1,:])))
##print(np.dot(u*s,v))
#exit()
#print(A)
w,v = np.linalg.eig(A)
#print(w)
#print(v)
B = A - np.eye(N)

