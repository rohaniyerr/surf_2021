import numpy as np
import sys

def nds(nu, dist, sigma, i):
    return nu[i]*dist[i]*sigma[i]


def calc_gas_evol(sigma, sigma_d, nu, vn, dist, dt):
    n  = len(sigma)
    Ag = np.empty(n)
    Bg = np.empty(n)
    Cg = np.empty(n)

    div = 10
    dt1 = dt/div
    dr  = dist[1] - dist[0]

    for j in range(div):
        # A(i,i)S(i,j+1) = S(i,j)     ... Ad(i)
        for i in range(n):
            if (i<(n-1)):
                Ag[i] = (-dt1/(dr*dr) * (0.5*nds(nu,dist,sigma,i-1) + nds(nu,dist,sigma,i) +0.5*nds(nu,dist,sigma,i+1))/sigma[i]) / dist[i]
            else:
                Ag[i] = (-dt1/(dr*dr) * (0.5*nds(nu,dist,sigma,i-1) + nds(nu,dist,sigma,i))/sigma[i]) / dist[i]

            # add advection term, considering upwind scheme
            if (vn[i]<0):
                Ag[i] += vn[i]*dt1/dr
                
            if (i<(n-1)):
                if (vn[i+1] > 0):
                    Ag[i] -= vn[i+1]*dt1/dr

        # A(i+1,i)S(i+1,j+1) = S(i,j) ... Bd(i)
        for i in range(n-1):
            Bd[i] =  (dt1/(dr*dr) * 0.5*(nds(nu,dist,sigma,i) + nds(nu,dist,sigma,i+1))/sigma[i+1]) / dist[i]
        
            if (vn[i+1] < 0):
                Bg[i] -= (dist[i+1]/dist[i]) * vn[i+1]*dt1/dr
        Bg[-1] = 0.0
            
        # A(i-1,i)S(i-1,j+1) = S(i,j) ... Cd(i)
        for i in range(1,n):
            Cg[i] =  dt1/(dr*dr) * 0.5*(nds(nu,dist,sigma,i-1) + nds(nu,dist,sigma,i))/ (sigma[i-1]*dist[i])

            if (vn[i]>0):
                Cg[i] += (dist[i-1]/dist[i]) * vn[i]*dt1/dr

        #print("B", sigma_d[-5:])
        #print("b", sigma[-5:])
        #print("Az", Ad[-5:])
        #print("Bz", Bd[-5:])
        #print("Cz", Cd[-5:])
        sigma_g = solve_Crank_Nicolson(Ad, Bd, Cd, sigma_g)
        # repeat for div times
        #print("A", sigma_d[-5:])
        #print("a", sigma[-5:])
        #print(nu[-5:])
        #print()

        if (np.isnan(sigma_d[-1])):
            sys.exit()
        
    # return
    return sigma_g

    
def calc_dust_evol(sigma, sigma_d, nu, vn, dist, dt):
    n  = len(sigma)
    Ad = np.empty(n)
    Bd = np.empty(n)
    Cd = np.empty(n)

    div = 10
    dt2 = dt/div
    dr  = dist[1] - dist[0]

    for j in range(div):
        # A(i,i)S(i,j+1) = S(i,j)     ... Ad(i)
        for i in range(n):
            if (i<(n-1)):
                Ad[i] = (-dt2/(dr*dr) * (0.5*nds(nu,dist,sigma,i-1) + nds(nu,dist,sigma,i) +0.5*nds(nu,dist,sigma,i+1))/sigma[i]) / dist[i]
            else:
                Ad[i] = (-dt2/(dr*dr) * (0.5*nds(nu,dist,sigma,i-1) + nds(nu,dist,sigma,i))/sigma[i]) / dist[i]

            # add advection term, considering upwind scheme
            if (vn[i]<0):
                Ad[i] += vn[i]*dt2/dr
                
            if (i<(n-1)):
                if (vn[i+1] > 0):
                    Ad[i] -= vn[i+1]*dt2/dr

        # A(i+1,i)S(i+1,j+1) = S(i,j) ... Bd(i)
        for i in range(n-1):
            Bd[i] =  (dt2/(dr*dr) * 0.5*(nds(nu,dist,sigma,i) + nds(nu,dist,sigma,i+1))/sigma[i+1]) / dist[i]
        
            if (vn[i+1] < 0):
                Bd[i] -= (dist[i+1]/dist[i]) * vn[i+1]*dt2/dr
        Bd[-1] = 0.0
            
        # A(i-1,i)S(i-1,j+1) = S(i,j) ... Cd(i)
        for i in range(1,n):
            Cd[i] =  dt2/(dr*dr) * 0.5*(nds(nu,dist,sigma,i-1) + nds(nu,dist,sigma,i))/ (sigma[i-1]*dist[i])

            if (vn[i]>0):
                Cd[i] += (dist[i-1]/dist[i]) * vn[i]*dt2/dr

        #print("B", sigma_d[-5:])
        #print("b", sigma[-5:])
        #print("Az", Ad[-5:])
        #print("Bz", Bd[-5:])
        #print("Cz", Cd[-5:])
        sigma_d = solve_Crank_Nicolson(Ad, Bd, Cd, sigma_d)
        # repeat for div times
        #print("A", sigma_d[-5:])
        #print("a", sigma[-5:])
        #print(nu[-5:])
        #print()

        if (np.isnan(sigma_d[-1])):
            sys.exit()
        
    # return
    return sigma_d
    
def solve_Crank_Nicolson(Ao, Bo, Co, S):
    theta = 0.5
    
    # explicit side
    n = len(S)
    S1 = np.empty(n)
    for i in range(n):
        S1[i] = Co[i]*theta*S[max(0,i-1)] + (1+Ao[i]*theta)*S[i] + Bo[i]*theta*S[min(i+1, n-1)]
    
    # convert to implicit-solver matrix
    Ai = np.empty(n)
    Bi = np.empty(n)
    Ci = np.empty(n)
    
    for i in range(n):
        Ai[i] = 1.0 - Ao[i]*(1-theta)
    for i in range(n-1):
        Bi[i] =     - Bo[i]*(1-theta)
    for i in range(1,n):
        Ci[i] =     - Co[i]*(1-theta)
    
    # solve tridiag
    S2 = solve_tridiag(Ai, Bi, Ci, S1)

    return S2

def solve_tridiag(Ao, Bo, Co, S):
    A = Ao
    B = Bo
    C = Co;
    
    imax = len(A);
    if (imax != len(B) or imax != len(C)):
        print("diffuse.cpp/solve_tridiag: wrong vector size.")
    
    # 1st row
    B[0] /= A[0]
    S[0] /= A[0]
    A[0] = 1.0
    
    for j in range(1, imax):
        # swipe out C[j] to 0.0
        A[j] -= B[j-1] * C[j]
        S[j] -= S[j-1] * C[j]
        C[j] = 0.0
        
        #  divide each component by A 
        B[j] /= A[j]
        S[j] /= A[j]
        A[j] = 1.0
    
    # solve diag
    for j in range(imax-2, -1, -1):
        # last row ... A[j-1] Sn[j-1] = Sb[j-1] */
        S[j] -= S[j+1] * B[j]

    return S
