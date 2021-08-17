import numpy as np
import warnings

def Bernstein(t):
    """
    Returns the weights of 3rd-order Bernstein polynomial
    t: [0.0:1.0]
    """
    t1 = 1.0 - t
    
    b0 = t1*t1*t1/6.0
    b1 = ( 3.0*t*t*t - 6.0*t*t         + 4.0) / 6.0
    b2 = (-3.0*t*t*t + 3.0*t*t + 3.0*t + 1.0) / 6.0
    b3 = t*t*t/6.0
    
    return np.array([[b0, b1, b2, b3]])

def ffd_vertex2d(P, st):
    """
    Returns 2D position (x,y) by free-form deformation
    P: (4,4,2) 4x4 2D grid for FFD
    st: relative position (s,t) in the center of 4x4 grid
    """
    assert P.shape==(4,4,2)
    w = np.dot(Bernstein(st[1]).T, Bernstein(st[0])).reshape(-1)

    x = (w * P[:,:,0].reshape(-1)).sum()
    y = (w * P[:,:,1].reshape(-1)).sum()
    
    return np.array([x, y])

def getIndexAndWeight(B, uv):
    """
    Returns index (i,j) for grid B and relative position (s,t) in that grid
    B: (N,M,2) regular 2D grid in the texture space
    uv: position in the texture space
    """
    u = uv[0]
    v = uv[1]

    '''
    for i in range(B.shape[0]-1):
        for j in range(B.shape[1]-1):
            u0,v0 = B[i  , j  ]
            u1,v1 = B[i+1, j+1]
            if (u0 <= u and u <=u1 and
                v0 <= v and v <=v1):
                s = (u-u0) / (u1-u0)
                t = (v-v0) / (v1-v0)
                return (i,j), (s,t)

    warnings.warn("({0},{1}) is out of FFD grid.".format(u,v), Warning)
    return (-1,-1), (0.0, 0.0)
    '''

    ############################################################
    ## B is regular grid, so we check in a simplified way
    ############################################################

    ## [CAUTION] check in U-direction
    check_u = False
    for j in range(B.shape[1]-1):
        u0 = B[0,j  ,0]
        u1 = B[0,j+1,0]

        if (u0 <= u and u <= u1):
            s = (u-u0) / (u1-u0)
            check_u = True
            break

    ## [CAUTION] check in V-direction
    check_v = False
    for i in range(B.shape[0]-1):
        v0 = B[i  ,0,1]
        v1 = B[i+1,0,1]

        if (v0 <= v and v <= v1):
            t = (v-v0) / (v1-v0)
            check_v = True
            break
    
    if (check_u and check_v):
        return (i,j), (s,t)

    warnings.warn("({0},{1}) is out of FFD grid.".format(u,v), Warning)
    return (-1,-1), (0.0, 0.0)

def FFD_precompute(B, u):
    """
    Returns precomputation values (ind, ST) for free-form deformation
    (B,u)---[FFD]--->(P,x)
    B: (N,M,2) regular 2D grid in the texture space
    u: (K,2) 2D vertices in the texture space
    """
    assert B.shape[2]==2
    assert u.shape[1]==2

    N, M, _ = B.shape
    K = u.shape[0]

    ind = np.full([K,2], -1, dtype=np.int32)
    ST  = np.zeros([K,2])

    for k in range(K):
        uv_k = u[k]
        (i,j), st = getIndexAndWeight(B, uv_k)

        ind[k,0] = i
        ind[k,1] = j
        ST[k] = st

        if (i==-1 and j==-1):
            warnings.warn("{0}-th vertex is out of FFD grid.".format(k), Warning)

    return ind, ST

def FFD(ind, ST, P):
    """
    Returns x in the image space with (K,2)
    (B,u)---[FFD]--->(P,x)
    ind: (K,2) indices list in the texture space
    ST : (K,2) st coordinate list in the texture space
    P  : (N,M,2) 2D deformed grid in the image space
    """
    assert ind.shape==ST.shape
    assert P.shape[2]==2

    N, M, _ = P.shape
    K = ST.shape[0]
    
    x = ST.copy()

    for k in range(K):
        i, j = ind[k]
        st  = ST[k]

        if (i==-1 and j==-1):
            warnings.warn("{0}-th vertex is out of FFD grid.".format(k), Warning)
            continue

        ## default 4 points
        P11 = P[i  ,j  ]
        P12 = P[i  ,j+1]
        P21 = P[i+1,j  ]
        P22 = P[i+1,j+1]

        ## 12 points for the outside of default 4 points
        P00 = P[i-1,j-1] if (i-1>=0) and (j-1>=0) else 2.0*P11 - P22
        P01 = P[i-1,j  ] if (i-1>=0)              else 2.0*P11 - P21
        P02 = P[i-1,j+1] if (i-1>=0)              else 2.0*P12 - P22
        P03 = P[i-1,j+2] if (i-1>=0) and (j+2< M) else 2.0*P12 - P21

        P10 = P[i  ,j-1] if (j-1>=0) else 2.0*P11 - P12
        P13 = P[i  ,j+2] if (j+2< M) else 2.0*P12 - P11

        P20 = P[i+1,j-1] if (j-1>=0) else 2.0*P21 - P22
        P23 = P[i+1,j+2] if (j+2< M) else 2.0*P22 - P21

        P30 = P[i+2,j-1] if (i+2< N) and (j-1>=0) else 2.0*P21 - P12
        P31 = P[i+2,j  ] if (i+2< N)              else 2.0*P21 - P11
        P32 = P[i+2,j+1] if (i+2< N)              else 2.0*P22 - P12
        P33 = P[i+2,j+2] if (i+2< N) and (j+2< M) else 2.0*P22 - P11

        ## free-form deformation lattice
        P_local = np.array([[P00,P01,P02,P03],
                            [P10,P11,P12,P13],
                            [P20,P21,P22,P23],
                            [P30,P31,P32,P33]])

        x[k] = ffd_vertex2d(P_local, st)

    return x
