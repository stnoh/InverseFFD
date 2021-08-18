from . import ffd2d
from scipy.sparse import dok_matrix
import numpy as np

def ConstructBarycentricWeights(B, u, simplified=True):
    """
    Returns sparse matrix based on the vertices' barycentric coordinate
    B: (N,M,2) regular 2D grid in the texture space
    u: 2D vertices in texture space.
    simplified: it controls the complexity of sparse matrix.
                if it is True (default), then it assumes 1st order B-spline deformation.
                if it is False, then it assumes 3rd order B-spline deformation.
    """
    assert len(B.shape)==3
    assert B.shape[2]==2 and u.shape[1]==2

    N, M, _ = B.shape
    K = u.shape[0]
    Nrows = 2*(K + (N-2)*M + N*(M-2))

    Bc_tilde_dok = dok_matrix((Nrows, N*M*2))

    if simplified:
        for k in range(K):
            u_k = u[k]
            (i,j), (w_u1,w_v1) = ffd2d.getIndexAndWeight(B, u_k)

            ## [CAUTION] this is barycentric coordinate
            w_u0 = 1.0 - w_u1
            w_v0 = 1.0 - w_v1

            w00 = w_u0 * w_v0
            w01 = w_u1 * w_v0
            w10 = w_u0 * w_v1
            w11 = w_u1 * w_v1

            ## compute indices for extended matrix
            B_row = 2 * k

            ## indices for 4 major points
            B_col_i0j0 = 2 * ((i+0)*M + (j+0) )
            B_col_i0j1 = 2 * ((i+0)*M + (j+1) )
            B_col_i1j0 = 2 * ((i+1)*M + (j+0) )
            B_col_i1j1 = 2 * ((i+1)*M + (j+1) )

            ## 4 major points
            Bc_tilde_dok[B_row  , B_col_i0j0  ] = w00
            Bc_tilde_dok[B_row+1, B_col_i0j0+1] = w00
            Bc_tilde_dok[B_row  , B_col_i0j1  ] = w01
            Bc_tilde_dok[B_row+1, B_col_i0j1+1] = w01
            
            Bc_tilde_dok[B_row  , B_col_i1j0  ] = w10
            Bc_tilde_dok[B_row+1, B_col_i1j0+1] = w10
            Bc_tilde_dok[B_row  , B_col_i1j1  ] = w11
            Bc_tilde_dok[B_row+1, B_col_i1j1+1] = w11
    
    else:
        def AddElement(condition, B_row, B_col, w_src, w_dst):
            if condition:
                Bc_tilde_dok[B_row  , B_col  ] = w_src
                Bc_tilde_dok[B_row+1, B_col+1] = w_src
            else:
                w_dst += w_src
            return w_dst

        for k in range(K):
            u_k = u[k]
            (i,j), (s,t) = ffd2d.getIndexAndWeight(B, u_k)

            ## compute indices for extended matrix
            B_row = 2 * k

            ## 16-points version: compute barycentric weights for 16 points
            w = np.dot(ffd2d.Bernstein(t).T, ffd2d.Bernstein(s))

            ## weights for 4 major points
            w11 = w[1,1]
            w12 = w[1,2]
            w21 = w[2,1]
            w22 = w[2,2]

            ## 12 minor points in advance
            w11 = AddElement( (i-1>=0 and j-1>=0), B_row, 2*((i-1)*M + (j-1)), w[0,0], w11 )
            w11 = AddElement( (i-1>=0           ), B_row, 2*((i-1)*M + (j  )), w[0,1], w11 )
            w12 = AddElement( (i-1>=0           ), B_row, 2*((i-1)*M + (j+1)), w[0,2], w12 )
            w12 = AddElement( (i-1>=0 and j+2< M), B_row, 2*((i-1)*M + (j+2)), w[0,3], w12 )

            w11 = AddElement( (           j-1>=0), B_row, 2*((i  )*M + (j-1)), w[1,0], w11 )
            w12 = AddElement( (           j+2< M), B_row, 2*((i  )*M + (j+2)), w[1,3], w12 )

            w21 = AddElement( (           j-1>=0), B_row, 2*((i+1)*M + (j-1)), w[2,0], w21 )
            w22 = AddElement( (           j+2< M), B_row, 2*((i+1)*M + (j+2)), w[2,3], w22 )

            w21 = AddElement( (i+2< N and j-1>=0), B_row, 2*((i+2)*M + (j-1)), w[3,0], w21 )
            w21 = AddElement( (i+2< N           ), B_row, 2*((i+2)*M + (j  )), w[3,1], w21 )
            w22 = AddElement( (i+2< N           ), B_row, 2*((i+2)*M + (j+1)), w[3,2], w22 )
            w22 = AddElement( (i+2< N and j+2< M), B_row, 2*((i+2)*M + (j+2)), w[3,3], w22 )

            ## indices for 4 major points
            B_col_i1j1 = 2*( (i+0)*M + (j+0) )
            B_col_i1j2 = 2*( (i+0)*M + (j+1) )
            B_col_i2j1 = 2*( (i+1)*M + (j+0) )
            B_col_i2j2 = 2*( (i+1)*M + (j+1) )

            ## 4 major points
            Bc_tilde_dok[B_row  , B_col_i1j1  ] = w11
            Bc_tilde_dok[B_row+1, B_col_i1j1+1] = w11
            Bc_tilde_dok[B_row  , B_col_i1j2  ] = w12
            Bc_tilde_dok[B_row+1, B_col_i1j2+1] = w12
            
            Bc_tilde_dok[B_row  , B_col_i2j1  ] = w21
            Bc_tilde_dok[B_row+1, B_col_i2j1+1] = w21
            Bc_tilde_dok[B_row  , B_col_i2j2  ] = w22
            Bc_tilde_dok[B_row+1, B_col_i2j2+1] = w22

    return Bc_tilde_dok

def ConstructRegularizationTerm(B, len_u):
    """
    Returns sparse matrix for regularization term
    B: (N,M,2) regular 2D grid in the texture space
    len_u: works as an offset in row
    """
    assert len(B.shape)==3
    N, M, _ = B.shape
    K = len_u
    Nrows = 2*(K + (N-2)*M + N*(M-2))

    L_tilde_dok = dok_matrix((Nrows, N*M*2))

    ## [CAUTION] vertical (ROW,col)
    cnt = 0
    for i in range(1,M-1):
        for j in range(N):

            B_row = 2 * ( K + cnt )
            B_col_i0j1 = 2 * ((i-1) + (j+0)*M )
            B_col_i1j1 = 2 * ((i+0) + (j+0)*M )
            B_col_i2j1 = 2 * ((i+1) + (j+0)*M )

            L_tilde_dok[B_row  , B_col_i0j1  ] = +1.0
            L_tilde_dok[B_row+1, B_col_i0j1+1] = +1.0
            L_tilde_dok[B_row  , B_col_i1j1  ] = -2.0
            L_tilde_dok[B_row+1, B_col_i1j1+1] = -2.0
            L_tilde_dok[B_row  , B_col_i2j1  ] = +1.0
            L_tilde_dok[B_row+1, B_col_i2j1+1] = +1.0

            cnt += 1
    
    ## [CAUTION] horizontal (row,COL)
    cnt = 0
    for i in range(M):
        for j in range(1,N-1):

            B_row = 2 * ( K + (M-2)*N + cnt )
            B_col_i1j0 = 2 * ((i+0) + (j-1)*M )
            B_col_i1j1 = 2 * ((i+0) + (j+0)*M )
            B_col_i1j2 = 2 * ((i+0) + (j+1)*M )

            L_tilde_dok[B_row  , B_col_i1j0  ] = +1.0
            L_tilde_dok[B_row+1, B_col_i1j0+1] = +1.0
            L_tilde_dok[B_row  , B_col_i1j1  ] = -2.0
            L_tilde_dok[B_row+1, B_col_i1j1+1] = -2.0
            L_tilde_dok[B_row  , B_col_i1j2  ] = +1.0
            L_tilde_dok[B_row+1, B_col_i1j2+1] = +1.0
            
            cnt += 1
    
    return L_tilde_dok

