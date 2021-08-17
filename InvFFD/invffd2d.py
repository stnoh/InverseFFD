from . import ffd2d
from scipy.sparse import dok_matrix

def ConstructBarycentricWeights(B, u):
    """
    Returns sparse matrix based on the vertices' barycentric coordinate
    B: (N,M,2) regular 2D grid in the texture space
    u: 2D vertices in texture space.
    """
    assert len(B.shape)==3
    assert B.shape[2]==2 and u.shape[1]==2

    N, M, _ = B.shape
    K = u.shape[0]
    Nrows = 2*(K + (N-2)*M + N*(M-2))

    Bc_tilde_dok = dok_matrix((Nrows, N*M*2))
    
    for k in range(K):
        u_k = u[k]
        (i,j), (w_u1,w_v1) = ffd2d.getIndexAndWeight(B, u_k)

        ## [CAUTION] this is barycentric coordinate
        w_u0 = 1.0 - w_u1
        w_v0 = 1.0 - w_v1

        ## compute indices for extended matrix
        B_row = 2 * k

        B_col_i0j0 = 2 * ((i+0)*M + (j+0) )
        B_col_i1j0 = 2 * ((i+1)*M + (j+0) )
        B_col_i0j1 = 2 * ((i+0)*M + (j+1) )
        B_col_i1j1 = 2 * ((i+1)*M + (j+1) )

        ## simplified version --- only 4 points are utilized.
        Bc_tilde_dok[B_row  , B_col_i0j0  ] = w_u0 * w_v0
        Bc_tilde_dok[B_row+1, B_col_i0j0+1] = w_u0 * w_v0
        Bc_tilde_dok[B_row  , B_col_i0j1  ] = w_u1 * w_v0
        Bc_tilde_dok[B_row+1, B_col_i0j1+1] = w_u1 * w_v0
        
        Bc_tilde_dok[B_row  , B_col_i1j0  ] = w_u0 * w_v1
        Bc_tilde_dok[B_row+1, B_col_i1j0+1] = w_u0 * w_v1
        Bc_tilde_dok[B_row  , B_col_i1j1  ] = w_u1 * w_v1
        Bc_tilde_dok[B_row+1, B_col_i1j1+1] = w_u1 * w_v1

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

