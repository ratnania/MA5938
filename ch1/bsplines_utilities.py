# ... imports
from numpy import empty
import numpy as np
import matplotlib.pyplot as plt
# ...

# ==========================================================
def find_span( knots, degree, x ):
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high-1
    else:
        # Perform binary search
        span = (low+high)//2
        while x < knots[span] or x >= knots[span+1]:
            if x < knots[span]:
                high = span
            else:
                low  = span
            span = (low+high)//2
        returnVal = span

    return returnVal

# ==========================================================
def all_bsplines( knots, degree, x, span ):
    left   = empty( degree  , dtype=float )
    right  = empty( degree  , dtype=float )
    values = empty( degree+1, dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values

# ==========================================================
def point_on_bspline_curve(knots, P, x):
    degree = len(knots) - len(P) - 1
    
    span = find_span( knots, degree, x )    
    b    = all_bsplines( knots, degree, x, span )    
    
    c = 0.
    for k in range(0, degree+1):
        c += b[k]*P[span-degree+k]
    return c

# ==========================================================
def translate_curve(knots, P, displacement):
    assert( P.shape[-1] == len(displacement) )
    
    n      = P.shape[0]
    degree = len(knots) - n - 1
    Q = np.zeros_like(P)
    Q[:,:] = P[:,:]
    for i in range(0, n):
        Q[i,:] += displacement
    
    return knots, Q

# ==========================================================
def rotate_curve(knots, P, angle, center=[0.,0.]):
    assert( P.shape[-1] == 2)
    assert( len(center) == 2)
    
    n      = P.shape[0]
    degree = len(knots) - n - 1

    ca = np.cos(angle)
    sa = np.sin(angle)

    Q = np.zeros_like(P)
    for i in range(0, n):
        Q[i,0] = center[0] + ca * ( P[i,0] - center[0] ) - sa * ( P[i,1] - center[1] )
        Q[i,1] = center[1] + sa * ( P[i,0] - center[0] ) + ca * ( P[i,1] - center[1] )
    
    return knots, Q

# ==========================================================
def homothetic_curve(knots, P, alpha, center=[0.,0.]):
    assert( P.shape[-1] == 2)
    assert( len(center) == 2)
    
    n      = P.shape[0]
    degree = len(knots) - n - 1

    Q = np.zeros_like(P)
    for i in range(0, n):
        Q[i,0] = center[0] + alpha * ( P[i,0] - center[0] )
        Q[i,1] = center[1] + alpha * ( P[i,1] - center[1] )
    
    return knots, Q

# ==========================================================
def plot_curve(knots, degree, P, color='b', label='P', with_ctrl_pts=True):
    n      = len(knots) - degree - 1

    nx = 101
    xs = np.linspace(0., 1., nx)

    Q = np.zeros((nx, 2))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)

    plt.plot(Q[:,0], Q[:,1], '-'+color)
    plt.plot(P[:,0], P[:,1], '--ok', linewidth=0.7)

    if with_ctrl_pts:
        for i in range(0, n):
            x,y = P[i,:]
            plt.text(x+0.05,y+0.05,'$\mathbf{'+label+'}_{' + str(i) + '}$')

# ==========================================================
def curve_insert_knot(knots, degree, P, t, times=1):
    """Insert the knots t times in the B-Spline curve defined by (knots, degree, P)."""
    n = len(knots) - degree - 1
    n_new = n + times

    # ... multiplicity of t if already in knots
    m = 0
    for u in knots:
        # TODO add machine precision
        if u == t:
            m += 1
    # ...

    # compute the span
    span = find_span( knots, degree, t )

    # ... create the new knot vector
    knots_new = np.zeros(n_new+degree+1, dtype=np.float)

    knots_new[:span+1] = knots[:span+1]
    knots_new[span+1:span+times+1] = t
    knots_new[span+times+1:] = knots[span+1:]
    # ...

    # ...
    d = P.shape[1]
    P_new = np.zeros((n_new, d), dtype=np.float)

    # temporary array
    R = np.zeros((degree+1, d), dtype=np.float)

    P_new[:span-degree+1] = P[:span-degree+1]
    P_new[span-m+times:] = P[span-m:]
    R[:degree-m+1] = P[span-degree:span-m+1]

    # insert t times
    for j in range(1, times+1):
        l = span - degree + j
        for i in range(0, degree-j-m+1):
            alpha = (t-knots[l+i])/(knots[i+span+1] - knots[l+i])
            R[i] = alpha*R[i+1] + (1. - alpha)*R[i]
        P_new[l] = R[0]
        P_new[span+times-j-m] = R[degree-j-m]

    for i in range(l+1, span-m):
        P_new[i] = R[i-l]
    # ...

    return knots_new, degree, P_new

# ==========================================================
# NOTE only true for open knot vector
def curve_elevate_degree(U_in, p_in, P_in, m=1):
    """elevate the B-Spline degree m times in the B-Spline curve defined by (U_in, degree, P)."""
    # ...
    U_in = list(U_in) # since we will use count method for list

    n_in = len(U_in) - p_in - 1
    k = p_in + 1
    nd = P_in.shape[-1]

    breaks         = np.unique(U_in)
    multiplicities = [U_in.count(t) for t in breaks]
    # ...

    # ... out parameters
    s = len(breaks) - 1

    p_out = p_in + m
    n_out = n_in + s * m
    # ...

    # ... allocate temporary arrays
    Phat   = np.zeros((n_in,p_in+1,nd), dtype=np.float)
    Qhat   = np.zeros((n_out,p_out+1,nd), dtype=np.float)
    alphas = np.ones(p_in+1, dtype=np.float)
    betas  = np.zeros(s, dtype=np.int32)
    # ...

    # ... allocate out arrays
    P_out = np.zeros((n_out,nd), dtype=np.float)
    U_out = np.zeros(n_out+p_out+1, dtype=np.float)
    # ...

    # ... compute alpha coeffs
    for j in range(0, p_in + 1 ):
        for l in range(1, j+1):
            alphas[j] *= ( k - l ) / ( k + m - l )
    # ...

    # ... compute beta coeffs
    betas[1] = multiplicities[1]
    for i in range(2, s):
        betas[i] = betas[i-1] + multiplicities[i]
    # ...

    # ... compute U_out
    i = 0
    for t, mult in zip(breaks, multiplicities):
        U_out[i:i+mult+m] = t
        i += mult+m
    # ...

    # ... STEP 1
    #     compute p(0,0:p-1) and p(beta(:),i)
    Phat[:, 0, :] = P_in[:, :]
    for j in range(1, p_in+1):
        for i in range(0, n_in-j):
            if not( U_in[i+k] == U_in[i+j] ):
                r = U_in[i+k] - U_in[i+j]
                Phat[i,j,:] = ( Phat[i+1, j-1, :] - Phat[i, j-1, :] ) / r

            else:
                Phat[i,j,:] = 0.
    # ...

    # ... STEP 2
    #     compute q(0,0:p-1)
    for j in range(0, p_in+1):
        Qhat[0, j, :] = alphas[j] * Phat[0, j, :]

    #     compute q(beta(l)+lm,j)
    for l in range(1, s):
        ml = multiplicities[l]
        bl = betas[l]
        for j in range(k-ml, k):
            Qhat[bl+l*m,j,:] = alphas[j] * Phat[bl,j,:]

    #     compute q(beta(l)+lm+i,j), i=1,m
    for l in range(1, s):
        bl = betas[l]
        for j in range(1, m+1):
            Qhat[bl+l*m, p_in,:] = Qhat[bl+l*m, p_in]
    # ...

    # ... STEP 3
    #     compute q(i,0)
    k = k + m

    # first we compute control points generated by the first column
    j = k - 1
    while( j >= 1 ):
        for i in range(0, k-j):
            if not( U_out[i+k] == U_out[i+j] ):
                Qhat[i+1,j-1,:] = Qhat[i,j-1,:] + Qhat[i,j,:] * ( U_out[i+k] - U_out[i+j] )
        j -= 1

    #
    for l in range(1, s):
        j = k-1
        bl = betas[l]
        while( j >= 1 ):
            for i in range(bl+l*m, bl+l*m+k-j):
                if not( U_out[i+k] == U_out[i+j] ):
                    Qhat[i+1,j-1,:] = Qhat[i,j-1,:] + Qhat[i,j,:] * ( U_out[i+k] - U_out[i+j] )
            j -= 1
    # ...

    # ... STEP 4
    #     compute P_out
    # TODO put to zero values that are at machine precision
    P_out[:,:] = Qhat[:,0,:]
    # ...

    return U_out, p_out, P_out