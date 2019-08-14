#!/usr/bin/env python
# coding: utf-8

# needed imports
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import bmat
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import product

from bsplines    import elements_spans  # computes the span for each element
from bsplines    import make_knots      # create a knot sequence from a grid
from bsplines    import quadrature_grid # create a quadrature rule over the whole 1d grid
from bsplines    import basis_ders_on_quad_grid # evaluates all bsplines and their derivatives on the quad grid
from quadratures import gauss_legendre

#==============================================================================
SplineSpace   = namedtuple('SplineSpace',
                           'knots, degree, nelements, nbasis, points, weights, spans, basis')
HcurlSpace = namedtuple('HcurlSpace', 'spaces, nbasis')
#==============================================================================

#==============================================================================
def create_space(ne, p, order=None, xmin=0., xmax=1.):
    grid  = np.linspace(xmin, xmax, ne+1)
    knots = make_knots(grid, p, periodic=False)
    spans = elements_spans(knots, p)

    # In[6]:
    nelements = len(grid) - 1
    nbasis    = len(knots) - p - 1

    # we need the value a B-Spline and its first derivative
    nderiv = 1

    # create the gauss-legendre rule, on [-1, 1]
    if order is None:
        order = p
    u, w = gauss_legendre( order )

    # for each element on the grid, we create a local quadrature grid
    points, weights = quadrature_grid( grid, u, w )

    # for each element and a quadrature points,
    # we compute the non-vanishing B-Splines
    basis = basis_ders_on_quad_grid( knots, p, points, nderiv )

    return SplineSpace(knots=knots,
                       degree=p,
                       nelements=nelements,
                       nbasis=nbasis,
                       points=points,
                       weights=weights,
                       spans=spans,
                       basis=basis)

#==============================================================================
def create_hcurl_space(nelements, degrees):
    ne1, ne2 = nelements
    p1,   p2 = degrees

    V1 = create_space(ne=ne1, p=p1,   order=p1)
    V2 = create_space(ne=ne2, p=p2,   order=p2)
    X1 = create_space(ne=ne1, p=p1-1, order=p1)
    X2 = create_space(ne=ne2, p=p2-1, order=p2)

    nbasis    = X1.nbasis * V2.nbasis + V1.nbasis * X2.nbasis

    return HcurlSpace(spaces=[(X1, V2), (V1, X2)],
                      nbasis=nbasis)

#==============================================================================
def assemble_matrices(V):

    (X1, V2), (V1, X2) = V.spaces

    # ... sizes
    ne1, ne2 = V1.nelements, V2.nelements
    p1, p2   = V1.degree, V2.degree

    spans_V1, spans_V2 = V1.spans, V2.spans
    spans_X1, spans_X2 = X1.spans, X2.spans

    basis_V1, basis_V2 = V1.basis, V2.basis
    basis_X1, basis_X2 = X1.basis, X2.basis

    weights_1, weights_2 = V1.weights, V2.weights
    points_1, points_2 = V1.points, V2.points

    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    # ...
    A_00 = np.zeros((X1.nbasis*V2.nbasis, X1.nbasis*V2.nbasis))
    A_01 = np.zeros((X1.nbasis*V2.nbasis, V1.nbasis*X2.nbasis))
    A_10 = np.zeros((V1.nbasis*X2.nbasis, X1.nbasis*V2.nbasis))
    A_11 = np.zeros((V1.nbasis*X2.nbasis, V1.nbasis*X2.nbasis))

    M_00 = np.zeros((X1.nbasis*V2.nbasis, X1.nbasis*V2.nbasis))
    M_01 = np.zeros((X1.nbasis*V2.nbasis, V1.nbasis*X2.nbasis))
    M_10 = np.zeros((V1.nbasis*X2.nbasis, X1.nbasis*V2.nbasis))
    M_11 = np.zeros((V1.nbasis*X2.nbasis, V1.nbasis*X2.nbasis))
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        for ie2 in range(0, ne2):
            i_span_V1 = spans_V1[ie1]
            i_span_V2 = spans_V2[ie2]
            i_span_X1 = spans_X1[ie1]
            i_span_X2 = spans_X2[ie2]

            # ... block entries [0, 0] ie: u in X1 x V2 ; v in X1 x V2
            for il_1 in range(0, p1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1):
                        for jl_2 in range(0, p2+1):
                            i1 = i_span_X1 - (p1-1) + il_1
                            i2 = i_span_V2 - p2 + il_2
                            j1 = i_span_X1 - (p1-1) + jl_1
                            j2 = i_span_V2 - p2 + jl_2

                            v_A = 0.0
                            v_M = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):

                                    u1_0  = basis_X1[ie1,jl_1,0,g1]*basis_V2[ie2,jl_2,0,g2]
                                    u1_x1 = basis_X1[ie1,jl_1,1,g1]*basis_V2[ie2,jl_2,0,g2]
                                    u1_x2 = basis_X1[ie1,jl_1,0,g1]*basis_V2[ie2,jl_2,1,g2]

                                    u2_0  = 0.
                                    u2_x1 = 0.
                                    u2_x2 = 0.

                                    v1_0  = basis_X1[ie1,il_1,0,g1]*basis_V2[ie2,il_2,0,g2]
                                    v1_x1 = basis_X1[ie1,il_1,1,g1]*basis_V2[ie2,il_2,0,g2]
                                    v1_x2 = basis_X1[ie1,il_1,0,g1]*basis_V2[ie2,il_2,1,g2]

                                    v2_0  = 0.
                                    v2_x1 = 0.
                                    v2_x2 = 0.

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    curl_u = u2_x1 - u1_x2
                                    curl_v = v2_x1 - v1_x2

                                    v_M += (u1_0*v1_0 + u2_0*v2_0) * wvol
                                    v_A += curl_u * curl_v * wvol

                            I = i1 + X1.nbasis*i2
                            J = j1 + X1.nbasis*j2

                            A_00[I, J] += v_A
                            M_00[I, J] += v_M

            # ...

            # ... block entries [1, 0] ie: u in X1 x V2 ;  v in V1 x X2
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2):
                    for jl_1 in range(0, p1):
                        for jl_2 in range(0, p2+1):
                            i1 = i_span_V1 - p1 + il_1
                            i2 = i_span_X2 - (p2-1) + il_2
                            j1 = i_span_X1 - (p1-1) + jl_1
                            j2 = i_span_V2 - p2 + jl_2

                            v_A = 0.0
                            v_M = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):

                                    u1_0  = basis_X1[ie1,jl_1,0,g1]*basis_V2[ie2,jl_2,0,g2]
                                    u1_x1 = basis_X1[ie1,jl_1,1,g1]*basis_V2[ie2,jl_2,0,g2]
                                    u1_x2 = basis_X1[ie1,jl_1,0,g1]*basis_V2[ie2,jl_2,1,g2]

                                    u2_0  = 0.
                                    u2_x1 = 0.
                                    u2_x2 = 0.

                                    v1_0  = 0.
                                    v1_x1 = 0.
                                    v1_x2 = 0.

                                    v2_0  = basis_V1[ie1,il_1,0,g1]*basis_X2[ie2,il_2,0,g2]
                                    v2_x1 = basis_V1[ie1,il_1,1,g1]*basis_X2[ie2,il_2,0,g2]
                                    v2_x2 = basis_V1[ie1,il_1,0,g1]*basis_X2[ie2,il_2,1,g2]

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    curl_u = u2_x1 - u1_x2
                                    curl_v = v2_x1 - v1_x2

                                    v_M += (u1_0*v1_0 + u2_0*v2_0) * wvol
                                    v_A += curl_u * curl_v * wvol

                            I = i1 + V1.nbasis*i2
                            J = j1 + X1.nbasis*j2

                            A_10[I, J] += v_A
                            M_10[I, J] += v_M
            # ...

            # ... block entries [0, 1] ie: u in V1 x X2 ; v in X1 x V2
            for il_1 in range(0, p1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2):
                            i1 = i_span_X1 - (p1-1) + il_1
                            i2 = i_span_V2 - p2 + il_2
                            j1 = i_span_V1 - p1 + jl_1
                            j2 = i_span_X2 - (p2-1) + jl_2

                            v_A = 0.0
                            v_M = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):

                                    u1_0  = 0.
                                    u1_x1 = 0.
                                    u1_x2 = 0.

                                    u2_0  = basis_V1[ie1,jl_1,0,g1]*basis_X2[ie2,jl_2,0,g2]
                                    u2_x1 = basis_V1[ie1,jl_1,1,g1]*basis_X2[ie2,jl_2,0,g2]
                                    u2_x2 = basis_V1[ie1,jl_1,0,g1]*basis_X2[ie2,jl_2,1,g2]

                                    v1_0  = basis_X1[ie1,il_1,0,g1]*basis_V2[ie2,il_2,0,g2]
                                    v1_x1 = basis_X1[ie1,il_1,1,g1]*basis_V2[ie2,il_2,0,g2]
                                    v1_x2 = basis_X1[ie1,il_1,0,g1]*basis_V2[ie2,il_2,1,g2]

                                    v2_0  = 0.
                                    v2_x1 = 0.
                                    v2_x2 = 0.

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    curl_u = u2_x1 - u1_x2
                                    curl_v = v2_x1 - v1_x2

                                    v_M += (u1_0*v1_0 + u2_0*v2_0) * wvol
                                    v_A += curl_u * curl_v * wvol

                            I = i1 + X1.nbasis*i2
                            J = j1 + V1.nbasis*j2

                            A_01[I, J] += v_A
                            M_01[I, J] += v_M

            # ...

            # ... block entries [1, 1] ie: u in V1 x X2 ; v in V1 x X2
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2):
                            i1 = i_span_V1 - p1 + il_1
                            i2 = i_span_X2 - (p2-1) + il_2
                            j1 = i_span_V1 - p1 + jl_1
                            j2 = i_span_X2 - (p2-1) + jl_2

                            v_A = 0.0
                            v_M = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):

                                    u1_0  = 0.
                                    u1_x1 = 0.
                                    u1_x2 = 0.

                                    u2_0  = basis_V1[ie1,jl_1,0,g1]*basis_X2[ie2,jl_2,0,g2]
                                    u2_x1 = basis_V1[ie1,jl_1,1,g1]*basis_X2[ie2,jl_2,0,g2]
                                    u2_x2 = basis_V1[ie1,jl_1,0,g1]*basis_X2[ie2,jl_2,1,g2]

                                    v1_0  = 0.
                                    v1_x1 = 0.
                                    v1_x2 = 0.

                                    v2_0  = basis_V1[ie1,il_1,0,g1]*basis_X2[ie2,il_2,0,g2]
                                    v2_x1 = basis_V1[ie1,il_1,1,g1]*basis_X2[ie2,il_2,0,g2]
                                    v2_x2 = basis_V1[ie1,il_1,0,g1]*basis_X2[ie2,il_2,1,g2]

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    curl_u = u2_x1 - u1_x2
                                    curl_v = v2_x1 - v1_x2

                                    v_M += (u1_0*v1_0 + u2_0*v2_0) * wvol
                                    v_A += curl_u * curl_v * wvol

                            I = i1 + V1.nbasis*i2
                            J = j1 + V1.nbasis*j2

                            A_11[I, J] += v_A
                            M_11[I, J] += v_M

            # ...
    # ...

    # ...
    A_00 = coo_matrix(A_00)
    A_01 = coo_matrix(A_01)
    A_10 = coo_matrix(A_10)
    A_11 = coo_matrix(A_11)

    A = bmat([[A_00, A_01], [A_10, A_11]])
    # ...

    # ...
    M_00 = coo_matrix(M_00)
    M_01 = coo_matrix(M_01)
    M_10 = coo_matrix(M_10)
    M_11 = coo_matrix(M_11)

    M = bmat([[M_00, M_01], [M_10, M_11]])
    # ...

    return A, M

#==============================================================================
#degrees   = (2,2)
degrees   = (3,3)

#nelements = (2, 2)
nelements = (4, 4)
#nelements = (16, 16)

V = create_hcurl_space(nelements, degrees)

# Assembling the **Stiffness** matrix is then done using
A, M = assemble_matrices(V)

# TODO: implement the function apply_bc
A = apply_bc(A)
M = apply_bc(M)

wh, vh = eigh(A.todense(), M.todense())

# TODO: implement the function exact_eigen
we = exact_eigen()

print(we[:4])
print(wh[:4])

#plt.plot(wh, label="$w_h$")
#plt.plot(we, label="$w_e$")
#plt.legend()
#plt.show()
