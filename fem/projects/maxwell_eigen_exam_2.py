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

from bsplines    import find_span       # computes the span for a point
from bsplines    import elements_spans  # computes the span for each element
from bsplines    import make_knots      # create a knot sequence from a grid
from bsplines    import quadrature_grid # create a quadrature rule over the whole 1d grid
from bsplines    import basis_ders_on_quad_grid # evaluates all bsplines and their derivatives on the quad grid
from quadratures import gauss_legendre

#==============================================================================
SplineSpace   = namedtuple('SplineSpace',
                           'knots, degree, nelements, nbasis, points, weights, spans, basis')
HcurlSpace = namedtuple('HcurlSpace', 'spaces, nbasis')

SplineSurface = namedtuple('SplineSurface', 'knots, degree, points')
SplineMapping = namedtuple('SplineMapping', 'degree, coeffs, spans, basis')
#==============================================================================

#==============================================================================
def make_square(origin=(0,0), length=1.):
    Tu  = [0., 0., 1., 1.]
    Tv  = [0., 0., 1., 1.]
    pu = 1
    pv = 1
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    origin = np.asarray(origin)

    P = np.asarray([[[0.,0.],[0.,1.]],[[1.,0.],[1.,1.]]])
    for i in range(0, 2):
        for j in range(0, 2):
            P[i,j,:] = origin + P[i,j,:]*length

    return SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)

#==============================================================================
def make_L_shape_C1(center=None):
    Tu  = [0., 0., 0., 0.5, 1., 1., 1.]
    Tv  = [0., 0., 0., 1., 1., 1.]

    pu = 2
    pv = 2
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    # ctrl points
    P = np.zeros((nu,nv,2))
    P[:,:,0] = np.asarray([[-1., -0.5, 0.], [-1., -0.707106781186548, 0.], [-1., -0.292893218813452, 0.], [1., 1., 1.]])
    P[:,:,1] = np.asarray([[-1., -1., -1.], [ 1.,  0.292893218813452, 0.], [ 1.,  0.707106781186548, 0.], [1., .5, 0.]])

    if not( center is None ):
        P[:,:,0] += center[0]
        P[:,:,1] += center[1]

    return SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)

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
def create_mapping(V, kind='square', **kwargs):
    if kind == 'square':
        srf = make_square(**kwargs)

    else:
        raise NotImplementedError('{} not implemented'.format(kind))

    (X1, V2), (V1, X2) = V.spaces

    T1, T2 = srf.knots
    p1, p2 = srf.degree

    # ... we will evaluate the bsplines associated to the mapping on the
    #     quadrature grid used for the approximation.
    #     notice that the mapping/space must fulfill a compatibility constraint.
    nderiv = 1
    basis_1 = basis_ders_on_quad_grid( T1, p1, V1.points, nderiv )
    basis_2 = basis_ders_on_quad_grid( T2, p2, V2.points, nderiv )
    basis = (basis_1, basis_2)
    # ...

    # ... we compute the spans with respect to the elements of the space and not
    #     the mapping
    spans_1  = np.zeros( V1.nelements, dtype=int )
    for ie in range(V1.nelements):
        xx = V1.points[ie,:]
        spans_1[ie] = find_span( T1, p1, xx[0] )

    spans_2  = np.zeros( V2.nelements, dtype=int )
    for ie in range(V2.nelements):
        xx = V2.points[ie,:]
        spans_2[ie] = find_span( T2, p2, xx[0] )

    spans = (spans_1, spans_2)
    # ...

    return SplineMapping(degree=srf.degree,
                         coeffs=srf.points,
                         spans=spans,
                         basis=basis)

#==============================================================================
def eval_mapping(basis_1, basis_2,
                 coeff_x, coeff_y,
                 x_values, y_values,
                 x_x1_values, y_x1_values, x_x2_values, y_x2_values):

    p1 = basis_1.shape[0] - 1
    k1 = basis_1.shape[2]
    p2 = basis_2.shape[0] - 1
    k2 = basis_2.shape[2]

    x_values[ : , : ] = 0.0
    y_values[ : , : ] = 0.0
    x_x1_values[ : , : ] = 0.0
    y_x1_values[ : , : ] = 0.0
    x_x2_values[ : , : ] = 0.0
    y_x2_values[ : , : ] = 0.0
    for g1,g2,jl1,jl2 in product(range(0, k1, 1),range(0, k2, 1),range(0, p1 + 1, 1),range(0, p2 + 1, 1)):
        Nj = basis_1[jl1,0,g1]*basis_2[jl2,0,g2]
        Nj_x1 = basis_1[jl1,1,g1]*basis_2[jl2,0,g2]
        Nj_x2 = basis_1[jl1,0,g1]*basis_2[jl2,1,g2]
        x_values[g1, g2] += Nj*coeff_x[jl1, jl2]
        y_values[g1, g2] += Nj*coeff_y[jl1, jl2]
        x_x1_values[g1, g2] += Nj_x1*coeff_x[jl1, jl2]
        y_x1_values[g1, g2] += Nj_x1*coeff_y[jl1, jl2]
        x_x2_values[g1, g2] += Nj_x2*coeff_x[jl1, jl2]
        y_x2_values[g1, g2] += Nj_x2*coeff_y[jl1, jl2]



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
def assemble_matrices(V, mapping):

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

    basis_M1, basis_M2 = mapping.basis
    spans_M1, spans_M2 = mapping.spans
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

    # ...
    x_values    = np.zeros((k1, k2))
    y_values    = np.zeros((k1, k2))
    x_x1_values = np.zeros((k1, k2))
    y_x1_values = np.zeros((k1, k2))
    x_x2_values = np.zeros((k1, k2))
    y_x2_values = np.zeros((k1, k2))
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        for ie2 in range(0, ne2):
            i_span_V1 = spans_V1[ie1]
            i_span_V2 = spans_V2[ie2]
            i_span_X1 = spans_X1[ie1]
            i_span_X2 = spans_X2[ie2]

            # ...
            i_span_M1 = spans_M1[ie1]
            i_span_M2 = spans_M2[ie2]

            coeff_x = mapping.coeffs[i_span_M1 - mapping.degree[0]:i_span_M1 + 1,
                                     i_span_M2 - mapping.degree[1]:i_span_M2 + 1,
                                     0]
            coeff_y = mapping.coeffs[i_span_M1 - mapping.degree[0]:i_span_M1 + 1,
                                     i_span_M2 - mapping.degree[1]:i_span_M2 + 1,
                                     1]

            eval_mapping(basis_M1[ie1,:,:,:], basis_M2[ie2,:,:,:],
                         coeff_x, coeff_y,
                         x_values, y_values,
                         x_x1_values, y_x1_values, x_x2_values, y_x2_values)
            # ...

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

                                    u1  = basis_X1[ie1,jl_1,0,g1]*basis_V2[ie2,jl_2,0,g2]
                                    u1_x1 = basis_X1[ie1,jl_1,1,g1]*basis_V2[ie2,jl_2,0,g2]
                                    u1_x2 = basis_X1[ie1,jl_1,0,g1]*basis_V2[ie2,jl_2,1,g2]

                                    u2  = 0.
                                    u2_x1 = 0.
                                    u2_x2 = 0.

                                    v1  = basis_X1[ie1,il_1,0,g1]*basis_V2[ie2,il_2,0,g2]
                                    v1_x1 = basis_X1[ie1,il_1,1,g1]*basis_V2[ie2,il_2,0,g2]
                                    v1_x2 = basis_X1[ie1,il_1,0,g1]*basis_V2[ie2,il_2,1,g2]

                                    v2  = 0.
                                    v2_x1 = 0.
                                    v2_x2 = 0.

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    # ...
                                    x    = x_values[g1,g2]
                                    y    = y_values[g1,g2]
                                    x_x1 = x_x1_values[g1,g2]
                                    y_x1 = y_x1_values[g1,g2]
                                    x_x2 = x_x2_values[g1,g2]
                                    y_x2 = y_x2_values[g1,g2]
                                    det_jac = x_x1*y_x2 - x_x2*y_x1
                                    inv_jac = 1.0/(x_x1*y_x2 - x_x2*y_x1)

                                    wvol *= abs(det_jac)

                                    u = (u1, u2)
                                    # TODO use preserving transformation to
                                    #      compute (u1,u2) in terms of u as well
                                    #      as curl_u
                                    u1 =
                                    u2 =
                                    curl_u =

                                    v = (v1, v2)
                                    # TODO use preserving transformation to
                                    #      compute (v1,v2) in terms of v as well
                                    #      as curl_v
                                    v1 =
                                    v2 =
                                    curl_v =
                                    # ...

                                    v_M += (u1*v1 + u2*v2) * wvol
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

                                    u1  = basis_X1[ie1,jl_1,0,g1]*basis_V2[ie2,jl_2,0,g2]
                                    u1_x1 = basis_X1[ie1,jl_1,1,g1]*basis_V2[ie2,jl_2,0,g2]
                                    u1_x2 = basis_X1[ie1,jl_1,0,g1]*basis_V2[ie2,jl_2,1,g2]

                                    u2  = 0.
                                    u2_x1 = 0.
                                    u2_x2 = 0.

                                    v1  = 0.
                                    v1_x1 = 0.
                                    v1_x2 = 0.

                                    v2  = basis_V1[ie1,il_1,0,g1]*basis_X2[ie2,il_2,0,g2]
                                    v2_x1 = basis_V1[ie1,il_1,1,g1]*basis_X2[ie2,il_2,0,g2]
                                    v2_x2 = basis_V1[ie1,il_1,0,g1]*basis_X2[ie2,il_2,1,g2]

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    # ...
                                    x    = x_values[g1,g2]
                                    y    = y_values[g1,g2]
                                    x_x1 = x_x1_values[g1,g2]
                                    y_x1 = y_x1_values[g1,g2]
                                    x_x2 = x_x2_values[g1,g2]
                                    y_x2 = y_x2_values[g1,g2]
                                    det_jac = x_x1*y_x2 - x_x2*y_x1
                                    inv_jac = 1.0/(x_x1*y_x2 - x_x2*y_x1)

                                    wvol *= abs(det_jac)

                                    u = (u1, u2)
                                    # TODO use preserving transformation to
                                    #      compute (u1,u2) in terms of u as well
                                    #      as curl_u
                                    u1 =
                                    u2 =
                                    curl_u =

                                    v = (v1, v2)
                                    # TODO use preserving transformation to
                                    #      compute (v1,v2) in terms of v as well
                                    #      as curl_v
                                    v1 =
                                    v2 =
                                    curl_v =
                                    # ...

                                    v_M += (u1*v1 + u2*v2) * wvol
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

                                    u1  = 0.
                                    u1_x1 = 0.
                                    u1_x2 = 0.

                                    u2  = basis_V1[ie1,jl_1,0,g1]*basis_X2[ie2,jl_2,0,g2]
                                    u2_x1 = basis_V1[ie1,jl_1,1,g1]*basis_X2[ie2,jl_2,0,g2]
                                    u2_x2 = basis_V1[ie1,jl_1,0,g1]*basis_X2[ie2,jl_2,1,g2]

                                    v1  = basis_X1[ie1,il_1,0,g1]*basis_V2[ie2,il_2,0,g2]
                                    v1_x1 = basis_X1[ie1,il_1,1,g1]*basis_V2[ie2,il_2,0,g2]
                                    v1_x2 = basis_X1[ie1,il_1,0,g1]*basis_V2[ie2,il_2,1,g2]

                                    v2  = 0.
                                    v2_x1 = 0.
                                    v2_x2 = 0.

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    # ...
                                    x    = x_values[g1,g2]
                                    y    = y_values[g1,g2]
                                    x_x1 = x_x1_values[g1,g2]
                                    y_x1 = y_x1_values[g1,g2]
                                    x_x2 = x_x2_values[g1,g2]
                                    y_x2 = y_x2_values[g1,g2]
                                    det_jac = x_x1*y_x2 - x_x2*y_x1
                                    inv_jac = 1.0/(x_x1*y_x2 - x_x2*y_x1)

                                    wvol *= abs(det_jac)

                                    u = (u1, u2)
                                    # TODO use preserving transformation to
                                    #      compute (u1,u2) in terms of u as well
                                    #      as curl_u
                                    u1 =
                                    u2 =
                                    curl_u =

                                    v = (v1, v2)
                                    # TODO use preserving transformation to
                                    #      compute (v1,v2) in terms of v as well
                                    #      as curl_v
                                    v1 =
                                    v2 =
                                    curl_v =
                                    # ...

                                    v_M += (u1*v1 + u2*v2) * wvol
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

                                    u1  = 0.
                                    u1_x1 = 0.
                                    u1_x2 = 0.

                                    u2  = basis_V1[ie1,jl_1,0,g1]*basis_X2[ie2,jl_2,0,g2]
                                    u2_x1 = basis_V1[ie1,jl_1,1,g1]*basis_X2[ie2,jl_2,0,g2]
                                    u2_x2 = basis_V1[ie1,jl_1,0,g1]*basis_X2[ie2,jl_2,1,g2]

                                    v1  = 0.
                                    v1_x1 = 0.
                                    v1_x2 = 0.

                                    v2  = basis_V1[ie1,il_1,0,g1]*basis_X2[ie2,il_2,0,g2]
                                    v2_x1 = basis_V1[ie1,il_1,1,g1]*basis_X2[ie2,il_2,0,g2]
                                    v2_x2 = basis_V1[ie1,il_1,0,g1]*basis_X2[ie2,il_2,1,g2]

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    # ...
                                    x    = x_values[g1,g2]
                                    y    = y_values[g1,g2]
                                    x_x1 = x_x1_values[g1,g2]
                                    y_x1 = y_x1_values[g1,g2]
                                    x_x2 = x_x2_values[g1,g2]
                                    y_x2 = y_x2_values[g1,g2]
                                    det_jac = x_x1*y_x2 - x_x2*y_x1
                                    inv_jac = 1.0/(x_x1*y_x2 - x_x2*y_x1)

                                    wvol *= abs(det_jac)

                                    u = (u1, u2)
                                    # TODO use preserving transformation to
                                    #      compute (u1,u2) in terms of u as well
                                    #      as curl_u
                                    u1 =
                                    u2 =
                                    curl_u =

                                    v = (v1, v2)
                                    # TODO use preserving transformation to
                                    #      compute (v1,v2) in terms of v as well
                                    #      as curl_v
                                    v1 =
                                    v2 =
                                    curl_v =
                                    # ...

                                    v_M += (u1*v1 + u2*v2) * wvol
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

mapping = create_mapping(V, kind='square', length=np.pi)

# Assembling the **Stiffness** matrix is then done using
A, M = assemble_matrices(V, mapping)

A = apply_bc(A)
M = apply_bc(M)

wh, vh = eigh(A.todense(), M.todense())

we = exact_eigen()

print('> exact  = ', we[:4])
print('> approx = ', wh[:4])

#plt.plot(wh, label="$w_h$")
#plt.plot(we, label="$w_e$")
#plt.legend()
#plt.show()
