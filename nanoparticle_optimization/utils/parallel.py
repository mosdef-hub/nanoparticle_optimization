from multiprocessing.pool import ThreadPool
import numpy as np
from numpy import vectorize, mgrid, squeeze, asarray, shape, argmin, zeros, arange
import scipy.optimize as opt

'''
Note: The following functions are lifted from https://gist.github.com/radarsat1/
1551648 and provide a parallelized version of Scipy's optimize.brute.
'''

def gridmap1(grid, func, threads):
    Jout = zeros(len(grid[0]))
    pool = ThreadPool(threads)
    def h(j):
        return func(grid[0][j])
    Jout[:] = pool.map(h, range(len(grid[0])))
    return Jout

def gridmap2(grid, func, threads):
    Jout = zeros(grid[0].shape)
    pool = ThreadPool(threads)
    def h(j):
        def g(k):
            fargs = []
            for i in range(grid.shape[0]):
                fargs.append(grid[i,j,k])
            return func(tuple(fargs))
        return map(g, range(grid.shape[1]))
    Jout[:] = pool.map(h, range(grid.shape[2]))
    return Jout

def gridmap3(grid, func, threads):
    Jout = zeros(grid[0].shape)
    pool = ThreadPool(threads)
    def h(j):
        def g(k):
            def f(l):
                fargs = []
                for i in range(grid.shape[0]):
                    fargs.append(grid[i,j,k,l])
                return func(tuple(fargs))
            return asarray(list(map(f, range(grid.shape[1]))))
        return asarray(list(map(g, range(grid.shape[2]))))
    Jout[:] = asarray(list(pool.map(h, range(grid.shape[3]))))
    return Jout

def parbrutemap(func, ranges, gridmap, args=(), Ns=20, full_output=0, finish=opt.fmin, threads=4):
    N = len(ranges)
    if N > 40:
        raise ValueError('Brute Force not possible with more than 40 variables.')
    lrange = list(ranges)
    for k in range(N):
        if type(lrange[k]) is not type(slice(None)):
            if len(lrange[k]) < 3:
                lrange[k] = tuple(lrange[k]) + (complex(Ns),)
            lrange[k] = slice(*lrange[k])
    if (N==1):
        lrange = lrange[0]

    def _scalarfunc(*params):
        params = squeeze(asarray(params))
        return func(params,*args)

    vecfunc = vectorize(_scalarfunc)
    grid = mgrid[lrange]
    if (N==1):
        grid = (grid,)

    Jout = gridmap(grid, func, threads)

    Nshape = shape(Jout)
    indx = argmin(Jout.ravel(),axis=-1)
    Nindx = zeros(N,int)
    xmin = zeros(N,float)
    for k in range(N-1,-1,-1):
        thisN = Nshape[k]
        Nindx[k] = indx % Nshape[k]
        indx = indx / thisN
    for k in range(N):
        xmin[k] = grid[k][tuple(Nindx)]

    Jmin = Jout[tuple(Nindx)]
    if (N==1):
        grid = grid[0]
        xmin = xmin[0]
    if callable(finish):
        vals = finish(func,xmin,args=args,full_output=1, disp=0)
        xmin = vals[0]
        Jmin = vals[1]
        '''
        if vals[-1] > 0:
            print('Warning: Final optimization did not succeed')
        '''
    if full_output:
        return xmin, Jmin, grid, Jout
    else:
        return xmin

# Code lifted from optimize.py in the SciPy packages.
def parbrute(func, ranges, args=(), Ns=20, full_output=0,
             finish=opt.fmin, threads=4):
    """Minimize a function over a given range by brute force, using a thread pool for calculation.
    Supports only up to 3-dimensional argument structures.
    :Parameters:
        func : callable ``f(x,*args)``
            Objective function to be minimized.
        ranges : tuple
            Each element is a tuple of parameters or a slice object to
            be handed to ``numpy.mgrid``.
        args : tuple
            Extra arguments passed to function.
        Ns : int
            Default number of samples, if those are not provided.
        full_output : bool
            If True, return the evaluation grid.
        threads : int
            Number of threads to use (default=4)
    :Returns: (x0, fval, {grid, Jout})
        x0 : ndarray
            Value of arguments to `func`, giving minimum over the grid.
        fval : int
            Function value at minimum.
        grid : tuple
            Representation of the evaluation grid.  It has the same
            length as x0.
        Jout : ndarray
            Function values over grid:  ``Jout = func(*grid)``.
    :Notes:
        Find the minimum of a function evaluated on a grid given by
        the tuple ranges.
    """
    if len(ranges)==1:
        return parbrutemap(func, ranges, gridmap1, args=args, Ns=Ns,
                           full_output=full_output,
                           finish=finish, threads=threads)
    elif len(ranges)==2:
        return parbrutemap(func, ranges, gridmap2, args=args, Ns=Ns,
                           full_output=full_output,
                           finish=finish, threads=threads)
    elif len(ranges)==3:
        return parbrutemap(func, ranges, gridmap3, args=args, Ns=Ns,
                           full_output=full_output,
                           finish=finish, threads=threads)
    else:
        raise Exception('Only argument dimensions of 1, 2, or 3 supported.')
