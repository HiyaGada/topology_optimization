# Elastic sheet optimiszation with payload on a hovering vehicle
#
# Domain is rectangular shape
#
# Governing equation is: div(sigma(u)) + b = 0 on Omega
# u = 0 on Gamma0
# sigma(u).n = T on Gamma
#
# Objective function: J(h)= int_Omega(f.u*dx) + intGamma(T.u*ds)
#
# Adjoint equation is: div(sigma(p)) + j'(u) = 0 on Omega
# p = 0 on Gamma0
# sigma(p).n = k'(u) on Gamma
#
# Adjoint solution is actually p= u

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
#from matplotlib import interactive
import numpy as np

from dolfin import *
from mshr import *

### Global settings

plt.ion() #Enables interactive mode
plt.show()
plt.rcParams['image.cmap'] = 'jet'

set_log_active(False) #Logging turned off

TOL = 1e-4 #For boundaries

### Constants

dim = 2

Lx = 4.0
Ly = 8.0

load_width = 0.05*Lx

E_ = 1.0
nu_ = 0.33

lambda_ = E_/(2.0*(1 + nu_))
mu_     = E_*nu_/((1 + nu_)*(1 - 2*nu_))

eps_void = 1e-3 #For numerical reasons it is chosen close to 0

### External loads

tx = 0.0
ty = (-1e-1)*0.5 #Per point load

bx = 0.0
by = 0.0

### Build mesh

Nx = 100
Ny = 200
mesh = RectangleMesh(
    Point(0.0,0.0), Point(Lx,Ly), Nx, Ny, "crossed" #We choose crossed for a more symmetric result
) 

#plot(mesh, title='Elastic sheet')
#plt.show()
#plt.pause(1000)

### Initialize finite element spaces

Vs = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, 'P', 1)

### Dummy test functions

v = TestFunction(V)
vs = TestFunction(Vs)

### Mark facets for identifying BCs

facets = MeshFunction('size_t', mesh, 1)
facets.set_all(0)

ds = Measure('ds', domain=mesh, subdomain_data=facets)

### Set up Dirichlet boundary condition

class Clamped(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and 
            near(x[1], Ly, TOL)
        )

clamped = Clamped()
clamped.mark(facets, 1)
dbc = DirichletBC(V, Constant((0.0, 0.0)), clamped)

### Set up external surface load 

class LoadLeft(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and
            near(x[0], Lx/4, load_width) and
            near(x[1], 0.0, TOL)
        ) 

loadleft = LoadLeft()
loadleft.mark(facets, 2)

class LoadRight(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and
            near(x[0], 3*Lx/4, load_width) and
            near(x[1], 0.0, TOL)
        ) 

loadright = LoadRight()
loadright.mark(facets, 3)

### Thickness distribution

h0 = 0.2
h = interpolate(Constant(h0), Vs)
hfrac = assemble(h*dx(mesh)) #Average volume

hmin = 0.0
hmax = 1.0

### Constants for Projection and Bisection algorithm

l0 = -0.5*h0
l1 = 1.5*h0
dl = 0.1*h0
lerr = 1e-3

## Interpolating function
def zeta(t):
    return t*t*t

def d_zeta(t):
    return 3*t*t

## Linear elastic strain
def epsilon(u):
    return (0.5*(grad(u) + grad(u).T))

## Linear elastic stress
def sigma(u, h):
    damage = (zeta(h) + (1.0 - zeta(h))*eps_void)
    return damage*(lambda_*tr(epsilon(u))*Identity(dim) + 2*mu_*epsilon(u))

## Derivative of objective function with respect to thickness
def d_obj(u, p, h):
    epsu = epsilon(u)
    epsp = epsilon(p)
    sig = lambda_*tr(epsu)*Identity(dim) + 2*mu_*epsu
    hfact = (1.0 - eps_void)*d_zeta(h)
    return -hfact*inner(sig,epsp)

### Primal problem

T1 = Constant((tx,ty))
T2 = Constant((tx,ty))

b = Constant((bx,by))

L = dot(b,v)*dx + dot(T1,v)*ds(2) + dot(T2,v)*ds(3)

def primal(h):
    u = TrialFunction(V)
    a = inner(sigma(u, h), epsilon(v))*dx
    u = Function(V)
    solve(a == L, u, dbc)
    return u

### Adjoint problem
# Ta1 = Constant((tx,ty))
# Ta2 = Constant((tx,ty))
# ba = Constant((bx,by))
# La = dot(ba,v)*dx + dot(Ta1,v)*ds(2) + dot(Ta2,v)*ds(3)
# def adjoint(h):
#     u = TrialFunction(V)
#     a = inner(sigma(u, h), epsilon(v))*dx
#     u = Function(V)
#     solve(a == La, u, dbc)
#     return u



### Regularization for h

alfah = 0.01
hr = TrialFunction(Vs)
ah = ((alfah**2)*dot(grad(hr),grad(vs))+ hr*vs)*dx 

def regularize_h(h): 
    L = h*vs*dx
    hr = Function(Vs)
    solve(ah == L, hr)
    return hr

## Utility functions
def Max(a, b):
    return (a + b + abs(a - b))/2

def Min(a, b):
    return (a + b - abs(a - b))/2

### Optimization loop

dt = 1.0 #0.25
max_iter = 250
skip = 20

#u_vtk = File('cantilever_deflection_pgd.pvd')
#h_vtk = File('cantilever_pgd.pvd')

for iter in range(max_iter + 1):
    #### Solve primal and adjoint problems
    u = primal(h)
    # p = adjoint(h)

    #### Compute gradient of objective function
    # dJ = d_obj(u,p,h)
    dJ = d_obj(u,u,h)

    #### Update h
    h = h - dt*dJ

    #### Enforce constraints by projection
    ###### Choose initial values of l0 and l1
    proj0 = assemble(Max(hmin, Min(hmax, h + l0))*dx(mesh))
    proj1 = assemble(Max(hmin, Min(hmax, h + l1))*dx(mesh))

    while proj0 > hfrac:
        l0 -= dl
        proj0 = assemble(Max(hmin, Min(hmax, h + l0))*dx(mesh))

    while proj1 < hfrac:
        l1 += dl
        proj1 = assemble(Max(hmin, Min(hmax, h + l1))*dx(mesh))

    ###### Bisection algorithm
    while (l1 - l0) > lerr:
        lmid = (l0 + l1)/2
        projmid = assemble(Max(hmin, Min(hmax, h + lmid))*dx(mesh))

        if projmid < hfrac:
            l0 = lmid
            proj0 = projmid
        else:
            l1 = lmid
            proj1 = projmid

    h = Max(hmin, Min(hmax, h + lmid))

    # h = max(hmin, min(hmax, h))
    h = regularize_h(h)

    #### Dump volume fraction
    hvol = assemble(h*dx(mesh))
    #fh.write('%d\t%f\n' % ((iter + 1), (hvol/hfrac)))
    # hvol = assemble(h*dx(mesh))/(Lx*Ly)
    # fh.write('%d\t%f\n' % ((iter + 1), hvol))

    #### Dump objective function
    #J = assemble(dot(b,u)*dx + dot(T,u)*ds(2))
    #fobj.write('%d\t%f\n' % ((iter + 1), J))

    #print(f'Iteration {iter + 1}: {J}')

    plot(h)
    plt.pause(0.0001)

    if iter % skip == 0:
        u.rename('u','u')
        h.rename('h','h')
        #u_vtk << (u, iter)
        #h_vtk << (h, iter)

## Close files
#fobj.close()
#fh.close()

### Plot
plot(u)
plt.show()
