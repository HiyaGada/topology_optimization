# Shape Optimization of a membrane
#
# Domain is rectangular in shape, load is applied on it, and we will be optimizing the thickness of this membrane/domain
#
# Governing equation is: div(h grad(u)) + f = 0 on D
# u = 0 on dD 
#
# Objective function: J(h)= int_D(0.5*{||u-u0||**2}*dx)
#
# Adjoint equation is: div(h grad(p)) - j'(u) = 0 on D 
# p = 0 on dD
#
# Adjoint solution is actually p= -u


from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt


from dolfin import *
from mshr import *

# Initializing domain
La= 1.0 
Lb= 1.0 #length of domain

xc= 0.75*La
yc= 0.5*Lb #position of the load

# Initializing load 
F= 5.0
r= 0.1 #width of the gaussian load 
l0=0
l1=1
lerr= 1e-3

# Build mesh
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)

# Boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, 0, boundary)

#plot(mesh, title="Unit Square")

# Define physical variables

## Displacement 
u = TrialFunction(V) 
p = TrialFunction(V) 
#h = TrialFunction(V)
v = TestFunction(V) 

## Thickness
h0 = 0.25 #initial thickness
h = h0 

hmin = 0.1
hmax = 1
hfrac = assemble(h0*dx(mesh))

# Define load
f = Expression('F*exp(-((pow(x[0]-xc, 2) + pow(x[1] - yc, 2))/(2*pow(r,2))))', degree=1, F=F, xc=xc, yc=yc, r=r)
u0 = Expression('U*exp(-((pow(x[0]-x0, 2) + pow(x[1] - y0, 2))/(2*pow(sigma,2))))', degree=1, U=1, x0=xc, y0=yc, sigma=0.1)

# Regularization problem

def regularization(h):
  epsilon = 0.01
  hr= TrialFunction(V)
  a3= (epsilon**2*dot(grad(hr),grad(v)) + hr*v)*dx
  L3 = dot(h,v)*dx
  hr = Function(V)
  solve(a3==L3,hr) 
  return hr
	
# Define functions Max and Min
def Max(a,b):
    return (a+b+abs(a-b))/2

def Min(a,b):
    return (a+b-abs(a-b))/2

# Optimization loop - GD

dt = 0.01 #GD variable
maxiter = 100

for i in range(maxiter):
	## Solve the primal problem
	u = TrialFunction(V)
	a1 = h*dot(grad(u),grad(v))*dx
	L1 = dot(f,v)*dx

	u = Function(V)
	solve(a1 == L1, u, bc)

	## Solve the adjoint problem
	p = TrialFunction(V)
	a2 = h*dot(grad(p),grad(v))*dx
	L2 = dot(-(u-u0),v)*dx

	p = Function(V)
	solve(a2 == L2, p, bc)

	## Step forward - Computing Gradient

	dJ = dot(grad(u), grad(p))
	h = h - dt*dJ
	
	## Projection operator is employed
	###P(h)= max(hmin, min(hmax, (h + l0)))

	proj0 = assemble(Max(hmin, Min(hmax,h + l0))*dx(mesh))
	proj1 = assemble(Max(hmin, Min(hmax,h + l1))*dx(mesh))

	
	### To choose appropriate starting l0 and l1
	while proj0>hfrac:
		l0 -= 0.1
		proj0 = assemble((Max(hmin,Min(hmax,h+l0))*dx(mesh)))
	while proj1<hfrac:
		l1 += 0.1
		proj1 = assemble((Max(hmin,Min(hmax,h+l1))*dx(mesh)))

	### Bisection algorithm
	while l1-l0 > lerr:
		lmid = (l0+l1)/2
		projmid = assemble(Max(hmin, Min(hmax,h+lmid))*dx(mesh))
		if projmid<hfrac:
			l0 = lmid
			proj0 = projmid
		else:
			l1 = lmid
			proj1 = projmid

	### Assign h
	h = Max(hmin, Min(hmax, h + l0))

	## Regularize
	h = regularization(h)
	plot(h)
	plt.pause(0.01)

## Perform regularization - volume constraints go here

# Plot
plt.show()

