from dolfin import *

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

# Define mesh, function space
mesh = Mesh("box_with_dent.xml.gz")
V = FunctionSpace(mesh, "Lagrange", 1)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx

# Assemble stiffness form
A = PETScMatrix()
assemble(a, tensor=A)

# Create eigensolver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

# Extract largest (first) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(0)

print("Largest eigenvalue: ", r)

# Initialize function and assign eigenvector
u = Function(V)
u.vector()[:] = rx

# Plot eigenfunction
plot(u)
interactive()