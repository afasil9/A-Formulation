#%%
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner, curl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem import dirichletbc
from dolfinx.io import XDMFFile
from ufl.core.expr import Expr
from petsc4py import PETSc
from ufl import SpatialCoordinate, as_vector, sin, pi, curl
from dolfinx.fem import assemble_scalar, form, Function
from matplotlib import pyplot as plt
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import petsc, Expression, locate_dofs_topological
from dolfinx.io import VTXWriter
from dolfinx.cpp.fem.petsc import (discrete_gradient,
                                   interpolation_matrix)
from basix.ufl import element
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from basix.ufl import element
from dolfinx import fem, io, la, default_scalar_type
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import Function, form, locate_dofs_topological, petsc
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    curl,
    cos,
    inner,
    cross,
)
import ufl
import sys
from ufl import variable
from scipy.linalg import norm


def monitor(ksp, its, rnorm):
        iteration_count.append(its)
        residual_norm.append(rnorm)
        print("Iteration: {}, preconditioned residual: {}".format(its, rnorm))

comm = MPI.COMM_WORLD
def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()

def L2_norm(v: Expr):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))

degree = 1

n = 4

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 200  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
gdim = domain.geometry.dim
facet_dim = gdim - 1 #Topological dimension 

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

alpha_in = 1.0 #Magnetic Permeability
beta_in = 1.0 #Conductivity -> This is set to 0 for Magnetostatic problems

nu = fem.Constant(domain, default_scalar_type(alpha_in))
sigma = fem.Constant(domain, default_scalar_type(beta_in))

gdim = domain.geometry.dim
top_dim = gdim - 1 #Topological dimension 

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = fem.functionspace(domain, nedelec_elem)

# total_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
# print(total_dofs)

x = SpatialCoordinate(domain)

def A_ex(x, t):
    return as_vector((
        x[1]**2 + x[0] * t, 
        x[2]**2 + x[1] * t, 
        x[0]**2 + x[2] * t))

x = SpatialCoordinate(domain)

uex = A_ex(x,t)


#Manually calculating the RHS

f0 = as_vector((
    -2 + x[0],
    -2 + x[1],
    -2 + x[2])
)

# Alternate using ufl
# f0 = curl(curl(uex)) + sigma * ufl.diff(uex,t)

u0  = ufl.TrialFunction(V)
v0 = ufl.TestFunction(V)

facets = mesh.locate_entities_boundary(domain, dim=(domain.topology.dim - 1),
                                    marker=lambda x: np.isclose(x[0], 0.0)|np.isclose(x[0], 1.0)|np.isclose(x[1], 0.0)|np.isclose(x[1], 1.0)|
                                        np.isclose(x[2], 0.0)|np.isclose(x[2], 1.0))
dofs = fem.locate_dofs_topological(V=V, entity_dim=top_dim, entities=facets)


# Initial conditions
u_n = Function(V)
u_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(u_expr)


bdofs0 = fem.locate_dofs_topological(V, entity_dim=facet_dim, entities=facets)
u_bc_expr_V = Expression(uex, V.element.interpolation_points())
u_bc_V = Function(V)
u_bc_V.interpolate(u_bc_expr_V)
bc_ex = dirichletbc(u_bc_V, bdofs0)
bc = bc_ex


a00 = dt * nu * inner(curl(u0), curl(v0)) * dx
a00 += inner((u0*sigma), v0) * dx 

a = form(a00)

L0 = dt* inner(f0, v0) *dx + sigma * inner(u_n, v0) * dx
L = form(L0)
# Solver steps

A_mat = assemble_matrix(a, bcs = [bc])
A_mat.assemble()

b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])


ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
ksp.setTolerances(rtol=1e-10)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

opts = PETSc.Options()
opts[f"{pc.prefix}pc_hypre_ams_cycle_type"] = 7
opts[f"{pc.prefix}pc_hypre_ams_tol"] = 0
opts[f"{pc.prefix}pc_hypre_ams_max_iter"] = 1
opts[f"{pc.prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
opts[f"{pc.prefix}pc_hypre_ams_print_level"] = 1
opts[f"{pc.prefix}pc_hypre_ams_amg_alpha_options"] = "10,1,3"
opts[f"{pc.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,3"
opts[f"{pc.prefix}pc_hypre_ams_print_level"] = 0


# Build discrete gradient
V_CG = fem.functionspace(domain, ("CG", degree))._cpp_object
G = discrete_gradient(V_CG, V._cpp_object)
G.assemble()
pc.setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(V)
    cvec_0.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
                                            np.zeros_like(x[0]),
                                            np.zeros_like(x[0]))))
    cvec_1 = Function(V)
    cvec_1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                            np.ones_like(x[0]),
                                            np.zeros_like(x[0]))))
    cvec_2 = Function(V)
    cvec_2.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                            np.zeros_like(x[0]),
                                            np.ones_like(x[0]))))
    pc.setHYPRESetEdgeConstantVectors(cvec_0.vector,
                                        cvec_1.vector,
                                        cvec_2.vector)
else:
    Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)

ksp.setFromOptions()

ksp.setUp()
pc.setUp()

uh = fem.Function(V)

iteration_count = []
residual_norm = []

print("norm of bc", L2_norm(u_bc_V))
print("norm of uex", L2_norm(uex))
print("norm of u_n", L2_norm(u_n))

# ksp.setMonitor(monitor)
ksp.solve(b, uh.vector)
res = A_mat * uh.vector - b
# print("Residual norm: ", res.norm())
u_n.x.array[:] = uh.x.array

print("norm of u_n after solve", L2_norm(u_n))
#%%
for i in range(num_steps):  
    t.expression().value += d_t

    u_bc_V.interpolate(u_bc_expr_V)

    with b.localForm() as loc:
        loc.set(0)

    # print(L2_norm(uex - u_n))
    
    b = petsc.assemble_vector(L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                    mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    ksp.solve(b, uh.x.petsc_vec)

    u_n.x.array[:] = uh.x.array


e = L2_norm(uh - uex)
# par_print(comm, e)
par_print(comm, f"||u - u_e||_L^2(Omega) = {e}")
