import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (ds, dx, inner, grad, div, curl, SpatialCoordinate, 
                 as_vector, sin, cos, pi, variable, TrialFunction, 
                 TestFunction, Measure, diff)
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.fem import (dirichletbc, assemble_scalar, form, Function, 
                         Expression, locate_dofs_topological)
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector, apply_lifting, set_bc)
from dolfinx.io import VTXWriter
from basix.ufl import element
from ufl.core.expr import Expr
from dolfinx.io import VTXWriter, XDMFFile, gmshio
from dolfinx.mesh import locate_entities_boundary, create_unit_cube, transfer_meshtag, RefinementOption, refine_plaza, GhostMode

iteration_count = []
residual_norm = []

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 200  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

n = 8
degree = 1

def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))

def monitor(ksp, its, rnorm):
        iteration_count.append(its)
        residual_norm.append(rnorm)
        print("Iteration: {}, preconditioned residual: {}".format(its, rnorm))

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

# Should be topology dim
gdim = domain.geometry.dim
facet_dim = gdim - 1

# Define function spaces
nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)

total_dofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs


# Magnetic Vector Potential
A = TrialFunction(A_space)
v = TestFunction(A_space)


def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
        for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0],
                                        boundaries[1]),
                            boundaries[2])

facets = mesh.locate_entities_boundary(domain, dim=facet_dim,
                                        marker= boundary_marker)
bdofs0 = fem.locate_dofs_topological(A_space, entity_dim=facet_dim, entities=facets)


# Define Exact Solutions for Magnetic Vector Potential and Electric Scalar Potential

def A_ex(x, t):
    return as_vector((cos(pi * x[1]) * sin(pi * t), cos(pi * x[2]) * sin(pi * t), cos(pi * x[0]) * sin(pi * t)))

x = SpatialCoordinate(domain)
aex = A_ex(x, t)

# Impose boundary conditions on the exact solution
u_bc_expr_A = Expression(aex, A_space.element.interpolation_points())
u_bc_A = Function(A_space)
u_bc_A.interpolate(u_bc_expr_A)
bc0_ex = dirichletbc(u_bc_A, bdofs0)

bc = bc0_ex

mu_R = fem.Constant(domain, default_scalar_type(1.0))
sigma = fem.Constant(domain, default_scalar_type(1.0))

# Weak Form

a00 = dt * (1 / mu_R) * inner(curl(A), curl(v)) * dx
a00 += inner((A*sigma), v) * dx 

a = form(a00)

A_mat = assemble_matrix(a, bcs = [bc])
A_mat.assemble()

# Need to interpolate if non-zero initially
A_n = Function(A_space)

j_e = (1 / mu_R) * curl(curl(aex)) + sigma*diff(aex,t)
time_l0 = (sigma * A_n)
f_time_l0 = dt * j_e + time_l0

L0 = inner(f_time_l0,v) * dx

L = form(L0)

b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

A_map = A_space.dofmap.index_map

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

opts = PETSc.Options()  # type: ignore
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

offset = A_space.dofmap.index_map.size_local * A_space.dofmap.index_map_bs

aerr = []
res = []

X = fem.functionspace(domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,)))
A_vis = fem.Function(X)
A_vis.interpolate(A_n)

A_file = io.VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
A_file.write(t.expression().value)

for i in range(num_steps):  
    t.expression().value += d_t

    u_bc_A.interpolate(u_bc_expr_A)

    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                    mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # print(f"b norm = {b.norm()}")
    sol = A_mat.createVecRight()
    residual = A_mat * sol - b
    res.append(residual.norm())

    ksp.solve(b, sol)

    A_n.x.array[:offset] = sol.array_r[:offset]

    A_vis.interpolate(A_n)
    A_file.write(t.expression().value)

A_file.close()

print(f"e_B  = {L2_norm(curl(A_n) - curl(aex))}")
