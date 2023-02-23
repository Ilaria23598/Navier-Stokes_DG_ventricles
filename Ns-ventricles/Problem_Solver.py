import mshr
import meshio
import math
import dolfin
from mpi4py import MPI
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import os
import sys
import getopt
import pandas as pd
import scipy.io

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + '/../utilities')


import ParameterFile_handler as prmh
import BCs_handler
import Mesh_handler
import XDMF_handler
import TensorialDiffusion_handler
#import Formulation_Elements as VFE
import Solver_handler
import HDF5_handler
import Common_main

import ErrorComputation_handler


# PROBLEM CONVERGENCE ITERATIONS
def problemconvergence(filename, conv):

	errors = pd.DataFrame(columns = ['Error_L2_u','Error_DG_u','Error_L2_p','Error_DG_p'])

	for it in range(0,conv):
		# Convergence iteration solver
		errors = problemsolver(filename, it, True, errors)
		errors.to_csv("/home/ilaria/Desktop/Navier-Stokes_ventricles/GaussDIRI.csv")


# PROBLEM SOLVER
def DirichletBoundary(X, param, BoundaryID, time, mesh):

	# Vector initialization
	bc = []

	# Skull Dirichlet BCs Imposition
	period = param['Temporal Discretization']['Problem Periodicity']
		
	if param['Boundary Conditions']['Skull BCs'] == "Dirichlet" :

		# Boundary Condition Extraction Value
		BCsType = param['Boundary Conditions']['Input for Skull BCs']
		BCsValueX = param['Boundary Conditions']['Skull Dirichlet BCs Value (Displacement x-component)']
		
		BCsColumnNameX = param['Boundary Conditions']['File Column Name Skull BCs (x-component)']
		

		# Control of problem dimensionality
		if (mesh.ufl_cell() == triangle):
			BCs = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)

		else:
			BCs = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)

		# Boundary Condition Imposition
		bc.append(DirichletBC(X, BCs, BoundaryID, 1))
		
		return bc


def problemsolver(filename, iteration = 0, conv = False, errors = False):

	# Import the parameters given the filename
	param = prmh.readprmfile(filename)
	parameters["ghost_mode"] = "shared_facet"

	# Handling of the mesh
	mesh = Mesh_handler.MeshHandler(param, iteration)
	hmax=mesh.hmax()
	hmin=mesh.hmin()
	#print(hmax)
	#print(hmin)

	# Importing the Boundary Identifier
	D = mesh.topology().dim()
	BoundaryID = BCs_handler.ImportFaceFunction(param, mesh)

	# Computing the mesh dimensionality
	D = mesh.topology().dim()
	
	# Define function spaces

	# Pressures and Velocity Functional Spaces
	if param['Spatial Discretization']['Method'] == 'DG-FEM':
		V= VectorElement('DG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree velocity']))
		Q= FiniteElement('DG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree pressure']))
    
	elif param['Spatial Discretization']['Method'] == 'CG-FEM':
		V= VectorElement('CG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree velocity']))
		Q= FiniteElement('CG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree pressure']))

	# Mixed FEM Spaces
	W_elem = MixedElement([V,Q])

	# Connecting the FEM element to the mesh discretization
	X = FunctionSpace(mesh, W_elem)
	ds_vent, ds_infl = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)

	# Construction of tensorial space
	X9 = TensorFunctionSpace(mesh, "DG", 0)

	# Diffusion tensors definition
	if param['Model Parameters']['Isotropic Diffusion'] == 'No':

		K = Function(X9)
		K = TensorialDiffusion_handler.ImportPermeabilityTensor(param, mesh, K)

	else:
		K = False
	
	if param['Model Parameters']['Steady/Unsteady'] == 'Unsteady':

		# Time step and normal definitions
		dt = Constant(param['Temporal Discretization']['Time Step'])
		T = param['Temporal Discretization']['Final Time']

		n = FacetNormal(mesh)
		
		# Solution functions definition
		x = Function(X)

		# Previous timestep functions definition
		x_old = Function(X)
		u_old, p_old = x_old.split(deepcopy=True)
		
		mu=param['Model Parameters']['mu']
		rho=param['Model Parameters']['rho']
		
		x_old_old = Function(X)
		u_old_old, p_old_old = x_old_old.split(deepcopy=True)

		# Time Initialization
		t = 0.0
		
		# Initial Condition Construction
		
		x = InitialConditionConstructor(param, mesh, X, x, p_old, u_old)

		# Output file name definition
		if conv:
				OutputFN = param['Output']['Output XDMF File Name'] + '_Ref' + str(iteration) + '_'
		else:
				OutputFN = param['Output']['Output XDMF File Name']

		# Save the time initial solution
		
		XDMF_handler.NSSolutionSave(OutputFN,x, t, param['Temporal Discretization']['Time Step'],param)
			
		# Time advancement of the solution
		x_old.assign(x)
		u_old, p_old = split(x_old)
		u_old_old=u_old
			
		U = Function(X)
		u,p=split(U)
		(v,q)= TestFunctions(X)
		
		it=0
		# Problem Resolution Cicle
		while t < (T-1e-17):
				it+=1
		
		# Temporal advancement
				t += param['Temporal Discretization']['Time Step']

			# Dirichlet Boundary Conditions vector construction
				bc = DirichletBoundary(X, param, BoundaryID, t, mesh)
				
				if param['Model Parameters']['Linear or NonLinear']=='NonLinear':
					if(mesh.ufl_cell()==triangle):
						a,L=VarFormMixUnSteadyNAVIERSTOKES2DSymGrad(param, u, v, p,q,  n, K, mesh,dt,t,u_old,p_old)
					else:
						a,L= VarFormUnsteadyMixNSSymGrad3D(param, u, v, p, q, dt, n, u_old,p_old, K, t, mesh, ds_vent, ds_infl)

					if param['Spatial Discretization']['Method'] == 'CG-FEM':
					#[bci.apply(A) for bci in bc]
					#[bci.apply(b) for bci in bc]
						A,b=assemble_system(a,L,bc)

					U = Solver_handler.NSNonLinearSolver(a,U, L, param)

					if(it%param['Output']['Timestep File Saving']==0):
						XDMF_handler.NSSolutionSave(OutputFN, U, t, param['Temporal Discretization']['Time Step'],param)
								
					if (MPI.comm_world.Get_rank() == 0):
					 	print("Problem at time {:.10f}".format(t), "solved")
				
			# Time advancement of the solution
				
					x_old.assign(U)
					u_old, p_old = x_old.split(deepcopy=True)
					
				else:
					u,p= TrialFunctions(X)
					if param['Domain Definition']['Type of Mesh'] == 'Built-in':
						if(mesh.ufl_cell()==triangle):
							a,L=VarFormMixUnSteadyNAVIERSTOKES2DSymGrad(param, u, v, p,q,  n, K, mesh,dt,t,u_old,p_old)
						else:
							if param['Model Parameters']['S or NS']=='NS':
								a,L=VarFormUnsteadyMixNSSymGrad3D(param, u, v, p, q, dt, n, u_old,p_old, K, t, mesh, ds_vent, ds_infl)
							else:
								a,L=VarFormMixUnSteadySTOKESSymGrad3D(param, dt, u, u_old,p_old,v, p,q, t, n, K, mesh)
							
											
					else: #geometry from file
						if param['Model Parameters']['S or NS']=='NS':
							if conv:
								
								a,L=VarFormUnsteadyMixNSSymGrad3D_CONV_CERV(param, u, v, p, q, dt, n, u_old,p_old, K, t, mesh, ds_vent, ds_infl)
							else:
									a,L=VarFormUnsteadyMixNSSymGradCERV_FINAL(param, u, v, p, q, dt, n, u_old,p_old, K, t, mesh, ds_vent, ds_infl)		
						
						else:
							if conv:
									
								a,L=VarFormUnsteadyStokesSymGrad_MIX_CONV_CERV(param, dt, u, u_old,p_old,v, p,q, t, n, K, mesh, ds_infl,ds_vent)
							else:
								a,L=VarFormMixUnSteadySTOKESSymGradCERV(param, dt, u, u_old,p_old,v, p,q, t, n, K, mesh, ds_infl,ds_vent)
					
					if param['Spatial Discretization']['Method'] == 'CG-FEM':
					#[bci.apply(A) for bci in bc]
					#[bci.apply(b) for bci in bc]
						A,b=assemble_system(a,L,bc)

			# Linear System Resolution
				
					x = Solver_handler.NSSolverSenzaMatrici(a, x, L, param)
					
			# Save the solution at time t
					if(it%param['Output']['Timestep File Saving']==0):
						XDMF_handler.NSSolutionSave(OutputFN, x, t, param['Temporal Discretization']['Time Step'],param)
				
				
					if (MPI.comm_world.Get_rank() == 0):
					 	print("Problem at time {:.10f}".format(t), "solved")
				
			# Time advancement of the solution
					
					x_old.assign(x)
					u_old, p_old = x_old.split(deepcopy=True)
				
		#ii=plot(u_old)
		#plt.colorbar(ii)
		#plt.show()
		#pp=plot(p_old)
		#plt.colorbar(pp)
		#plt.show()
	    
	# Error of approximation
		if conv:
		        	errors = ErrorComputation_handler.Navier_Stokes_Errors(param, x, errors, mesh, iteration, t, n)
		if (MPI.comm_world.Get_rank() == 0):
			         print(errors)
		return errors
		
#***********************STEADY PART***********************************************************#
		
	else:
	
		n = FacetNormal(mesh)
		
		# Solution functions definition
		U = Function(X)
		u,p=split(U)
		
		# Test functions definition
		(v,q)= TestFunctions(X)
		if param['Model Parameters']['Linear or NonLinear'] == 'NonLinear':
			if (mesh.ufl_cell()==triangle):
				a, L = VarFormMixSteadyNAVIERSTOKES2DSymGrad(param, u, v, p,q, n, K, mesh)
			else:
				a,L=VarFormMixSteadyNAVIERSTOKES3DSymGrad(param, u, v, p,q,  n, K, mesh)
			
			if conv:
					OutputFN = param['Output']['Output XDMF File Name'] + '_Ref' + str(iteration) + '_'
			else:
					OutputFN = param['Output']['Output XDMF File Name']
			if param['Spatial Discretization']['Method'] == 'CG-FEM':
					[bci.apply(A) for bci in bc]
					[bci.apply(b) for bci in bc]

			# Linear System Resolution
			
			U=Solver_handler.NSNonLinearSolver(a,U, L, param)
			XDMF_handler.NSSolutionSaveSteady(OutputFN, U)
		
		        # Get sub-functions
			u, p = U.split()
		else:
			
			U = Function(X)
			u,p=split(U)
			(u,p)= TrialFunctions(X)
			
			a, L = VarFormDirichSteadySTOKESSymGrad(param, u, v, p, q, n, K, mesh)
			
			   # Problem Solver Definition
			if conv:

					OutputFN = param['Output']['Output XDMF File Name'] + '_Ref' + str(iteration) + '_'
			else:

					OutputFN = param['Output']['Output XDMF File Name']
			if param['Spatial Discretization']['Method'] == 'CG-FEM':
					[bci.apply(A) for bci in bc]
					[bci.apply(b) for bci in bc]

			# Linear System Resolution
			
			A, bb = assemble_system(a, L)
		
			solve(A, U.vector(), bb)
			u,p=split(U)
		
		iii=plot(u)
		plt.colorbar(iii)
		plt.show()
		ppp=plot(p)
		plt.colorbar(ppp)
		plt.show()
		
				
		if (MPI.comm_world.Get_rank() == 0):
					print("Problem solved!")

		# Error of approximation
		if conv:
				errors = ErrorComputation_handler.Navier_Stokes_ErrorsSteady(param, U, errors, mesh, iteration, n)
			
		       
		if (MPI.comm_world.Get_rank() == 0):
			         print(errors)

		return errors
#########################################################################################################################
#						Variational Formulation Definition					#
#########################################################################################################################

#******************2D*********************************#
def VarFormMixSteadyNAVIERSTOKES2D(param, u, v, p,q,  n, K, mesh):
	class  Bottom(SubDomain):
         def inside(self, x, on_boundary):
           return near(x[1], 0) 
           
	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
  
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
	def tensor_jump(u, n):
		return  (outer(u('+'),n('+')))+outer(u('-'),n('-'))
	def tensor_jump_b(u, n):
		return  outer(u, n)

        
	#def tensor_jump(u, n):
         #  return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        
	#def tensor_jump_b(u, n):
         # return 1/2*(outer(u, n)+ outer(n, u))
          
         
	mu=param['Model Parameters']['mu']
	rho=param['Model Parameters']['rho']
	h = CellDiameter(mesh)
	fx=param['Model Parameters']['Forcing Terms_1x']
	fy=param['Model Parameters']['Forcing Terms_1y']
	fz=param['Model Parameters']['Forcing Terms_1z']
	gx=param['Model Parameters']['g1 Termsx']
	gy=param['Model Parameters']['g1 Termsy']
	gz=param['Model Parameters']['g1 Termsz']
	gNbx=param['Model Parameters']['g2 Termsx']
	gNby=param['Model Parameters']['g2 Termsy']
	gNbz=param['Model Parameters']['g2 Termsz']

	boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
	ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
	bottom=Bottom()
	bottom.mark(boundary_markers, 1)
	f=Expression((fx,fy),degree=6, mu=mu,rho=rho)
	g=Expression((gx,gy),degree=6, mu=mu,rho=rho)
	gNb=Expression((gNbx,gNby),degree=6, mu=mu,rho=rho)
	

	# EQUATION  CG
	
	# DISCONTImuOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		
 
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		m= param['Spatial Discretization']['Polynomial Degree pressure']
		l=param['Spatial Discretization']['Polynomial Degree velocity']
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
		sigmav_b=gammav*l*l*mu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	# EQUATION DG
		
		a=(mu*inner(grad(u), grad(v))*dx)\
		 - (mu*inner(avg(grad(v)), tensor_jump(u, n))*dS) \
		 - (mu*inner(avg(grad(u)), tensor_jump(v, n))*dS)\
		  + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
    		  + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
    		  - (div(v)*p*dx) + (q*div(u)*dx)\
    		   + (inner(avg(p), jump(v, n))*dS)\
    		    - (inner(avg(q), jump(u, n))*dS) \
    		  + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0)) \
     	      - mu*inner(grad(v), tensor_jump_b(u,n))*ds(0) \
	     	 - mu*inner(grad(u),tensor_jump_b(v,n))*ds(0) \
	     	 + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0) \
		    +rho*inner(grad(u)*u, v)*dx\
		    +rho*0.5*div(u)*inner(u,v)*dx\
		   -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
			-rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS\
		   -inner(f, v)*dx \
		  +mu*inner(grad(v), tensor_jump_b(g,n))*ds(0) \
		  -sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0) + dot(g, n)*q*ds(0)\
		    -dot(gNb,v)*ds(1) 
     
		L=0
	return a, L
	
def VarFormMixSteadyNAVIERSTOKES2DSymGrad(param, u, v, p,q,  n, K, mesh):
	class  Bottom(SubDomain):
         def inside(self, x, on_boundary):
           return near(x[1], 0) 
           
	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
  
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        
	def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        
	def tensor_jump_b(u, n):
          return 1/2*(outer(u, n)+ outer(n, u))
	#def tensor_jump(u, n):
         #	return  (outer(u('+'),n('+')))+outer(u('-'),n('-'))
	#def tensor_jump_b(u, n):
         #	return  outer(u, n)

	
	theta = param['Temporal Discretization']['Theta-Method Parameter']
	
	mu=param['Model Parameters']['mu']
	rho=param['Model Parameters']['rho']
	h = CellDiameter(mesh)
	fx=param['Model Parameters']['Forcing Terms_1x']
	fy=param['Model Parameters']['Forcing Terms_1y']
	fz=param['Model Parameters']['Forcing Terms_1z']
	gx=param['Model Parameters']['g1 Termsx']
	gy=param['Model Parameters']['g1 Termsy']
	gz=param['Model Parameters']['g1 Termsz']
	gNbx=param['Model Parameters']['g2 Termsx']
	gNby=param['Model Parameters']['g2 Termsy']
	gNbz=param['Model Parameters']['g2 Termsz']

	boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
	ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
	bottom=Bottom()
	bottom.mark(boundary_markers, 1)
	f=Expression((fx,fy),degree=6, mu=mu,rho=rho)
	g=Expression((gx,gy),degree=6, mu=mu,rho=rho)
	gNb=Expression((gNbx,gNby),degree=6, mu=mu,rho=rho)
	
	# EQUATION  CG
	
	
	# DISCONTImuOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		
 
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		m=param['Spatial Discretization']['Polynomial Degree pressure']
		l=param['Spatial Discretization']['Polynomial Degree velocity']
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
		sigmav_b=gammav*l*l*mu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	# EQUATION DG
		
		a=(2*mu*inner(sym(grad(u)), grad(v))*dx) - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
      + (sigmap*dot(jump(p, n), jump(q, n))*dS) - (div(v)*p*dx) + (q*div(u)*dx) + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS) \
      + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0)) \
      - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds(0) - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds(0) \
      + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0) \
    +rho*inner(grad(u)*u, v)*dx\
    +rho*0.5*div(u)*inner(u,v)*dx\
    -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS\
   -inner(f, v)*dx \
  +2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds(0) -sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0) + dot(g, n)*q*ds(0)\
    -dot(gNb,v)*ds(1) 
     
		L=0
	return a, L

def VarFormMixUnSteadyNAVIERSTOKES2DSymGrad(param, u, v, p,q,  n, K, mesh,dt,time,u_old,p_old):
	class  Bottom(SubDomain):
         def inside(self, x, on_boundary):
           return near(x[1], 0) 
           
	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
  
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        
	def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        
	def tensor_jump_b(u, n):
          return 1/2*(outer(u, n)+ outer(n, u))
        #def tensor_jump(u, n):
        #	return  (outer(u('+'),n('+')))+outer(u('-'),n('-'))
        # def tensor_jump_b(u, n):
        #	return  outer(u, n)

	mu=param['Model Parameters']['mu']
	rho=param['Model Parameters']['rho']
	h = CellDiameter(mesh)
	fx=param['Model Parameters']['Forcing Terms_1x']
	fy=param['Model Parameters']['Forcing Terms_1y']
	fz=param['Model Parameters']['Forcing Terms_1z']
	gx=param['Model Parameters']['g1 Termsx']
	gy=param['Model Parameters']['g1 Termsy']
	gz=param['Model Parameters']['g1 Termsz']
	gNbx=param['Model Parameters']['g2 Termsx']
	gNby=param['Model Parameters']['g2 Termsy']
	gNbz=param['Model Parameters']['g2 Termsz']

	boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
	ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
	bottom=Bottom()
	bottom.mark(boundary_markers, 1)
	f=Expression((fx,fy),degree=6, mu=mu,t=time,rho=rho)
	g=Expression((gx,gy),degree=6, mu=mu,t=time,rho=rho)
	gNb=Expression((gNbx,gNby),degree=6, mu=mu,t=time,rho=rho)
	
	# EQUATION  CG
	
	
	# DISCONTImuOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		m=param['Spatial Discretization']['Polynomial Degree pressure']
		l=param['Spatial Discretization']['Polynomial Degree velocity']
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
		sigmav_b=gammav*l*l*mu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	# EQUATION DG
		
		a=rho*dot(u,v)/dt*dx+(2*mu*inner(sym(grad(u)), grad(v))*dx) - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
      + (sigmap*dot(jump(p, n), jump(q, n))*dS) - (div(v)*p*dx) + (q*div(u)*dx) + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS) \
      + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0)) \
      - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds(0) - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds(0) \
      + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0) \
    +rho*inner(grad(u)*u_old, v)*dx\
    +rho*0.5*div(u_old)*inner(u,v)*dx\
   -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS
     
		L=rho*dot(u_old,v)/dt*dx-(-inner(f, v)*dx \
  +2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds(0) -sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0) + dot(g, n)*q*ds(0)\
    -dot(gNb,v)*ds(1) )
	return a, L
	
#**********************3D*****************************************#	
def VarFormMixSteadyNAVIERSTOKES3DSymGrad(param, u, v, p,q,  n, K, mesh):
	class  Bottom(SubDomain):
         def inside(self, x, on_boundary):
           return near(x[2], 0) 
           
	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
  
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
	def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        
	def tensor_jump_b(u, n):
          return 1/2*(outer(u, n)+ outer(n, u))
         #def tensor_jump(u, n):
        #	return  (outer(u('+'),n('+')))+outer(u('-'),n('-'))
        # def tensor_jump_b(u, n):
        #	return  outer(u, n)

	
	theta = param['Temporal Discretization']['Theta-Method Parameter']
	
	mu=param['Model Parameters']['mu']
	rho=param['Model Parameters']['rho']
	h = CellDiameter(mesh)
	fx=param['Model Parameters']['Forcing Terms_1x']
	fy=param['Model Parameters']['Forcing Terms_1y']
	fz=param['Model Parameters']['Forcing Terms_1z']
	gx=param['Model Parameters']['g1 Termsx']
	gy=param['Model Parameters']['g1 Termsy']
	gz=param['Model Parameters']['g1 Termsz']
	gNbx=param['Model Parameters']['g2 Termsx']
	gNby=param['Model Parameters']['g2 Termsy']
	gNbz=param['Model Parameters']['g2 Termsz']

	boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
	ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
	bottom=Bottom()
	bottom.mark(boundary_markers, 1)
	f=Expression((fx,fy,fz),degree=6, mu=mu,rho=rho)
	g=Expression((gx,gy,gz),degree=6, mu=mu,rho=rho)
	gNb=Expression((gNbx,gNby,gNbz),degree=6, mu=mu,rho=rho)
	
	# EQUATION  CG
	
	
	# DISCONTImuOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		
 
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		m=param['Spatial Discretization']['Polynomial Degree pressure']
		l= param['Spatial Discretization']['Polynomial Degree velocity']
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
		sigmav_b=gammav*l*l*mu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	# EQUATION DG
		
		a=(2*mu*inner(sym(grad(u)), grad(v))*dx) - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
      + (sigmap*dot(jump(p, n), jump(q, n))*dS) - (div(v)*p*dx) + (q*div(u)*dx) + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS) \
      + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0)) \
      - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds(0) - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds(0) \
      + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0) \
    +rho*inner(grad(u)*u, v)*dx\
    +rho*0.5*div(u)*inner(u,v)*dx\
    -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS\
   -inner(f, v)*dx \
  +2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds(0) -sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0) + dot(g, n)*q*ds(0)\
    -dot(gNb,v)*ds(1) 
     
		L=0
	return a, L
	
def VarFormUnsteadyMixNSSymGrad3D(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):
	
        class Bottom(SubDomain):
       	 def inside(self, x, on_boundary):
       	  return on_boundary and near(x[2], 0)
       
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
           
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n)) 
     
        boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
        bottom = Bottom()
        bottom.mark(boundary_markers, 1)
        
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
        rho=param['Model Parameters']['rho']
 
        h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
        gammap=param['Model Parameters']['gammap']
        gammav=param['Model Parameters']['gammav']
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
        sigmav_b=gammav*l*l*mu/h
 
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        time_prev = time-param['Temporal Discretization']['Time Step']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time,rho=rho)
        	
        	g= Expression((gx,gy), degree=2, t=time, mu=mu,rho=rho)
        	g_old= Expression((gx,gy), degree=2, t=time_prev, mu=mu,rho=rho)
        	
        	gNb = Expression((gNbx,gNby), degree=2, t=time, mu=mu,rho=rho)
        	
        else:
        	f = Expression((fx,fy,fz), degree = 6, t=time, mu=mu,rho=rho)
        	
        	g = Expression((gx,gy,gz), degree=6, t=time, mu=mu,rho=rho)
        	g_old = Expression((gx,gy,gz), degree=6, t=time_prev, mu=mu,rho=rho)
        	
        	gNb = Expression((gNbx,gNby,gNbz), degree=6, t=time, mu=mu,rho=rho)
        
        a=rho*dot(u,v)/Constant(dt)*dx\
        +(2*mu*inner(sym(grad(u)), grad(v))*dx) \
        - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
        - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) \
        + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
        + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS)\
        + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0))\
        - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds(0) \
        - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds(0)\
        + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0)\
   	 +rho*inner(grad(u)*u_old, v)*dx\
        +rho*0.5*div(u_old)*inner(u,v)*dx\
        -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        -0.5*rho*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS\
    
       	
        L=rho*dot(u_old,v)/dt*dx\
        -(-inner(f, v)*dx+2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds(0)\
    	-sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0) \
    	+ dot(g, n)*q*ds(0)\
    	-dot(gNb,v)*ds(1))\
   	
        return a, L

def VarFormMixUnSteadySTOKESSymGrad3D(param, dt, u, u_n,p_n,v, p,q, time, n, K, mesh):
	class  Bottom(SubDomain):
         def inside(self, x, on_boundary):
           return near(x[2], 0) 
           
	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
  
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
	#def tensor_jump(u, n):
        #	return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
	#def tensor_jump_b(u, n):
        #	return 1/2*(outer(u, n)+ outer(n, u))
        
        
	def tensor_jump(u, n):
           return (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        
	def tensor_jump_b(u, n):
          return (outer(u,n))
   
	
	theta = param['Temporal Discretization']['Theta-Method Parameter']
	
	mu=param['Model Parameters']['mu']
	rho=param['Model Parameters']['rho']
	h = CellDiameter(mesh)
	fx=param['Model Parameters']['Forcing Terms_1x']
	fy=param['Model Parameters']['Forcing Terms_1y']
	fz=param['Model Parameters']['Forcing Terms_1z']
	gx=param['Model Parameters']['g1 Termsx']
	gy=param['Model Parameters']['g1 Termsy']
	gz=param['Model Parameters']['g1 Termsz']
	gNbx=param['Model Parameters']['g2 Termsx']
	gNby=param['Model Parameters']['g2 Termsy']
	gNbz=param['Model Parameters']['g2 Termsz']

	boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
	ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
	bottom=Bottom()
	bottom.mark(boundary_markers, 1)
	f=Expression((fx,fy,fz),degree=6, mu=mu,t=time,rho=rho)
	g=Expression((gx,gy,gz),degree=6, mu=mu,t=time,rho=rho)
	gNb=Expression((gNbx,gNby,gNbz),degree=6, mu=mu,t=time,rho=rho)
	
	
	# DISCONTImuOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
	
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		m=param['Spatial Discretization']['Polynomial Degree pressure']
		l=param['Spatial Discretization']['Polynomial Degree velocity']
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
		sigmav_b=gammav*l*l*mu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	# EQUATION DG
		
		a=rho*dot(u,v)/Constant(dt)*dx\
		+(2*mu*inner(sym(grad(u)), grad(v))*dx) \
		- (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
		- (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) \
		+ (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
	      + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
	      - (div(v)*p*dx) + (q*div(u)*dx) \
	      + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS) \
	      + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0)) \
	      - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds(0) \
	      - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds(0) \
	      + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0) 
  
     
		L= -(-inner(f, v)*dx-rho*inner(u_n,v)/Constant(dt)*dx \
  		+2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds(0) \
  		-sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0) + dot(g, n)*q*ds(0)\
   		 -dot(gNb,v)*ds(1)) 
	return a, L


def VarFormMixSteadyNAVIERSTOKES3D(param, u, v, p,q,  n, K, mesh):

	class  Bottom(SubDomain):
         def inside(self, x, on_boundary):
           return near(x[2], 0) 
           
	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
  
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        
	def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        
	def tensor_jump_b(u, n):
          return 1/2*(outer(u, n)+ outer(n, u))
          
      #def tensor_jump(u, n):
          # return (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        
	#def tensor_jump_b(u, n):
         # return (outer(u,n))
         
	mu=param['Model Parameters']['mu']
	rho=param['Model Parameters']['rho']
	h = CellDiameter(mesh)
	fx=param['Model Parameters']['Forcing Terms_1x']
	fy=param['Model Parameters']['Forcing Terms_1y']
	fz=param['Model Parameters']['Forcing Terms_1z']
	gx=param['Model Parameters']['g1 Termsx']
	gy=param['Model Parameters']['g1 Termsy']
	gz=param['Model Parameters']['g1 Termsz']
	gNbx=param['Model Parameters']['g2 Termsx']
	gNby=param['Model Parameters']['g2 Termsy']
	gNbz=param['Model Parameters']['g2 Termsz']

	boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
	ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
	bottom=Bottom()
	bottom.mark(boundary_markers, 1)
	f=Expression((fx,fy,fz),degree=6, mu=mu,rho=rho)
	g=Expression((gx,gy,gz),degree=6, mu=mu,rho=rho)
	gNb=Expression((gNbx,gNby,gNbz),degree=6, mu=mu,rho=rho)
	
	
	# DISCONTImuOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters	
 
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		m=param['Spatial Discretization']['Polynomial Degree pressure']
		l=param['Spatial Discretization']['Polynomial Degree velocity']
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
		sigmav_b=gammav*l*l*mu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	

	# EQUATION  CG
	a = (mu*inner(grad(u), grad(v))*dx) \
	+rho*inner(grad(u)*u, v)*dx \
	- (div(v)*p*dx) + (q*div(u)*dx) 
	
	L= inner(f, v)*dx +dot(gNb,v)*ds(1) 
	
	# DISCONTImuOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':
		# Definition of the stabilization parameters
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		m=param['Spatial Discretization']['Polynomial Degree pressure']
		l=param['Spatial Discretization']['Polynomial Degree velocity']
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
		sigmav_b=gammav*l*l*mu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	# EQUATION DG
		a= a +(- (mu*inner(avg(grad(v)), tensor_jump(u, n))*dS) 
		- (mu*inner(avg(grad(u)), tensor_jump(v, n))*dS)\
                + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
                + (sigmap*dot(jump(p, n), jump(q, n))*dS)\
		+ (inner(avg(p), jump(v, n))*dS) \
		- (inner(avg(q), jump(u, n))*dS) \
                + (dot(v, n)*p*ds(0)) \
                - (dot(u, n)*q*ds(0)) \
                - mu*inner(grad(v), tensor_jump_b(u,n))*ds(0) \
                - mu*inner(grad(u),tensor_jump_b(v,n))*ds(0) \
                + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0) \
                +rho*0.5*div(u)*inner(u,v)*dx\
               -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
      	       -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS)
                
     
		L= L+(- mu*inner(grad(v), tensor_jump_b(g,n))*ds(0) \
		+sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0) \
		- dot(g, n)*q*ds(0))
	return a, L
	

def VarFormUnsteadyMixNAVIERSTOKES3D(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):

        class Bottom(SubDomain):
       	 def inside(self, x, on_boundary):
       	  return on_boundary and near(x[2], 0)
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
           
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n)) 
        boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
        bottom = Bottom()
        bottom.mark(boundary_markers, 1)
        
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        time_prev = time-param['Temporal Discretization']['Time Step']
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
        rho=param['Model Parameters']['rho']
 
        h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
        gammap=param['Model Parameters']['gammap']
        gammav=param['Model Parameters']['gammav']
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
        sigmav_b=gammav*l*l*mu/h
 
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time,rho=rho)
        	
        	g= Expression((gx,gy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu)
        	
        	gNb = Expression((gNbx,gNby), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu)
        	
        else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu,rho=rho)
        	
        	g = Expression((gx,gy,gz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu)
        	
        	gNb = Expression((gNbx,gNby,gNbz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu)
        	
        
	# EQUATION  CG    
  
        a= rho*inner(u,v)/Constant(dt)*dx\
        +(mu*inner(grad(u), grad(v))*dx) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0)) \
        +rho*inner(grad(u)*u_old, v)*dx\
        +rho*0.5*div(u_old)*inner(u,v)*dx
         
        L= inner(f, v)*dx\
        +rho*inner(u_old,v)/Constant(dt)*dx\
        +dot(gNb,v)*ds(1)\
        -dot(g, n)*q*ds(0)
  
        
        if param['Spatial Discretization']['Method'] == 'DG-FEM':
        	a=  a + (inner(avg(p), jump(v, n))*dS) \
        	-(inner(avg(q), jump(u, n))*dS)\
        	-(mu*inner(avg(grad(v)), tensor_jump(u, n))*dS)\
        	-(mu*inner(avg(grad(u)), tensor_jump(v, n))*dS)\
        	+ (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
        	+ (sigmap*dot(jump(p, n), jump(q, n))*dS) - mu*inner(grad(v), tensor_jump_b(u,n))*ds(0) \
        	- mu*inner(grad(u),tensor_jump_b(v,n))*ds(0)\
        	+ sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0)\
        	-rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS
        	L= L -mu*inner(grad(v), tensor_jump_b(g,n))*ds(0) \
        	+sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0)
   	
        return a, L



#*************CERVELLO**********************************************************#
def VarFormMixUnSteadySTOKESSymGradCERV(param, dt, u, u_n,p_n,v, p,q, time, n, K, mesh,ds_infl,ds_vent):
	
           
	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
  
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
	def tensor_jump(u, n):
        	return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
	def tensor_jump_b(u, n):
        	return 1/2*(outer(u, n)+ outer(n, u))    
        
	#def tensor_jump(u, n):
          # return (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        
	#def tensor_jump_b(u, n):
         # return (outer(u,n))


	theta = param['Temporal Discretization']['Theta-Method Parameter']
	
	mu=param['Model Parameters']['mu']
	rho=param['Model Parameters']['rho']
	h = CellDiameter(mesh)
	fx=param['Model Parameters']['Forcing Terms_1x']
	fy=param['Model Parameters']['Forcing Terms_1y']
	fz=param['Model Parameters']['Forcing Terms_1z']
	gx=param['Model Parameters']['g1 Termsx']
	gy=param['Model Parameters']['g1 Termsy']
	
	BCsType = param['Boundary Conditions']['Input for Ventricles BCs']
	BCsValueX = param['Boundary Conditions']['Skull Dirichlet BCs Value']
	BCsColumnNameX = param['Boundary Conditions']['File Column Name Ventricles BCs']
	period = param['Temporal Discretization']['Problem Periodicity']
	if (mesh.ufl_cell() == triangle):
		C = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)
	else:
		C= BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)
		
	g=Expression((param['Model Parameters']['g1 Termsz']),degree=6,mu=mu,t=time,c=C)
	gNbx=param['Model Parameters']['g2 Termsx']
	gNby=param['Model Parameters']['g2 Termsy']
	gNbz=param['Model Parameters']['g2 Termsz']

	f=Expression((fx,fy,fz),degree=6, mu=mu,t=time)
	#g=Expression((gx,gy,gz),degree=6, mu=mu,t=time)
	gNb=Expression((gNbx,gNby,gNbz),degree=6, mu=mu,t=time)

	# EQUATION  CG
	
	
	# DISCONTImuOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		m=param['Spatial Discretization']['Polynomial Degree pressure']
		l=param['Spatial Discretization']['Polynomial Degree velocity']
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
		sigmav_b=gammav*l*l*mu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	# EQUATION DG
		
		a=rho*dot(u,v)/Constant(dt)*dx\
		+(2*mu*inner(sym(grad(u)), grad(v))*dx) \
		- (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
		- (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) \
		+ (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
                + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
                - (div(v)*p*dx) + (q*div(u)*dx) \
                + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS) \
                + (dot(v, n)*p*ds_vent) - (dot(u, n)*q*ds_vent) \
                - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds_vent \
                - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds_vent \
                + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds_vent
  
     
		L= -(-inner(f, v)*dx-rho*inner(u_n,v)/Constant(dt)*dx \
                  +2*mu*inner(sym(grad(v)), tensor_jump_b(g*n,n))*ds_vent \
                  -sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g*n,n))*ds_vent \
                  + g*q*ds_vent\
                  -dot(gNb,v)*ds_infl)
	return a, L


def VarFormUnsteadyMixNSSymGradCERV_pre(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):

        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n))

        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
        rho=param['Model Parameters']['rho']
        h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
        gammap=param['Model Parameters']['gammap']
        gammav=param['Model Parameters']['gammav']
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
        sigmav_b=gammav*l*l*mu/h
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        BCsType = param['Boundary Conditions']['Input for Ventricles BCs']
        BCsValueX = param['Boundary Conditions']['Skull Dirichlet BCs Value']
        BCsColumnNameX = param['Boundary Conditions']['File Column Name Ventricles BCs']
        period = param['Temporal Discretization']['Problem Periodicity']
        if (mesh.ufl_cell() == triangle):
        	C = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)
        else:
        	C= BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)
        g=Expression((param['Model Parameters']['g1 Termsz']),degree=6, t=time, mu=mu,rho=rho,c=C)
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time)
        	
        	#g= Expression((gx,gy), degree=2, t=time, mu=mu)
        	
        	gNb = Expression((gNbx,gNby), degree=2, t=time, mu=mu)
        	
        else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu)
        	
        	#g = Expression((gx,gy,gz), degree=2, t=time, mu=mu)
        	
        	gNb = Expression((gNbx,gNby,gNbz), degree=2, t=time, mu=mu)
        
	# EQUATION  CG    
  
        a= rho*inner(u,v)/Constant(dt)*dx\
        + (2*mu*inner(sym(grad(u)), sym(grad(v)))*dx) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (dot(v, n)*p*ds_vent)\
        - (dot(u, n)*q*ds_vent) \
        +rho*inner(grad(u)*u_old, v)*dx\
        +rho*0.5*div(u_old)*inner(u,v)*dx
         
        L= inner(f, v)*dx\
          +rho*inner(u_old,v)/Constant(dt)*dx\
          +dot(gNb,v)*ds_infl\
          -g*q*ds_vent
  
        
        if param['Spatial Discretization']['Method'] == 'DG-FEM':
        	a=  a + (inner(avg(p), jump(v, n))*dS) \
        	- (inner(avg(q), jump(u, n))*dS)\
                - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
                - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS)\
                + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
                + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
                - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds_vent \
                - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds_vent\
                + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds_vent\
                -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS

        	L= L  -(2*mu*inner(sym(grad(v)), tensor_jump_b(g*n,n))*ds_vent) \
        	+(sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g*n,n))*ds_vent)
   	
        return a, L
def VarFormUnsteadyMixNSSymGradCERV(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):

        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n))


  
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
        rho=param['Model Parameters']['rho']
        h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
        gammap=param['Model Parameters']['gammap']
        gammav=param['Model Parameters']['gammav']
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
        sigmav_b=gammav*l*l*mu/h
 
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        BCsType = param['Boundary Conditions']['Input for Ventricles BCs']
        BCsValueX = param['Boundary Conditions']['Skull Dirichlet BCs Value']
        BCsColumnNameX = param['Boundary Conditions']['File Column Name Ventricles BCs']
        period = param['Temporal Discretization']['Problem Periodicity']
        if (mesh.ufl_cell() == triangle):
        	C = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)
        else:
        	C= BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)
        g=Expression((param['Model Parameters']['g1 Termsz']),degree=6, t=time, mu=mu,rho=rho,c=C)
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time)
        	
        	#g= Expression((gx,gy), degree=2, t=time, mu=mu)
        	
        	gNb = Expression((gNbx,gNby), degree=2, t=time, mu=mu)
        	
        else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu)
        	
        	#g = Expression((gx,gy,gz), degree=2, t=time, mu=mu)
        	
        	gNb = Expression((gNbx,gNby,gNbz), degree=2, t=time, mu=mu)
      
        a=rho*dot(u,v)/Constant(dt)*dx\
        +(2*mu*inner(sym(grad(u)), grad(v))*dx) \
        - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
        - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) \
        - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds_vent \
        - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds_vent\
        + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
        + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds_vent\
        + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS)\
        + (dot(v, n)*p*ds_vent) - (dot(u, n)*q*ds_vent)\
        +rho*inner(grad(u)*u_old, v)*dx\
        +rho*0.5*div(u_old)*inner(u,v)*dx\
       -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS
	
        L=rho*dot(u_old,v)/Constant(dt)*dx\
        +inner(f, v)*dx\
        -2*mu*inner(sym(grad(v)), tensor_jump_b(g*n,n))*ds_vent\
    	+sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g*n,n))*ds_vent \
    	- g*q*ds_vent\
    	+dot(gNb,v)*ds_infl
   	
        return a, L
def VarFormUnsteadyMixNSSymGradCERV_FINAL(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):

        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n))

        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
        rho=param['Model Parameters']['rho']
        h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
        gammap=param['Model Parameters']['gammap']
        gammav=param['Model Parameters']['gammav']
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
        sigmav_b=gammav*l*l*mu/h
 
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        BCsType = param['Boundary Conditions']['Input for Ventricles BCs']
        BCsValueX = param['Boundary Conditions']['Skull Dirichlet BCs Value']
        BCsColumnNameX = param['Boundary Conditions']['File Column Name Ventricles BCs']
        period = param['Temporal Discretization']['Problem Periodicity']
        if (mesh.ufl_cell() == triangle):
        	C = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)
        else:
        	C= BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)
        g=Expression((param['Model Parameters']['g1 Termsz']),degree=6, t=time, mu=mu,rho=rho,c=C)
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time)
        	
        	gNb = Expression((gNbx,gNby), degree=2, t=time, mu=mu)
        	
        else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu)
        	   	
        	gNb = Expression((gNbx,gNby,gNbz), degree=2, t=time, mu=mu)
      
        a=rho*dot(u,v)/Constant(dt)*dx\
        +(2*mu*inner(sym(grad(u)), grad(v))*dx) \
        - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
        - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) \
        - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds_vent \
        - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds_vent\
        + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
        + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds_vent\
        + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS)\
        + (dot(v, n)*p*ds_vent) - (dot(u, n)*q*ds_vent)\
        +rho*inner(grad(u)*u_old, v)*dx\
        +rho*0.5*div(u_old)*inner(u,v)*dx\
        -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS
	
        L=rho*dot(u_old,v)/Constant(dt)*dx\
        +inner(f, v)*dx\
        -2*mu*inner(sym(grad(v)), tensor_jump_b(g*n,n))*ds_vent\
    	+sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g*n,n))*ds_vent \
    	- g*q*ds_vent\
    	+dot(gNb,v)*ds_infl
   	
        return a, L

def VarFormUnsteadyMixNSSymGrad3D_CONV_CERV(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):
	
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
           
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n)) 
       
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
        rho=param['Model Parameters']['rho']
 
        h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
        gammap=param['Model Parameters']['gammap']
        gammav=param['Model Parameters']['gammav']
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
        sigmav_b=gammav*l*l*mu/h
 
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time,rho=rho)
        	
        	g= Expression((gx,gy), degree=2, t=time, mu=mu,rho=rho)
        	
        	gNb = Expression((gNbx,gNby), degree=2, t=time, mu=mu,rho=rho)
        	
        else:
        	f = Expression((fx,fy,fz), degree = 6, t=time, mu=mu,rho=rho)
        	
        	g = Expression((gx,gy,gz), degree=6, t=time, mu=mu,rho=rho)
        	
        	gNb = Expression((gNbx,gNby,gNbz), degree=6, t=time, mu=mu,rho=rho)
        
        a=rho*dot(u,v)/dt*dx\
        +(2*mu*inner(sym(grad(u)), grad(v))*dx) \
        - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
        - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) \
        + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
        + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS)\
        + (dot(v, n)*p*ds_vent) - (dot(u, n)*q*ds_vent)\
        - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds_vent \
        - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds_vent\
        + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds_vent\
        +rho*inner(grad(u)*u_old, v)*dx\
        -rho*inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        +rho*0.5*div(u_old)*inner(u,v)*dx\
        -rho*0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS
	
        L=rho*dot(u_old,v)/dt*dx\
        -(-inner(f, v)*dx+2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds_vent\
    	-sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds_vent \
    	+ dot(g, n)*q*ds_vent\
    	-dot(gNb,v)*ds_infl)
   	
        return a, L
        
def VarFormUnsteadyStokesSymGrad_FULLDIRI_CONV_CERV(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):
	
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
           
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n)) 
       
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
        rho=param['Model Parameters']['rho']
 
        h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
        gammap=param['Model Parameters']['gammap']
        gammav=param['Model Parameters']['gammav']
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
        sigmav_b=gammav*l*l*mu/h
 
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time,rho=rho)
        	
        	g= Expression((gx,gy), degree=2, t=time, mu=mu,rho=rho)
        	
        	gNb = Expression((gNbx,gNby), degree=2, t=time, mu=mu,rho=rho)
        	
        else:
        	f = Expression((fx,fy,fz), degree = 6, t=time, mu=mu,rho=rho)
        	
        	g = Expression((gx,gy,gz), degree=6, t=time, mu=mu,rho=rho)
        	
        	gNb = Expression((gNbx,gNby,gNbz), degree=6, t=time, mu=mu,rho=rho)
        
        a=rho*dot(u,v)/Constant(dt)*dx\
        +(2*mu*inner(sym(grad(u)), grad(v))*dx) \
        - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
        - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) \
        + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
        + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS)\
        + (dot(v, n)*p*ds_vent) - (dot(u, n)*q*ds_vent)\
        - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds_vent \
        - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds_vent\
        + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds_vent\
         + (dot(v, n)*p*ds_infl) - (dot(u, n)*q*ds_infl)\
        - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds_infl \
        - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds_infl\
        + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds_infl\
	
        L=rho*dot(u_old,v)/dt*dx\
        -(-inner(f, v)*dx+2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds_vent\
    	-sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds_vent) \
    	-(-inner(f, v)*dx+2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds_infl\
    	-sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds_infl) \
    	- dot(g, n)*q*ds_vent\
    	- dot(g, n)*q*ds_infl\
    	
   	
        return a, L

def VarFormUnsteadyStokesSymGrad_MIX_CONV_CERV(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):
	
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
           
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n)) 
       
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
        rho=param['Model Parameters']['rho']
 
        h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
        gammap=param['Model Parameters']['gammap']
        gammav=param['Model Parameters']['gammav']
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
        sigmav_b=gammav*l*l*mu/h
 
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time,rho=rho)
        	
        	g= Expression((gx,gy), degree=2, t=time, mu=mu,rho=rho)
        	
        	gNb = Expression((gNbx,gNby), degree=2, t=time, mu=mu,rho=rho)
        	
        else:
        	f = Expression((fx,fy,fz), degree = 6, t=time, mu=mu,rho=rho)
        	
        	g = Expression((gx,gy,gz), degree=6, t=time, mu=mu,rho=rho)
        	
        	gNb = Expression((gNbx,gNby,gNbz), degree=6, t=time, mu=mu,rho=rho)
        
        a=rho*dot(u,v)/Constant(dt)*dx\
        +(2*mu*inner(sym(grad(u)), grad(v))*dx) \
        - (2*mu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
        - (2*mu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS) \
        + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
        + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (inner(avg(p), jump(v, n))*dS) - (inner(avg(q), jump(u, n))*dS)\
        + (dot(v, n)*p*ds_vent) - (dot(u, n)*q*ds_vent)\
        - 2*mu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds_vent \
        - 2*mu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds_vent\
        + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds_vent\
	
        L=rho*dot(u_old,v)/dt*dx\
        -(-inner(f, v)*dx+2*mu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds_vent\
    	-sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds_vent \
    	+ dot(g, n)*q*ds_vent\
    	-dot(gNb,v)*ds_infl)
   	
        return a, L



##############################################################################################################################
#				Constructor of Initial Condition from file or constant values 				     #
##############################################################################################################################

def InitialConditionConstructor(param, mesh, X, x, p_old, u_old):

	# Solution Initialization
	p0 = param['Model Parameters']['Initial Condition (Pressure)']
	
	u0 = param['Model Parameters']['Initial Condition (Velocity)']

	if (mesh.ufl_cell()==triangle):
		x0 = Constant((p0, u0[0], u0[1]))

	else:
		x0 = Constant((p0, u0[0], u0[1], u0[2]))

	x = interpolate(x0, X)

	# Initial Condition Importing from Files
	if param['Model Parameters']['Initial Condition from File (Pressure)'] == 'Yes':

		p_old = HDF5.ImportICfromFile(param['Model Parameters']['Initial Condition File Name'], mesh, p__old,param['Model Parameters']['Name of IC Function in File'])
		assign(x.sub(0), p_old)

	return x


######################################################################
#				Main 				     #
######################################################################

if __name__ == "__main__":

	Common_main.main(sys.argv[1:], cwd, '/../physics')

	if (MPI.comm_world.Get_rank() == 0):
		
		print("Problem Solved!")

