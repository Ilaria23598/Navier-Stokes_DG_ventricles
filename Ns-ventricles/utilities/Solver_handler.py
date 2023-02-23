import mshr
from dolfin import *
import dolfin
from fenics import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import sys
import pandas as pd
import time
import scipy.sparse as sps

############################################################################################################
# Control the input parameters about the type of mesh to be used and call the generation/import procedures #
############################################################################################################
def compute_eig0(solver, neig_min=1, neig_max=1000): 
     neig = neig_min     
     ret = None     
     while True:         
	     neig = min(neig, neig_max)         
	     solver.solve(neig)         
	     try:
	     	ret = solver.get_eigenvalue(0)
	     	break         
	     except Exception:             
	     	pass         
	     if neig >= neig_max:             
	     	raise f"Reached neig_max = {neig_max}"         
	     neig = 2*neig     
     return ret 

def cond(F):     
	A = PETScMatrix()     
	dummy = rhs(F)     
	assemble_system(lhs(F), dummy, A_tensor=A)     
	eigenSolver = SLEPcEigenSolver(A)     
	eigenSolver.parameters["spectrum"]="smallest magnitude"     
	eigen_min, _ = compute_eig0(eigenSolver)         
	eigenSolver.parameters["spectrum"]="largest magnitude"     
	eigen_max, _ = compute_eig0(eigenSolver)     
	return abs(eigen_max / eigen_min) 
  
def NSSolver(A, x, b, param, t,P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	         
                 Ainv = LUSolver(A)
		 
                 Ainv.solve(x.vector(), b)
                
	return x

def NSSolverSteady(a, x, L, param, P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	         
	         A, bb = assemble_system(a, L)
	         solve(A, x.vector(), bb)
	

	return x

def NSSolverSenzaMatrici(a, x, L, param, P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	         
                 solve(a==L, x,solver_parameters={'linear_solver' : 'mumps'})
                 
        #**********PROVA AUTOVALORI, NUMERO CONDIZIONAMENTO**********************
	#A = PETScMatrix() 
	#assemble(a, tensor=A) 
	
	#start = time.time() 
	#eigenSolver = SLEPcEigenSolver(A) 
	#eigenSolver.parameters["spectrum"]="largest magnitude"
	
	#eigenSolver.solve(1) 
	#eigen_max = eigenSolver.get_eigenvalue(0)[0] 
	#eigenSolver.parameters["spectrum"]="smallest magnitude"
	#eigenSolver.parameters["spectral_transform"]="shift-and-invert"
	#eigenSolver.parameters["spectral_shift"]=0.0
	#eigenSolver.solve(1) 
	##eigen_min = eigenSolver.get_eigenvalue(0)[0] 
	#end = time.time() 
	#print(eigen_max)
	#print(eigen_min)
	#print("Condition number {0:.2e} ({1:.2e} s)".format(eigen_max/eigen_min,  end-start)) 
	
	#start = time.time() 
	#cond_numpy = np.linalg.cond(A.array()) 
	#end = time.time() 
	#print("Numpy condition number {0:.2e} ({1:.2e} s)".format(cond_numpy,end-start))
	
	#********************PROVA SALVARE MATRICI******************************************
	#A=assemble(a)
	#mdic={"A":A}
	#A_in=A.array()
	#sio.savemat("ciao.mat",A)
	#ProvaA=as_backend_type(A).array() #funziona ma solo con mesh MOLTO PICCOLE
	#dolfin.parameters.linear_algebra_backend = "Eigen"
	#rows, cols, values = A.data()
	
	#Ma = sps.csr_matrix((values, cols, rows))
	#as_backend_type(A).array()
	#np.savetxt("Prova.txt",assemble(a).array())
	#file_result=XDMFFile("MA.xdmf")
	#file_result.parameters["flush_output"]=True
	#file_result.parameters["functions_share_mesh"]=True
	
	

	return x
	
	
	
def LinearSolver(a, x, L, param, P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	
		solve(a==L,x)
		#solve(a==L,x,solver_parameters={'linear_solver' : 'mumps'})
	

	elif param["Linear Solver"]["Type of Solver"] == "Iterative Solver":

		soltype = param["Linear Solver"]["Iterative Solver"]
		precon = param["Linear Solver"]["Preconditioner"]
		A=assemble(a)
		b=assemble(L)

		solver = PETScKrylovSolver(soltype, precon)

		if P == False:
			solver.set_operator(A)

		else:
			solver.set_operators(A,P)

		solver.parameters["relative_tolerance"] = 1e-8
		solver.parameters["absolute_tolerance"] = 1e-8
		solver.parameters["nonzero_initial_guess"] = True
		solver.parameters["monitor_convergence"] = False
		solver.parameters["report"] = True
		solver.parameters["maximum_iterations"] = 100000

		solver.solve(x.vector(), b)

	return x

       

		
	
def NSNonLinearSolver(a, U, L, param, P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	
		a=a-L
		solve(a==0,U)
		#DF=derivative(a,u)
		
		
		#x=solve(a==0,u,solver_parameters = {"newton_solver":{'maximum_iterations':100000,"linear_solver" : "mumps","relaxation_parameter":0.9,'relative_tolerance':1E-9,'absolute_tolerance':1E-9,"convergence_criterion" : "residual"}})
		

	return U
	
	
