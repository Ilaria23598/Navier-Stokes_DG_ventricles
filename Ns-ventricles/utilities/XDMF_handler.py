import mshr
import dolfin
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import math
import ParameterFile_handler as prmh
import pyrameters as PRM


#####################################################################
#		Generation of XDMF Solution File		    #
#####################################################################

def SolutionFileCreator(file_name):

	# XDMF File Creation
	xdmf = XDMFFile(MPI.comm_world, file_name)

	# Allow the sharing of a mesh for different functions
	xdmf.parameters['functions_share_mesh'] = True

	# Not allowing the overwriting of functions
	xdmf.parameters['rewrite_function_mesh'] = False

	return xdmf



#####################################################################
#		Save MPET Solution at given time		    #
#####################################################################

def MPETSolutionSave(filename, x, time, dt):

	# Time Step Reconstruction
	timestep = math.ceil(time/dt)

	# Filename reconstruction
	filenameout = filename + "_"

	if timestep < 1000000:
		filenameout = filenameout + "0"
	if timestep < 100000:
		filenameout = filenameout + "0"
	if timestep < 10000:
		filenameout = filenameout + "0"
	if timestep < 1000:
		filenameout = filenameout + "0"
	if timestep < 100:
		filenameout = filenameout + "0"
	if timestep < 10:
		filenameout = filenameout + "0"

	filenameout = filenameout + str(timestep) + ".xdmf"

	xdmf = SolutionFileCreator(filenameout)

	# Splitting of the solution in the complete space
	pc, pa, pv, pe, u = x.split(deepcopy=True)

	# Rename the functions to improve the understanding in save
	pc.rename("pc", "Capillary Pressure")
	pa.rename("pa", "Arterial Pressure")
	pv.rename("pv", "Venous Pressure")
	pe.rename("pe", "CSF Pressure")
	u.rename("u", "Displacement")

	# File Update
	xdmf.write(pc, time)
	xdmf.write(pa, time)
	xdmf.write(pv, time)
	xdmf.write(pe, time)
	xdmf.write(u,  time)

	xdmf.close()
#####################################################################
#		Save MPET Solution at given time		    #
#####################################################################

def NSSolutionSave(filename, x, t, dt,param):

	# Time Step Reconstruction
	timestep = math.ceil(t/dt)

	# Filename reconstruction
	filenameout = filename + "_"

	if timestep < 1000000:
		filenameout = filenameout + "0"
	if timestep < 100000:
		filenameout = filenameout + "0"
	if timestep < 10000:
		filenameout = filenameout + "0"
	if timestep < 1000:
		filenameout = filenameout + "0"
	if timestep < 100:
		filenameout = filenameout + "0"
	if timestep < 10:
		filenameout = filenameout + "0"

	filenameout = filenameout + str(timestep) + ".xdmf"

	xdmf = SolutionFileCreator(filenameout)

	# Splitting of the solution in the complete space
	u, p = x.split(deepcopy=True)
	#p_ex=Expression((param['Convergence Test']['Exact Solution Pressure']), degree=6,t=t)
	#corr=assemble((p_ex-p)*dx)
	#QQ=FunctionSpace(mesh, 'DG',1)
	#pnew=TrialFunction(QQ)
	#pnew=p+corr
	#p=p+corr

	# Rename the functions to improve the understanding in save
	
	u.rename("u", "Velocity")
	p.rename("p", " Pressure")

	# File Update
	xdmf.write(u, t)
	xdmf.write(p, t)
	#print(t)

	xdmf.close()
def NSSolutionSaveSteady(filename, x):

	# Filename reconstruction
	filenameout = filename + "_"

	filenameout = filenameout + ".xdmf"

	xdmf = SolutionFileCreator(filenameout)

	# Splitting of the solution in the complete space
	u,p = x.split()

	# Rename the functions to improve the understanding in save
	p.rename("p", " Pressure")
	u.rename("u", "Velocity")

	# File Update
	xdmf.write(p,0)
	xdmf.write(u,0)

	xdmf.close()
def NSSolutionSaveSteadyExact(filename, u,p):

	# Filename reconstruction
	filenameout = filename + "_"

	filenameout = filenameout + ".xdmf"

	xdmf = SolutionFileCreator(filenameout)

	# Rename the functions to improve the understanding in save
	
	u.rename("u_ex", "Velocity")
	p.rename("p_ex", " Pressure")

	# File Update
	xdmf.write(p,0)
	xdmf.write(u,0)

	xdmf.close()
#####################################################################
#			Save Blood Darcy Solution		    #
#####################################################################

def BloodDarcySolutionSave(filename, x):

	filenameout = filename + ".xdmf"

	xdmf = SolutionFileCreator(filenameout)

	# Splitting of the solution in the complete space
	pc, pa, pv, = x.split(deepcopy=True)

	# Rename the functions to improve the understanding in save
	pc.rename("pc", "Capillary Pressure")
	pa.rename("pa", "Arterial Pressure")
	pv.rename("pv", "Venous Pressure")

	# File Update
	xdmf.write(pc, 0)
	xdmf.write(pa, 0)
	xdmf.write(pv, 0)

	xdmf.close()

#####################################################################
#	     Save Arterioles-Venules Direction Solution		    #
#####################################################################

def AVDirSolutionSave(filename, x):

	# Filename reconstruction
	filenameout = filename + ".xdmf"

	xdmf = SolutionFileCreator(filenameout)

	x.rename("t", "Normalized Thickness")

	# File Update
	xdmf.write(x, 0)

	xdmf.close()
