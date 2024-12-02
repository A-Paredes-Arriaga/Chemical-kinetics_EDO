"""
Created on We Apr 10 19:49:27 2024

@author: Paredes-Arriaga, Alejandro; et. al.
email: alejandro.paredes@correo.nucleares.unam.mx

This code solves the ODE system describing the reaction mechanism 
of formic acid at pH 1.5 under gamma irradiation.
For Python version 3.9
You need to have installed the scipy, nunpy and matplotlib libraries.
Copy the code and run it in any Python interpreter.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# General function
def f(t,y):
    # External source term (f_i)
       # Variables in source term
    idgy = 0.90             # Dose intensity [Gy/min]
    Id = idgy*6000          # Dose intensity (rad/h) 
    MH2O = 18.02            # Molecular mass of water
    nA = 6.022*(10**23)     # Avogadro's number
    M_H = 1.00784           # M_i = Molecular mass i sps.
    G_H = 3.4               # Gi = Radiochemical constant of the i sps.
    M_OH = 17.007      
    G_OH = 2.7013      
    M_H2O2 = 34.0147      
    G_H2O2 = 0.7043        
    # External source term equation for each free radical
    # produced by water radiolysis(f_i, equation 2).
    f_H = [((6.2*(10**11))/(3.6*nA))*(M_H/MH2O)*G_H*Id]
    f_OH = [((6.2*(10**11))/(3.6*nA))*(M_OH/MH2O)*G_OH*Id]
    f_H2O2 = [((6.2*(10**11))/(3.6*nA))*(M_H2O2/MH2O)*G_H2O2*Id]
    # Rate constants
    k1 = 4.4*10**5
    k2 = 1.4*10**8   
    k3 = 2.1*10**8  
    k4 = 2.6*10**9   
    k5 = 5.0*10**7  
    k6 = 1.0*10**6
    # Numerical index of each chemical species     
        # H°--------0
        # OH°-------1
        # H_2O_2°---2
        # HCOOH-----3 
        # °HCOO-----4
        # H_2-------5
        # CO_2------6 
        # H_2O------7  
    # Chemical reaction of formic acid at pH 1.5 under gamma radiation
    r1 = k1 * y[0] * y[3]
    r2 = k2 * y[1] * y[3]
    r3 = k3 * y[0] * y[4]
    r4 = k4 * y[1] * y[4]
    r5 = k5 * y[2] * y[4]  
    r6 = k6 * y[0] * y[6]
    # System are expressed as a system of coupled 
    # ordinary differential equations (in linear form) 
    dAdt = f_H  - r1 - r3  -r6               # d[H°]/dt     
    dBdt = f_OH - r2 - r4 + r5               # d[°OH]/dt   
    dCdt = f_H2O2 - r5                       # d[H_2O_2°]/dt    
    dDdt = - r1 - r2                         # d[HCOOH]/dt    
    dEdt = + r1 + r2 - r3 - r4 -r5  +r6      # d[°HCOO]/dt    
    dFdt = + r1 + r3                         # d[H_2]/dt    
    dGdt = + r3 + r4 + r5 - r6               # d[CO_2]/dt      
    dHdt = + r2 + r4 + r5                    # d[H_2O]/dt       
    return [dAdt, dBdt, dCdt, dDdt, dEdt, dFdt, dGdt, dHdt]     
#Set initial conditions of each molecule(concentration in mol/L)
y0 = np.array([0, 0, 0, 0.001, 0, 0, 0, 1])
#Set sinterval of stegration 
h_span = np.array([0, 160000]) 
#Set the steps number of solution
#Returns num evenly spaced samples, calculated over the interval [start, stop].
h = np.linspace(h_span[0], h_span[1],160000)
#Solver
sol = solve_ivp(f, h_span, y0, method='BDF',t_eval=h)
#Compute solutions
h = sol.t            
A = sol.y[0]         #d[H°]/dt
B = sol.y[1]         #d[°OH]/dt
C = sol.y[2]         #d[H_2O_2°]/dt
D = sol.y[3]         #d[HCOOH]/dt
E = sol.y[4]         #d[°HCOO]/dt 
F = sol.y[5]         #d[H_2]/dt
G = sol.y[6]         #d[CO_2]/dt
H = sol.y[7]         #d[H_2O]/dt 
#Plot
plt.plot(h, D, 'k-') #Call the solution for any molecule in the system
plt.plot(h, G, 'm--') 
plt.plot(h, F, 'b--') 
#Some plot details
plt.legend(['$HCOOH$', '$CO_2$', '$H_2$'])
plt.xlabel('Dose [Gy]')
plt.ylabel('Concentration [mol/L]')
Q = 80    # equivalence of step (h) to Gy
plt.xticks([0,500*Q, 1000*Q, 1500*Q, 2000*Q],
           [0, 500, 1000, 1500, 2000])
plt.grid()
