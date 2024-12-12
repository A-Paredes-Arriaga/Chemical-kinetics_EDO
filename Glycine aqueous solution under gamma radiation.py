"""
Created on We Apr 10 19:49:27 2024

@author: Paredes-Arriaga, Alejandro; et. al.
email: alejandro.paredes@correo.nucleares.unam.mx

This code solves the ODE system describing the reaction mechanism 
of glycine aqueous solution under gamma irradiation.
For Python version 3.9
You need to have installed the SciPy, NumPy and matplotlib libraries.
Copy the code and run it in any Python interpreter.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# General function
def f(t,y):
    # External source term (f_i)
       # Variables in source term
    idkgy = 300 # Dose intensity [Gy/min]
    Id = idkgy * 6000        # Dose intensity (rad/h)
    MH2O = 18.02        # Molecular mass of water
    nA = 6.022*(10**23) # Avogadro's number
    M_H = 1.00784    # M_i = Molecular mass i sps.
    G_H = 0.5981        # Gi = Radiochemical constant of the i sps.
    M_eaq = 1.00784    
    G_eaq = 2.7013      
    M_OH = 17.007      
    G_OH = 2.7013     
    # External source term equation for each free radical
    # produced by water radiolysis(f_i, equation 2).
    f_H = [((6.2*(10**11))/(3.6*nA))*(M_H/MH2O)*G_H*Id]
    f_eaq = [((6.2*(10**11))/(3.6*nA))*(M_eaq/MH2O)*G_eaq*Id]
    f_OH = [((6.2*(10**11))/(3.6*nA))*(M_OH/MH2O)*G_OH*Id]
    # Rate constants
    k1 = 1.7*10**4
    k2 = 77  
    k3 = 8.8*10**3
    kn = 1.0*10**2
    # Numerical index of each chemical species  
    # Glycine__________________0
    # OH•______________________1
    # H•_______________________2
    # e-_aq____________________3 
    # H_2O_____________________4
    # H_2______________________5 
    # NH3+C•HOO-_______________6
    # NH_3+____________________7 
    # C•H2COO-_________________8
    # •NH2+CH2COO-_____________9
    # (NH_2+=HCCOO-)___________10
    # CH_3COO-_________________11
    # NH_4+____________________12
    # CH(=O)COO-_______________13 
    # HCHO_____________________14
    # CO_2_____________________15
    # Chemical reaction of glycine in aqueous solution under gamma radiation
    r1 = k1 * y[0] * y[1]  
    r2 = k2 * y[0] * y[2]  
    r3 = k3 * y[0] * y[3]
    r4 = kn * y[10]
    r5 = kn * y[7] * y[9]
    r6 = kn * y[11] * y[4]
    # Reaction mechanism are expressed as a system of coupled 
    # ordinary differential equations (in linear form) 
    dAdt = - r1 - r2 -r3 + (r4)       # Gly_______________0
    dBdt = f_OH - r1                  # OH•_______________1
    dCdt = f_H - r2                   # H•________________2
    dDdt = f_eaq - r3                 # e-_aq_____________3 
    dEdt = +r1 - r6                   # H2O_______________4
    dFdt = +r2                        # H_2_______________5 
    dGdt = + r1 + r2 - r5             # NH3+C•HOO-________6
    dHdt = + (r3*0.5) + (r6*0.2)      # NH_3+_____________7 
    dIdt = + (r3*0.5) - r5            # C•H2COO-__________8
    dJdt = + (r3*0.5) - r4*2          # •NH2+CH2COO-______9
    dKdt = + (r4) + r5 -r6            # (NH_2+=HCCOO-)____10
    dLdt = + r5                       # CH_3COO-__________11
    dMdt = +(r6 * 0.8)                # NH_4+_____________12
    dNdt = +(r6 * 0.8)                # CH(=O)COO-________13 
    dOdt = +(r6 * 0.2)                # HCHO______________14
    dPdt = +(r6 * 0.2)                # CO_2______________15
    return [dAdt, dBdt, dCdt, dDdt, dEdt,
            dFdt, dGdt, dHdt, dIdt, dJdt, 
            dKdt, dLdt, dMdt, dNdt, dOdt,
            dPdt]     
#Set sinterval of stegration
t_span = np.array([0, 34000]) 
#Set the steps number of solution
#Returns num evenly spaced samples, calculated over the interval 
times = np.linspace(t_span[0], t_span[1],34000)
#Set initial conditions of each molecule(concentration in mol/L)
y0 = np.array([1*10**-1, 0, 0, 0, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#Solver
sol = solve_ivp(f, t_span, y0, method='Radau', t_eval=times)
#Compute solutions
t = sol.t
A = sol.y[0]  #Gly
B = sol.y[1]  #OH•
C = sol.y[2]  #H•
D = sol.y[3]  #eaq
E = sol.y[4]  #H2O
F = sol.y[5]  #H2
G = sol.y[6]  #H
H = sol.y[7]  #
I = sol.y[8]  #NH3
J = sol.y[9]  #  
K = sol.y[10] #
L = sol.y[11] #
M = sol.y[12] #
N = sol.y[13] #NH4+
O = sol.y[14] # 
P = sol.y[15] #HCHO
#Plot
plt.plot(t, A, 'k')    #numerical solution of glycine
#Some plot details
plt.xlabel('Dose [kGy]')
plt.ylabel('Concentration [mol/L]')
Q = 160 # equivalence of step (h) to Gy
plt.xticks([0*Q,50*Q,100*Q,150*Q, 200*Q],
           [0, 50, 100, 150,200,])
plt.ylim(0.035, 0.11)
plt.legend(['Glycine numerical model \n(short reaction mechanism)'], 
           loc='best')
plt.grid()
print(A[0], A[(20*Q)-1], A[(40*Q)-1], A[(100*Q)-1], A[(200*Q)-1], sep='\n')
