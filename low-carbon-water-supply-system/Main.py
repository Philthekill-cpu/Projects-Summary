"""
This code was designed by Ping Fan Teng to allow the user to
calculate the carbon cost in sizing the diameter of mains piping, using
an implicit solver for the colebrook equation and experimentally derived 
results to find the changes in roughness over time. values to find must be
saved in the same folder as .csv file, named test, with one column of varying 
diameters, the first row must be labeled Diameter. Secondly, the file for
the roughness aging properties must be given as a column of polynomial
coefficents. The first row must be named Coefficients, the second must be 
the initial roughness, and then in descending order the coefficents must be
given, e.g. C1x^2 + C2x + C3 with initial roughness C0 to be listed in a 
column as C0, C1, C2, C3. The paper used is:
https://www.researchgate.net/publication/340438992_EFFECT_OF_TIME_ON_PIPE_ROUGHNESS
its best to choose the material in the paper closest to yours. This must also
be saved as a .csv file named polyPVC
To activate mulitple diameter analysis as described, simply go to options 
settings and set find_multidiameter = True.
If you wish to lookup one value, put find_diameter = True and change df to be
your desired value.
In the initialise vairables section you will also need to modify parameters as
you see fit for your system
"""
###Carbon Analysis for Mains Piping


##import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root
from scipy.constants import g

##options
#settings
find_diameter = False
find_multidiameter = True
df = 710


##intialise varaiables
#waterproperties
w_Temp = 10 #temperature (degrees celcius)
rho = 999.7 #density (kg/m3)
dyn = 0.0013076 #dymanic viscosity in (Pa.s)
kin = dyn/rho #kinematic viscosity in (m2/s)
#pipe properties
roughness0 = 0.00152 #initial roughness in mm
min_d = 315 #minimum pipe size (mm)
max_d = 1200 #maximum pipe size (mm)
l = 400 #section length (m)
t_coe = 0.9231 #stores wall thickness coefficient
material = 'PE100' #material
material_rho = 1061.81 #material density kg/m3
#flow properties
q_req = 26040 #required flowrate (m3/day)
q = (q_req*1.0)/(24*60*60) #flowrate used within a margin of safety (m3/s)
#counters
step = int((max_d-min_d)/5) #steps in array
T = 25 # model length (years)
h = T/step #step size (years/step)
sum_var = 0 #used to calculate sum
#pump properties
eta = 0.80 #estimated pump efficiency
#carbon data
carbon = 91 #base emission coefficent(gCO2/kwh)
mat_fac = 3.11 #carbon factor of material - kgCo2/kgmat


##initialise arrays
t = np.linspace(0, T, step+1) #time array (years)
d = np.linspace(min_d,max_d, step+1) #Diameter Array (mm)
D = d*t_coe/1000 #stores internal diameters (m)
v = q/(np.pi*((D/2)**2)) #Velocity Array (m/s)
Re = (D*v)/kin #Reynolds Number Array
fr = np.empty((step+1, step+1)) #array to store friction coefficent
r = np.empty_like(t) #array to store roughness (mm)
h_loss = np.empty_like(fr) #stores head losses (m)
p_pump = np.empty_like(h_loss) #stores power requirements (kWh per timestep)
c_timestep = np.empty_like(p_pump) #stores carbon per timestep
c_timeyear = np.empty_like(c_timestep) #stores carbon per for full timestep
total_op_carbon = np.empty_like(d) #stores carbon per for full timestep
vol = np.empty_like(total_op_carbon) #initialise volume array
cap_carb = np.empty_like(vol) #capital carbon value array
total_carbon = np.empty_like(cap_carb) #total carbon over lifecycle
lngth = np.empty_like(d) #array to store varying lengths for investigation


##main code
#define colebrook model
def model(x,i,n):
    return (-2*np.log10((2.51/(Re[i]*np.sqrt(x))) + ((r[i]/1000)/(3.72*D[n]))) - 1.0/np.sqrt(x))

#define implicit colebrook model solver
def colebrook(i,n):
    return root(model, 0.02, args=(i, n))

#function to calculate headlosses using darcy weissbach
def headloss():
    for n in range(step+1):
        for i in range(step+1):
            h_loss[i,n] = fr[i,n]*((l/D[n])*((v[n]**2)/(2*g)))      

#plotter
def plt_carbon(d, total_carbon, cap_carb, total_op_carbon):
    plt.figure(figsize=(10, 6))
    plt.plot(d, total_carbon, label='Total Lifetime Carbon', color='tab:blue')
    plt.plot(d, cap_carb, label='Capital Carbon', linestyle='dashed', color='tab:red')
    plt.plot(d, total_op_carbon, label='Operating Carbon', linestyle='dotted', color='tab:green')
    plt.title(f"Lifetime Carbon Cost vs Diameter for {material} Mains Pipe, Length {l}m over {T} year Operating Period")
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Equivalent Carbon Cost (Tonnes CO2)')
    plt.legend()
    plt.grid(True)
    plt.show()

#defines main function
def main():
    #calculate roughness over time
    read = pd.read_csv('PolyPVC.csv')
    roughness_calc_var = read.values
    r[0] = roughness_calc_var[0]
    for i in range(1,step+1):
        r[i] = roughness_calc_var[0]*((((t[i]/50)**2)*roughness_calc_var[1])+((t[i]/50)*roughness_calc_var[2])+roughness_calc_var[3])
    for n in range(step+1):
        for i in range(step+1):
            fr[n,i] = colebrook(i,n).x[0]
    headloss()
    #calculate power required by pump
    p_pump = (h_loss*rho*g*q)/(1000*eta)
    #calculate carbon per timestep
    c_timestep = p_pump*(carbon*(10**-6)) #tonnes CO2 per hour per timestep
    #calculate carbon for full timestep
    c_timeyear = c_timestep * 365.25 * 24 * h
    #calculate for lifetime operating costs of carbon
    total_op_carbon = np.sum(c_timeyear, axis=0)
    #calculate volume of pipe material
    vol = ((np.pi * ((d / 2000) ** 2)) - (np.pi * ((D / 2) ** 2))) * l
    #calculate capital carbon cost of pipe material
    cap_carb = vol * ((material_rho * mat_fac) / 1000)
    #calculate total carbon cost for diameters
    total_carbon = cap_carb + total_op_carbon
    #finder
    if find_diameter == True:
        indexes = np.where(np.isin(d, df))[0]
        combined_array = np.column_stack((d[indexes], total_op_carbon[indexes], cap_carb[indexes], total_carbon[indexes]))
        #export
        combined_df = pd.DataFrame(combined_array, columns=['Diameter (mm)', 'Lifetime Operating Carbon (Tonnes CO2)', 'Capital Carbon Cost (Tonnes CO2)', 'Equivalent lifetime Carbon Cost (Tonnes CO2)'])
        combined_csv_path = 'combined_array.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined array exported to {combined_csv_path}")
    elif find_multidiameter == True:
        read = pd.read_csv('Test.csv')
        values_to_find = read.values
        indexes = np.where(np.isin(d, values_to_find))[0]
        combined_array = np.column_stack((d[indexes], total_op_carbon[indexes], cap_carb[indexes], total_carbon[indexes]))
        #find minimum value
        column_index = 3
        min_carb_index = np.argmin(combined_array[:, column_index])
        min_carb_dia = combined_array[min_carb_index, 0]
        min_carb = combined_array[min_carb_index, 3]
        print(f"optimal diameter = {min_carb_dia}mm, {min_carb} equivalent tonnes CO2")
        #export
        combined_df = pd.DataFrame(combined_array, columns=['Diameter (mm)', 'Lifetime Operating Carbon (Tonnes CO2)', 'Capital Carbon Cost (Tonnes CO2)', 'Equivalent lifetime Carbon Cost (Tonnes CO2)'])
        combined_csv_path = 'combined_array.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        print()
        print('-----------------------------------------------------------------------')
        print(f"Combined array exported to {combined_csv_path}")
        print('-----------------------------------------------------------------------')
    # Print and plot results
    print(f"steps taken: {step}, minimum diameter: {min_d}mm, maximum diameter: {max_d}mm, length: {l}m")
    print('-----------------------------------------------------------------------')
    print('diameters')
    print('-----------------------------------------------------------------------')
    print(d)
    print('-----------------------------------------------------------------------')
    print('capital carbon')
    print('-----------------------------------------------------------------------')
    print(cap_carb)
    print('-----------------------------------------------------------------------')
    print('operating carbon')
    print('-----------------------------------------------------------------------')
    print(total_op_carbon)
    print('-----------------------------------------------------------------------')
    print('total carbon')
    print('-----------------------------------------------------------------------')
    print(total_carbon)
    # Plot total lifetime carbon vs diameter
    plt_carbon(d, total_carbon, cap_carb, total_op_carbon)

    
##run main code
main()
