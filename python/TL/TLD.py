import numpy as np
import scipy.special as sc
import scipy.constants
from scipy.special import lambertw, exp1
import matplotlib.pyplot as plt
import iminuit

Npeaks = 5   #number of peaks
Npeak_pars = 4  #number of parameters per peak
N = 0

k=scipy.constants.physical_constants['Boltzmann constant in eV/K']
K=k[0]
#Define theoretical function
def TLD(T, AN, TM, E, B):
    DD = (-E / (K*T)) 
    DM = (-E / (K*TM))
    EI = sc.expi(DD)
    EIM = sc.expi(DM)
    FF = T * np.exp(DD) + (E / K) * EI
    FFM = TM * np.exp(DM) + (E / K) * EIM
    ZZM = (B / (1 - B))-np.log((1 - B) / B)+(E / (K * TM * TM))*np.exp(E/(K*TM))*((FFM)/(1 - 1.05 * np.power(B, 1.26)))
    ZM = np.exp(ZZM)
    LAM1M = np.real(sc.lambertw(ZM))
    ZZ = (B / (1 - B)) - np.log((1 - B) / B) + (E / (K * TM * TM)) * np.exp(E / (K * TM)) * ((FF) / (1 - 1.05 * np.power(B, 1.26)))
    if np.any(ZZ < 700.0):
        Z = np.exp(ZZ)
        LAM1 = np.real(sc.lambertw(Z))
    else:
        LAM1 = ZZ - np.log(ZZ)
    FTE = np.exp(((E / (K * T)) * ((T - TM) / (TM))))
    SOMA = FTE * ((LAM1M + np.power(LAM1M, 2)) / (LAM1 + np.power(LAM1, 2)))
    return AN * SOMA

def BGR(T, ENTA, D0, ALA):
    return ENTA + D0 * 10e-8 * np.exp(T / ALA)

# Define theoretical function for total peak (function to fit)
def TL_theo1(T, par):
    TL = 0.0
    # first sum up all the peak contributions
    for i in range(Npeaks):
        TL += TLD(T, par[Npeak_pars * i + 0], par[Npeak_pars * i + 1], par[Npeak_pars * i + 2], par[Npeak_pars * i + 3])

    # then add the background
    TL += BGR(T, par[Npeak_pars * Npeaks + 0], par[Npeak_pars * Npeaks + 1], par[Npeak_pars * Npeaks + 2])
    return TL

def TL_theo(t, par):
    par_array = np.frombuffer(par,
                              dtype=float,
                              count=Npeak_pars * Npeaks + 3)
    return TL_theo1(t[0], par_array)

# Read experimental data file
dirdata = "/python/TLD/" #Change file path
datafile = input("Experimental data file name : ")    #Enter filename with .dat or .txt
infile = dirdata + datafile
print("Data file to read: ", infile)

channel, T_data, TL_data = np.loadtxt(infile,
                                dtype=[('', 'i4'), ('', 'f8'),
                                       ('', 'f8')],
                                usecols=(0, 1, 2),
                                unpack=True)
N=len(channel)

#Print some data lines on screen 
for i in range(0,10):
    print("{} {} {}".format(channel[i], T_data[i], TL_data[i])) # print first 10 data lines
print("...")
for i in range(N - 10, N):
    print("{} {} {}".format(channel[i], T_data[i], TL_data[i])) # print last 10 data lines
print("Data points: %f" %N)

# Read parameters file
dirpara = "/python/TL/"
parainfile = input("Parameters file name :  ") # Write parameters filename with .txt
parafile = dirpara + parainfile
print("The filename you entered was: ", parafile)

peaks = np.loadtxt(parafile,
                       comments='#',
                       usecols=(0, 1, 2, 3),
                       dtype=float)
for i in range(Npeaks*Npeak_pars + 3):
    print("Par line %d: %7.4f %7.4f %7.4f %7.4f" %(i+1, peaks[i][0], peaks[i][1], peaks[i][2], peaks[i][3]))

#Select the Low and Upper limits of the Curve Fitting
# LL = int(input("Enter low limit LL: "))
# LU = int(input("Enter upper limit LU: "))
# L = LL
# N = LU - LL - 1

# for i in range(0,N):
#     TL_data[i] = TL_data[L]
#     T_data[i] = 0.0 + T_data[L]
#     L += 1

#Find min, max for the plots
gmin = TL_data[0]
gmax = TL_data[0]
for i in range(N):
    if TL_data[i] < gmin:
        gmin = TL_data[i]
    if TL_data[i] > gmax:
        gmax = TL_data[i]

par_names = ['p_{}'.format(i + 1) for i in range(Npeak_pars*Npeaks + 3)]
par_vals = [peaks[i][0] for i in range (Npeak_pars*Npeaks + 3)]
err_names = ['err_{}'.format(p) for p in par_names]
err_vals = [peaks[i][1] for i in range (Npeak_pars*Npeaks + 3)]
limit_names = ['limit_{}'.format(p) for p in par_names]
limit_low = [peaks[i][2] for i in range (Npeak_pars*Npeaks + 3)]
limit_up = [peaks[i][3] for i in range (Npeak_pars*Npeaks + 3)]     
init_par = dict(zip(par_names, par_vals))
errors = dict(zip(par_names, err_vals))
limit_vals = list(zip(limit_low, limit_up))
limits = dict(zip(par_names, limit_vals))

#Define function to minimize
def minimize_func(*par):
    return sum(((TL_theo1(T, par) - TL)**2/TL) for T, TL in zip(T_data, TL_data))   

m = iminuit.Minuit(
    fcn=minimize_func,
    name=par_names,
    **init_par
    )
m.errors =  dict(zip(par_names, err_vals))
m.limits(limits)
m.simplex()
m.migrad()  
print(m.init_params)

#Store in variables fit values of parameters
par = np.ndarray(shape=1000, dtype=float)
for i in range(Npeaks * Npeak_pars + 3):
    par[i]=m.values[i]

#Evaluating Individual glow-peaks
P1 = np.zeros(N, dtype=float)
for i in range(0, N): P1[i]=TLD(T_data[i], par[0],par[1],par[2],par[3])
P2 = np.zeros(N, dtype=float)
if Npeaks>1:
    for i in range(0, N): P2[i]=TLD(T_data[i], par[4],par[5],par[6],par[7])
else:
    for i in range (0,N): P2[i]=0
P3 = np.zeros(N, dtype=float)
if Npeaks>2:
    for i in range(0, N): P3[i]=TLD(T_data[i], par[8],par[9],par[10],par[11])
else:
    for i in range (0,N): P3[i]=0
P4=np.zeros(N, dtype=float)
if Npeaks>3:
    for i in range(0, N): P4[i]=TLD(T_data[i], par[12],par[13],par[14],par[15])
else:
    for i in range (0,N): P4[i]=0    
P5=np.zeros(N, dtype=float)
if Npeaks>4:
    for i in range(0, N): P5[i]=TLD(T_data[i], par[16],par[17],par[18],par[19])
else:
    for i in range (0,N): P5[i]=0     
P6=np.zeros(N, dtype=float)
if Npeaks>5:
    for i in range(0, N): P6[i]=TLD(T_data[i], par[20],par[21],par[22],par[23])
else:
    for i in range (0,N): P6[i]=0 
P7=np.zeros(N, dtype=float)
if Npeaks>6:
    for i in range(0, N): P7[i]=TLD(T_data[i], par[24],par[25],par[26],par[27])
else:
    for i in range (0,N): P7[i]=0 
P8=np.zeros(N, dtype=float)
if Npeaks>7:
    for i in range(0, N): P8[i]=TLD(T_data[i], par[28],par[29],par[30],par[31])
else:
    for i in range (0,N): P8[i]=0 
P9=np.zeros(N, dtype=float)
if Npeaks>8:
    for i in range(0, N): P9[i]=TLD(T_data[i], par[32],par[33],par[34],par[35])
else:
    for i in range (0,N): P9[i]=0 
P10=np.zeros(N, dtype=float)
if Npeaks>9:
    for i in range(0, N): P10[i]=TLD(T_data[i], par[36],par[37],par[38],par[39])
else:
    for i in range (0,N): P10[i]=0 
        
PG = np.zeros(N, dtype=float)
for i in range(0, N):
    PG[i] = BGR(T_data[i], par[Npeak_pars * Npeaks + 0], par[Npeak_pars * Npeaks + 1], par[Npeak_pars * Npeaks + 2])

TLtheo = np.zeros(N, dtype=float)
for i in range(N):
    TLtheo[i] = P1[i]+P2[i]+P3[i]+P4[i]+P5[i]+P6[i]+P7[i]+P8[i]+P9[i]+P10[i]+PG[i]

OloExp = 0
OloTheo = 0
OloP1 = 0
OloP2 = 0
OloP3 = 0
OloP4 = 0
OloP5 = 0
OloP6 = 0
OloP7 = 0
OloP8 = 0
OloP9 = 0
OloP10 = 0
OloPg = 0
for i in range(N):
    OloExp+=TL_data[i]
    OloTheo += TLtheo[i]
    OloP1 += P1[i]
    OloP2 += P2[i]
    OloP3 += P3[i]
    OloP4 += P4[i]
    OloP5 += P5[i]
    OloP6 += P6[i]
    OloP7 += P7[i]
    OloP8 += P8[i]
    OloP9 += P9[i]
    OloP10 += P10[i]
    OloPg += PG[i]

#Evaluation of The Figure Of Merit, FOM value
i=0
for i in range(N):
    FOM=round(100*(np.sum(abs(TL_data[i]-TLtheo[i]))/(np.sum(TLtheo[i]))),3)
print("OloTheo = ", OloTheo)
print("FOM ={}% ".format(FOM))

#Plot experimental data
x = np.zeros(N, dtype=float)
for i in range(N): x[i] = T_data[i]
y = np.zeros(N, dtype=float)
for i in range(N): y[i] = TL_data[i]
plt.scatter(x, y, color='k', label="data")
#Plot fitted curve
xf = np.arange(T_data[0], T_data[N-1])
yf = TL_theo1(xf, par)
plt.plot(xf, yf, color='r', label="fit")

#Plot the individual peaks
x1 = np.zeros(N, dtype=float)
for i in range(N): x1[i] = T_data[i]
y1 = np.zeros(N, dtype=float)
for i in range(N): y1[i] = P1[i]
plt.plot(x1, y1, color='g', label='1st peak')

x2 = np.zeros(N, dtype=float)
for i in range(0,N): x2[i] = T_data[i]
y2 = np.zeros(N, dtype=float)
for i in range(N): y2[i] = P2[i]
plt.plot(x2, y2, color='c', label='2nd peak')

x3=np.zeros(N, dtype=float) 
for i in range(0,N): x3[i]=T_data[i]
y3=np.zeros(N, dtype=float) 
for i in range(0,N): y3[i]=P3[i]
plt.plot(x3, y3, color='m', label='3rd peak')

x4=np.zeros(N, dtype=float) 
for i in range(0,N): x4[i]=T_data[i]
y4=np.zeros(N, dtype=float) 
for i in range(0,N): y4[i]=P4[i]
plt.plot(x4, y4, color='y', label='4th peak')

x5=np.zeros(N, dtype=float) 
for i in range(0,N): x5[i]=T_data[i]
y5=np.zeros(N, dtype=float) 
for i in range(0,N): y5[i]=P5[i]
plt.plot(x5, y5, color='b', label='5th peak')

x6=np.zeros(N, dtype=float) 
for i in range(0,N): x6[i]=T_data[i]
y6=np.zeros(N, dtype=float) 
for i in range(0,N): y6[i]=P6[i]
plt.plot(x6, y6, color='tab:orange', label='6th peak')

x7=np.zeros(N, dtype=float) 
for i in range(0,N): x7[i]=T_data[i]
y7=np.zeros(N, dtype=float) 
for i in range(0,N): y7[i]=P7[i]
plt.plot(x7, y7, color='tab:purple', label='7th peak')

x8=np.zeros(N, dtype=float) 
for i in range(0,N): x8[i]=T_data[i]
y8=np.zeros(N, dtype=float) 
for i in range(0,N): y8[i]=P8[i]
plt.plot(x8, y8, color='tab:pink', label='8th peak')

xg = np.zeros(N, dtype=float)
for i in range(N): xg[i] = T_data[i]
yg = np.zeros(N, dtype=float)
for i in range(N): yg[i] = PG[i]
plt.plot(xg, yg, color='tab:olive', label='Background')

plt.title('TL Data')
plt.xlabel('T(K)')
plt.ylabel('TL signal (a.u.)')
plt.grid()
plt.legend()
plt.savefig('plot.png')
plt.show()

#Saving the results on disk
#Three types of results are saved
# A. The Glow-curve and its Individual TL peaks
# B. The Curve fitting parameters for all TL peaks
# C. A series of files are opened, each one corresponding to an individual TL peak 

print('Run file was: ', infile)
dirout="/python/TL/out/"
filep=input('Output data file name: ')
outname1 = str(dirout + filep+'.out')
outputFile = open(outname1, 'w')
for i in range(N):
    outputFile.write("{} {} {} {} {} {} {} {} {} \n" .format(T_data[i], TL_data[i], TL_theo1(T_data[i], par), P1[i], P2[i], P3[i], P4[i], P5[i], PG[i]))  
outputFile.close()
print("=========================================\n")

dirpar = "/python/TL/para/"
filepar = input('Output parameter file name: ')
outname2 = str(dirpar + filepar +'.par')
parputFile = open(outname2,'w')
parputFile.write( "Theo={}, FOM={} " .format(OloTheo, FOM))
parputFile.write("\n ************************************************************ \n")
parputFile.write("Integral, Imaximum, Tmax, E, b ")
parputFile.write("\n ************************************************************ \n")
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP1, par[0], par[1], par[2], par[3]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP2, par[4], par[5], par[6], par[7]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP3, par[8], par[9], par[10], par[11]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP4, par[12], par[13], par[14], par[15]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP5, par[16], par[17], par[18], par[19]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP6, par[20], par[21], par[22], par[23]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP7, par[24], par[25], par[26], par[27]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP8, par[28], par[29], par[30], par[31]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP9, par[32], par[33], par[34], par[35]))
parputFile.write("{}, {}, {}, {}, {} \n" .format(OloP10, par[36], par[37], par[38], par[39]))
parputFile.write("{}, {}, {}, {} \n" .format(OloPg, par[40], par[41], par[42]))
parputFile.close()

#Results of each TL Peak in separate data file 
print("Saving Behaviors of Individual Peaks \n")
apofasi = int(input("Input your decision. Yes=1, No=2 :  \n"))

if apofasi==1:
    dirpeaks = str("C:/Users/ekvla/Downloads/thesis/Experimental/minuit/iminuit/TLD/PARA/")
    fileindex = input("Output parameter file name: ")  
    
    outnamep1 = str(dirpeaks + fileindex + ".P1")
    P1outFile = open(outnamep1, 'w')
    P1outFile.write("FOM={}, OloP1={}, par[0]={}, par[1]={}, par[2]={}, par[3]={} " .format(FOM, OloP1, par[0], par[1], par[2], par[3]))
    P1outFile.close()
    
    outnamep2 = str(dirpeaks + fileindex + ".P2")
    P2outFile = open(outnamep2, 'w')
    P2outFile.write("FOM={}, OloP2={}, par[4]={}, par[5]={}, par[6]={}, par[7]={} " .format(FOM, OloP2, par[4], par[5], par[6], par[7]))
    P2outFile.close()

    outnamep3 = str(dirpeaks + fileindex + ".P3")
    P3outFile = open(outnamep3, 'w')
    P3outFile.write("FOM={}, OloP3={}, par[8]={}, par[9]={}, par[10]={}, par[11]={}" .format(FOM, OloP3, par[8], par[9], par[10], par[11]))
    P3outFile.close()

    outnamep4 = str(dirpeaks + fileindex + ".P4")
    P4outFile = open(outnamep4, 'w')
    P4outFile.write("FOM={}, OloP4={}, par[12]={}, par[13]={}, par[14]={}, par[15]={}" .format(FOM, OloP4, par[12], par[13], par[14], par[15]))
    P4outFile.close()

    outnamep5 = str(dirpeaks + fileindex + ".P5")
    P5outFile = open(outnamep5, 'w')
    P5outFile.write("FOM={}, OloP5={}, par[16]={}, par[17]={}, par[18]={}, par[19]={}" .format(FOM, OloP5, par[16], par[17], par[18], par[19]))
    P5outFile.close()

    outnamep6 = str(dirpeaks + fileindex + ".P6")
    P6outFile = open(outnamep6, 'w')
    P6outFile.write("FOM={}, OloP6={}, par[20]={}, par[21]={}, par[22]={}, par[23]={}" .format(FOM, OloP6, par[20], par[21], par[22], par[23]))
    P6outFile.close()

    outnamep7 = str(dirpeaks + fileindex + ".P7")
    P7outFile = open(outnamep7, 'w')
    P7outFile.write("FOM={}, OloP7={}, par[24]={}, par[25]={}, par[26]={}, par[27]={}" .format(FOM, OloP7, par[24], par[25], par[26], par[27]))
    P7outFile.close()

    outnamep8 = str(dirpeaks + fileindex + ".P8")
    P8outFile = open(outnamep8, 'w')
    P8outFile.write("FOM={}, OloP8={}, par[28]={}, par[29]={}, par[30]={}, par[31]={}" .format(FOM, OloP8, par[28], par[29], par[30], par[31]))
    P8outFile.close()

    outnamep9 = str(dirpeaks + fileindex + ".P9")
    P9outFile = open(outnamep9, 'w')
    P9outFile.write("FOM={}, OloP9={}, par[32]={}, par[33]={}, par[34]={}, par[35]={}" .format(FOM, OloP9, par[32], par[33], par[34], par[35]))
    P9outFile.close()

    outnamep10 = str(dirpeaks + fileindex + ".P10")
    P10outFile = open(outnamep10, 'w')
    P10outFile.write("FOM={}, OloP10={}, par[36]={}, par[37]={}, par[38]={}, par[39]={}" .format(FOM, OloP10, par[36], par[37], par[38], par[39]))
    P10outFile.close()

    print("DONE!!!")

if apofasi==2:
    print("DONE!!!")