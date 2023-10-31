import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
import iminuit

Npeaks = 5      #number of peaks
Npeak_pars = 3  #number of parameters per peak

# Theoretical function and background
def CW(T, Im, l, R):
    C=(1-R)/R
    ZZ=(1/C) - np.log(C) + ((l*T)/(1-R))
    W=0.0
    if np.any(ZZ<700.0):
        Z=ZZ
        W=np.real(lambertw(np.exp(Z)))
    else:
        W=ZZ- np.log(ZZ)
    return Im*l*(1.0/(W+W**2))

def BGR(T, AN, BM, xn):
    return AN + BM * np.power(T, xn)    

# Define theoretical function for total peak (function to fit)
def CW_theo1(T, *par):
    cw = 0.0
    # first sum up all the peak contributions
    for i in range(Npeaks):
        cw += CW(T, par[Npeak_pars * i + 0], par[Npeak_pars * i + 1], par[Npeak_pars * i + 2])

    # then add the background
    cw += BGR(T, par[Npeak_pars * Npeaks + 0], par[Npeak_pars * Npeaks + 1], par[Npeak_pars * Npeaks + 2])
    return cw

def CW_theo(t, *par):
    par_array = np.frombuffer(par,
                              dtype=float,
                              count=Npeak_pars * Npeaks + 3)
    return CW_theo1(t[0], *par_array)

# Read experimental file
dirdata = "/python/CWOSL/"
datafile = input("Enter experimental data file name : ")#Enter filename with .dat or .txt
infile = dirdata + datafile
print("Data file to read: ", infile)

channel, T_data, CW_data = np.loadtxt(infile,
                                    dtype=[('', 'i4'), ('', 'f8'),
                                        ('', 'f8')],
                                    usecols=(0, 1, 2),
                                    unpack=True)
N = len(channel)

#Print some data lines on screen 
for i in range(0,10):
    print("{} {} {}".format(channel[i], T_data[i], CW_data[i])) # print first 10 data lines
print("...")
for i in range(N - 10, N):
    print("{} {} {}".format(channel[i], T_data[i], CW_data[i])) # print last 10 data lines
print(f"Data points = ", N)

#Select the Low and Upper limits of the Curve Fitting
LL = input("Enter low limit LL: ")
LU = input("Enter upper limit LU: ")
L = int(LL)
N = int(LU) - int(LL) - 1

for i in range(N):
    CW_data[i] = CW_data[L]
    T_data[i] = 0.0 + T_data[L]
    L += 1


# Read parameters file
dirpara = "/python/CWOSL/"
parainfile = input("Parameters file name :  ") # Write parameters filename with .txt
parafile = dirpara + parainfile
print("The filename you entered was: ", parafile)

peaks = np.loadtxt(parafile,
                       comments='#',
                       usecols=(0, 1, 2, 3),
                       dtype=float)
for i in range(Npeaks*Npeak_pars + 3):
    print("Par line %d: %7.4f %7.4f %7.4f %7.4f" %(i+1, peaks[i][0], peaks[i][1], peaks[i][2], peaks[i][3]))

#Find min, max for the plots
gmin =CW_data[0]
gmax = CW_data[N-1]
for i in range(N):
    if CW_data[i] < gmin:
        gmin = CW_data[i]
    if CW_data[i] > gmax:
        gmax = CW_data[i]
        
par_names = ['p_{}'.format(i + 1) for i in range(Npeak_pars*Npeaks + 3)]
par_vals = [peaks[i][0] for i in range (Npeak_pars*Npeaks + 3)]
err_names = ["error_"+ p for p in par_names]
err_vals = [peaks[i][1] for i in range (Npeak_pars*Npeaks + 3)]
limit_names = ["limit_"+ p for p in par_names]
limit_low = [peaks[i][2] for i in range (Npeak_pars*Npeaks + 3)]
limit_up = [peaks[i][3] for i in range (Npeak_pars*Npeaks + 3)]
        
init_par = dict(zip(par_names, par_vals))

errors = dict(zip(err_names, err_vals))

limit_vals = list(zip(limit_low, limit_up))
limits = dict(zip(limit_names, limit_vals))

#Define function to minimize
def minimize_func(*par):
    return sum(((CW_theo1(T, *par) - TL)**2/TL) for T, TL in zip(T_data, CW_data))   

m = iminuit.Minuit(
    fcn=minimize_func,
    name=par_names,
    **init_par
    )

m.simplex(ncall=2000)  
m.migrad(ncall=2000)
print(m.init_params)

#Store in variables fit values of parameters
par = np.ndarray(shape=1000, dtype=float)
for i in range(Npeaks * Npeak_pars + 3):
    par[i]=m.values[i]


#Evaluating Individual glow-peaks
P1 = np.zeros(N, dtype=float)
for i in range(N):
    P1[i] = CW(T_data[i], par[0], par[1], par[2])
P2 = np.zeros(N, dtype=float)
if Npeaks > 1:
    for i in range(N):
        P2[i] = CW(T_data[i], par[3], par[4], par[5])
else:
    for i in range(N): P2[i] = 0
P3 = np.zeros(N, dtype=float)
if Npeaks > 2:
    for i in range(N):
        P3[i] = CW(T_data[i], par[6], par[7], par[8])
else:
    for i in range(N): P3[i] = 0
P4 = np.zeros(N, dtype=float)
if Npeaks > 3:
    for i in range(N):
        P4[i] = CW(T_data[i], par[9], par[10], par[11])
else:
    for i in range(N): P4[i] = 0
P5 = np.zeros(N, dtype=float)
if Npeaks > 4:
    for i in range(N):
        P5[i] = CW(T_data[i], par[12], par[13], par[14])
else:
    for i in range(N): P5[i] = 0
PG = np.zeros(N, dtype=float)
for i in range(0, N):
    PG[i] = BGR(T_data[i], par[Npeak_pars * Npeaks + 0], par[Npeak_pars * Npeaks + 1], par[Npeak_pars * Npeaks + 2])

CWtheo = np.zeros(N, dtype=float)
for i in range(N):
    CWtheo[i] = P1[i]+P2[i]+P3[i]+P4[i]+P5[i]+PG[i]

OloTheo = 0
OloP1 = 0
OloP2 = 0
OloP3 = 0
OloP4 = 0
OloP5 = 0
OloPg = 0
for i in range(N):
    OloTheo += CWtheo[i]
    OloP1 += P1[i]
    OloP2 += P2[i]
    OloP3 += P3[i]
    OloP4 += P4[i]
    OloP5 += P5[i]
    OloPg += PG[i]


#Evaluation of The Figure Of Merit, FOM value
i=0
for i in range(N):
    FOM=round(100*(np.sum(abs(CW_data[i]-CWtheo[i]))/(np.sum(CWtheo[i]))),3)
print("OloTheo = ", OloTheo)
print("FOM ={}% ".format(FOM))

x = np.zeros(N, dtype=float)
for i in range(N): x[i] = T_data[i]
y = np.zeros(N, dtype=float)
for i in range(N): y[i] = CW_data[i]
plt.scatter(x, y, color='k', label="data")

xf = np.linspace(T_data[0], T_data[N - 1], Npeaks*Npeak_pars + 3)
yf = CW_theo1(xf, *par)
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
plt.plot(x4, y4, color='tab:orange', label='4th peak')

x5=np.zeros(N, dtype=float) 
for i in range(0,N): x5[i]=T_data[i]
y5=np.zeros(N, dtype=float) 
for i in range(0,N): y5[i]=P5[i]
plt.plot(x5, y5, color='b', label='5th peak')

xg = np.zeros(N, dtype=float)
for i in range(N): xg[i] = T_data[i]
yg = np.zeros(N, dtype=float)
for i in range(N): yg[i] = PG[i]
plt.plot(xg, yg, color='tab:olive', label='Background')

plt.title('CW Data')
plt.xlabel('T (s)')
plt.ylabel('CW OSL Intensity (a.u.)')
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

#Saving the results on disk
#Three types of results are saved
# A. The Glow-curve and its Individual TL peaks
# B. The Curve fitting parameters for all TL peaks
# C. A series of files are opened, each one corresponding to an individual TL peak

print('Run file was: ', infile)
dirout="/python/CWOSL/out/"
filep=input('Output data file name: ')
outname1 = str(dirout + filep+'.out')
outputFile = open(outname1, 'w')
for i in range(N):
    outputFile.write("{} {} {} {} {} {} {} {} {} \n" .format(round(T_data[i],3), round(CW_data[i],3), round(CW_theo1(T_data[i], *par),3), round(P1[i],3), round(P2[i],3), round(P3[i],3), round(P4[i],3), round(P5[i],3), round(PG[i]),3))
outputFile.close()
print("=========================================\n")

dirpar = "/python/CWOSL/para/"
filepar = input('Output parameter file name: ')
outname2 = str(dirpar + filepar +'.par')
parputFile = open(outname2,'w')
parputFile.write( "Theo={}, FOM={} " .format(round(OloTheo,3), FOM))
parputFile.write("\n ************************************************************ \n")
parputFile.write("Integral    Imaximum    l    R")
parputFile.write("\n ************************************************************ \n")
parputFile.write("{}, {}, {}, {} \n" .format(round(OloP1,3), round(par[0],3), round(par[1],3), round(par[2],3)))
parputFile.write("{}, {}, {}, {} \n" .format(round(OloP2,3), round(par[3],3), round(par[4],3), round(par[5],3)))
parputFile.write("{}, {}, {}, {} \n" .format(round(OloP3,3), round(par[6],3), round(par[7],3), round(par[8],3)))
parputFile.write("{}, {}, {}, {} \n" .format(round(OloP4,3), round(par[9],3), round(par[10],3), round(par[11],3)))
parputFile.write("{}, {}, {}, {} \n" .format(round(OloP5,3), round(par[12],3), round(par[13],3), round(par[14],3)))
parputFile.write("{}, {}, {}, {} \n" .format(round(OloPg,3), round(par[15],3), round(par[16],3), round(par[17],3)))
parputFile.close()

#Results of each TL Peak in separate data file 
print("Saving Behaviors of Individual Peaks \n")
apofasi = int(input("Input your decision. Yes=1, No=2 :  \n"))

if apofasi==1:
    dirpeaks = "/python/CWOSL/para/"
    fileindex = input("Output parameter file name: ")  
    
    outnamep1 = str(dirpeaks + fileindex + ".P1")
    P1outFile = open(outnamep1, 'w')
    P1outFile.write("FOM={}, OloP1={}, par[0]={}, par[1]={}, par[2]={} " .format(FOM, OloP1, par[0], par[1], par[2]))
    P1outFile.close()
    
    outnamep2 = str(dirpeaks + fileindex + ".P2")
    P2outFile = open(outnamep2, 'w')
    P2outFile.write("FOM={}, OloP2={}, par[3]={}, par[4]={}, par[5]={} " .format(FOM, OloP2, par[3], par[4], par[5]))
    P2outFile.close()

    outnamep3 = str(dirpeaks + fileindex + ".P3")
    P3outFile = open(outnamep3, 'w')
    P3outFile.write("FOM={}, OloP3={}, par[6]={}, par[7]={}, par[8]={} " .format(FOM, OloP3, par[6], par[7], par[8]))
    P3outFile.close()

    outnamep4 = str(dirpeaks + fileindex + ".P4")
    P4outFile = open(outnamep4, 'w')
    P4outFile.write("FOM={}, OloP4={}, par[9]={}, par[10]={}, par[11]={}" .format(FOM, OloP4, par[9], par[10], par[11]))
    P4outFile.close()

    outnamep5 = str(dirpeaks + fileindex + ".P5")
    P5outFile = open(outnamep5, 'w')
    P5outFile.write("FOM={}, OloP5={}, par[12]={}, par[13]={}, par[14]={}" .format(FOM, OloP5, par[12], par[13], par[14]))
    P5outFile.close()

    print("DONE!!!")

if apofasi==2:
    print("DONE!!!")