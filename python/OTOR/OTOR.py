import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
import iminuit

Np = 1      #number of peaks
Npars = 3  #number of parameters per peak

#Function
def fourPL(D, R, Dc, Io):
	return (Io)*(1+((np.real(lambertw((R-1)*np.exp(R-1-(D/(Dc))))))/(1-R)))

def Theo1(D, *par):
    TL = 0.0
    for i in range(Np):
        TL += fourPL(D, par[Npars * i + 0], par[Npars * i + 1], par[Npars * i + 2])
    return TL

def Theo(d, *par):
    par_array = np.frombuffer(par, dtype=float, count=Np * Npars)
    return Theo1(d[0], *par_array)

# Read experimental file
dirdata = "/python/OTOR/"
datafile = input("Enter experimental data file name : ")#Enter filename with .dat or .txt
infile = dirdata + datafile
print("Data file to read: ", infile)

Dx, TL_data = np.loadtxt(infile, dtype=[('', 'f8'), ('', 'f8')],usecols=(0, 1), unpack=True)
N=len(Dx)
#Print some data lines on screen 
for i in range(0,10):
    print("{} {}".format(Dx[i], TL_data[i])) # print first 10 data lines
print("...")
for i in range(N - 10, N):
    print("{} {}".format(Dx[i], TL_data[i])) # print last 10 data lines
print(f"Data points = ", N)

# Read parameters file
dirpara = "/python/OTOR/"
parainfile = input("Parameters file to read :  ") # Write parameters filename with .txt
parafile = dirpara + parainfile
print("The filename you entered was: ", parafile)

peaks = np.loadtxt(parafile,
                    comments='#',
                    usecols=(0, 1, 2, 3),
                    dtype=float)
for i in range(Np*Npars):
    print("Par line %d: %7.4f %7.4f %7.4f %7.4f" %(i+1, peaks[i][0], peaks[i][1], peaks[i][2], peaks[i][3]))

par_names = ['p_{}'.format(i + 1) for i in range(Np * Npars)]
par_vals = [peaks[i][0] for i in range (Np * Npars)]
err_names = ["error_"+ p for p in par_names]
err_vals = [peaks[i][1] for i in range (Np * Npars)]
limit_names = ["limit_"+ p for p in par_names]
limit_low = [peaks[i][2] for i in range (Np * Npars)]
limit_up = [peaks[i][3] for i in range (Np * Npars)]
        
init_par = dict(zip(par_names, par_vals))
print(init_par)
errors = dict(zip(err_names, err_vals))

limit_vals = list(zip(limit_low, limit_up))
limits = dict(zip(limit_names, limit_vals))

#Define function to minimize
def minimize_func(*par):
    return sum(((Theo1(D, *par) - TL)**2/TL) for D, TL in zip(Dx, TL_data))   

m = iminuit.Minuit(
    fcn=minimize_func,
    name=par_names,
    **init_par,
    )
m.simplex()
m.migrad()  
print(m.init_params)

#Store in variables fit values of parameters
par = np.ndarray(shape=1000, dtype=float)
for i in range(Np * Npars):
    par[i]=m.values[i]

#Evaluating Individual peaks
P1 = np.zeros(10000, dtype=float)
for i in range(N):
    P1[i] = fourPL(Dx[i], par[0], par[1], par[2])
    
TLtheo = np.zeros(10000, dtype=float)
for i in range(N):
    TLtheo[i] = P1[i] 
    
OloTheo = 0
OloP1 = 0
for i in range(N):
    OloTheo += TLtheo[i]
    OloP1 += P1[i]

#FOM value
i=0
for i in range(N):
    FOM=round(100*(np.sum(abs(TL_data[i]-TLtheo[i]))/(np.sum(TLtheo[i]))),3)
print("OloTheo = ", OloTheo)
print("FOM ={}% ".format(FOM))

#Plot experimental data
x = np.zeros(N, dtype=float)
for i in range(N): x[i] = Dx[i]
y = np.zeros(N, dtype=float)
for i in range(N): y[i] = TL_data[i]
plt.scatter(x, y, color='k', label="data")

# #Plot fitting curve
x_min, x_max = np.amin(Dx), np.amax(Dx)
xf = np.linspace(x_min, x_max, 10000)
yf = Theo1(xf, *par)
plt.plot(xf, yf, color='r', label="fit")
#Plot the individual peaks
x1 = np.zeros(N, dtype=float)
for i in range(N): x1[i] = Dx[i]
y1 = np.zeros(N, dtype=float)
for i in range(N): y1[i] = P1[i]
plt.plot(x1, y1, color='g', label='1st peak')

plt.xscale('log')
plt.yscale('log')
plt.title ('DOSE RESPONSE')
plt.xlabel('DOSE (Gy)')
plt.ylabel('TL(A.U)')
plt.legend()
plt.savefig('plot.png', bbox_inches='tight')
plt.show()

#Saving the results on disk
dirout="/python/OTOR/"
filed=input('Output data file name: ')
outname1 = str(dirout + filed+'.txt')
outputFile = open(outname1, 'w')
outputFile.write("DOSE (Gy), Iexp(A.U), Itheo (A.U)\n")
for i in range(N):
    outputFile.write("{}     {}     {}\n" .format(Dx[i], TL_data[i], Theo1(Dx[i],*par)))  
outputFile.close()
print("=========================================\n")

filep=input('Output parameter file name: ')
outname2 = str(dirout + filep + '.txt')
parameters=['R:', 'Dc:', 'Io:']
with open( outname2, 'w') as para:
    i=0
    for i in range(3):
            para.write("{} {}\n".format(parameters[i], par[i]))
            i+=1
print("DONE!!!")