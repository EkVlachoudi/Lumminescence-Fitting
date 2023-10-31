import ctypes
from array import array as arr
from gettext import dgettext
import scipy.special as sc
import numpy as np
import matplotlib.pyplot as plt
import csv
from ROOT import TMinuit, TGraph, TCanvas, TF1, TMultiGraph, TLegend

#Function
def fourPL(I, b, Dc, a, D):
     return np.float(I*(1-(np.real(sc.lambertw(b* np.exp (b) * np.exp (-D/Dc)))/b)**a))

Np=1
Npars=4
N=0

def TL_theo1(D, par):
    TL = 0.0
    for i in range(Np):
        TL += fourPL(par[Npars * i + 0], par[Npars * i + 1], par[Npars * i + 2], par[Npars * i + 3], D)
    return TL

def TL_theo(d, par):
    par_array = np.frombuffer(par, dtype=ctypes.c_double, count=Np * Npars)
    return TL_theo1(d[0], par_array)

class fit_func():
    def __init__(self, N, Ds, TLs):
        self.N = N
        self.D = Ds
        self.TL_exp = TLs

    def __call__(self, npar, deriv, f, par, iflag):
        # npar=4
        f.value = 0.0
        pp = np.frombuffer(par,
                           dtype=ctypes.c_double,
                           count=Np * Npars)
        for i in range(self.N):
            d = self.TL_exp[i] - TL_theo1(self.D[i], pp)
            f.value = f.value + d * d / self.TL_exp[i]


def TTOR(print_all_data = False):
    # Read parameters file
    dirpara = "/pyroot/TTOR/"
    parainfile = input("Parameters file to read :  ") # Write parameters filename with .txt
    parafile = dirpara + parainfile
    print("The filename you entered was: ", parafile)

    peaks = np.loadtxt(parafile,
                       comments='#',
                       usecols=(0, 1, 2, 3),
                       dtype=float)
    for i in range(Np*Npars):
        print("Par line %d: %7.4f %7.4f %7.4f %7.4f" %(i+1, peaks[i][0], peaks[i][1], peaks[i][2], peaks[i][3]))
    
    # Read experimental file
    dirdata = "/pyroot/TTOR/"
    datafile = input("Enter experimental data file name : ")#Enter filename with .dat or .txt
    infile = dirdata + datafile
    print("Data file to read: ", infile)

    Dx, TL_exp = np.loadtxt(infile, dtype=[('', 'f8'), ('', 'f8')],usecols=(0, 1), unpack=True)
    N=len(Dx)

    #Create canvas
    cv = TCanvas("cv_TTOR", "cv_TTOR")
    cv.SetFillColor(0)
    cv.SetLogx()
    cv.SetLogy()

    #Initialize TMinuit for the fit
    ierflg = ctypes.c_int()
    arglist = arr('d', 2*[0.1]) # ser error definition
    
    mini = TMinuit(Np * Npars)
    fcn = fit_func(N, Dx, TL_exp)
    mini.SetFCN(fcn)  # set function to minimize
    arglist[0] = 1
    mini.mnexcm("SET ERR", arglist, 1, ierflg)

    #Set starting values and step sizes for parameters
    #1st peak
    ipar = 0
    mini.mnparm(ipar, "I", peaks[0][0], peaks[0][1], peaks[0][2],peaks[0][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "b", peaks[1][0], peaks[1][1], peaks[1][2], peaks[1][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Dc", peaks[2][0], peaks[2][1], peaks[2][2], peaks[2][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "a", peaks[3][0], peaks[3][1], peaks[3][2], peaks[3][3], ierflg)
    
    #Minimization step
    arglist[0] = 2000  # max number of calls to fcn before giving up
    arglist[1] = 1e-20  # tolerance

    mini.mnexcm("SIMPLEX", arglist, 2, ierflg)
    mini.mnexcm("MIGRAD", arglist, 2, ierflg)  # execute the minimisation

    #Get result from Minuit
    par = np.ndarray(shape=1000, dtype=float)
    epar = np.ndarray(shape=1000, dtype=float)
    for i in range(Np * Npars):
        temp_par = ctypes.c_double()
        temp_epar = ctypes.c_double()
        # retrieve parameters and errors
        mini.GetParameter(int(i), temp_par, temp_epar)
        par[i] = temp_par.value
        epar[i] = temp_epar.value

    #Evaluating Individual peaks
    P1 = np.zeros(10000, dtype=float)
    for i in range(N):
        P1[i] = fourPL(par[0], par[1], par[2], par[3], Dx[i])
        
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
        FOM=round(100*(np.sum(abs(TL_exp[i]-TLtheo[i]))/(np.sum(TLtheo[i]))),3)
    print("OloTheo = ", OloTheo)
    print("FOM ={}% ".format(FOM))

    #ROOT Graphs
    gx = np.zeros(10000)
    gy = np.zeros(10000)
    for i in range(N):
        gx[i] = Dx[i]
        gy[i] = TL_exp[i]

    g = TGraph(N, gx, gy)
    g.SetName("g")
    g.SetMarkerColor(1)
    g.SetMarkerStyle(21)
    g.SetMarkerSize(0.7)
    g.SetTitle(";DOSE(Gy) ;TL(a.u.)")
    g.GetYaxis().SetTitleOffset(1.25)

    #Plot the fitted
    fitted = TF1("fitted", TL_theo, Dx[0], Dx[N - 1], Np * Npars)
    fitted.SetLineColor(2)
    fitted.SetLineWidth(2)
    fitted.SetParameters(par)

    gmulti = TMultiGraph()
    gmulti.SetTitle(";DOSE(Gy);TL(A.U.)")
    gmulti.Add(g, "p")
    gmulti.GetListOfFunctions().Add(fitted)
    gmulti.Draw("a")

    leg = TLegend(0.8, 0.7, 0.9, 0.8)
    leg.AddEntry(g, "g", "p")
    leg.AddEntry(fitted, "fitted", "l")
    leg.Draw()
   
    #Saving the results on disk
    dirout=str('/pyroot/TTOR/')
    filed=input('Output data file name: ')
    outname1 = str(dirout + filed+'.txt')
    outputFile = open(outname1, 'w')
    outputFile.write("DOSE (Gy), Iexp(A.U), Itheo (A.U)\n")
    for i in range(N):
        outputFile.write("{}     {}     {} \n" .format(Dx[i], TL_exp[i], TL_theo1(Dx[i],par)))  
    outputFile.close()
    print("=========================================\n")
    
    filep=input('Output parameter file name: ')
    outname2 = str(dirout + filep + '.txt')
    parameters=['I:', 'b:', 'Dc:', 'a:']
    with open( outname2, 'w') as para:
        i=0
        for i in range(4):
             para.write("{} {}\n".format(parameters[i], par[i]))
             i+=1
    print("DONE!!!")

    cv.Draw()
    input('Press Enter to close')
    
if __name__ == "__main__":
    TTOR()
