
import ctypes
from array import array as arr
from scipy.special import lambertw
import numpy as np
from ROOT import TGraph, TCanvas, TF1, TMultiGraph, TLegend
from ROOT import TMinuit

# Theoretical function and background
def LM(Im, Tm, R, T):
    c=(1-R)/R
    zz=(1/c)-np.log(c)+((T**2)/((Tm**2)*(1-R)*(1+0.534156*(R**0.7917))))
    z_m=(1/c)-np.log(c)+(1/((1-R)*(1+0.534156*(R**0.7917))))
    w_m=np.real(lambertw(np.exp(z_m)))
    w=0.0
    if np.any(zz<700.00):
        z=zz
        w=np.real(lambertw(np.exp(z)))
    else:
        w=zz-np.log(zz)
    return T*(Im/Tm)*((w_m+w_m**2)/(w+w**2))

def BGR(AN, BM, xn, T):
    return AN + BM * np.power(T, xn)

Npeaks = 5  # number of peaks
Npeak_pars = 3  # number of parameters per peak
N = 0  # number of data point in glow curve that is read

# Define theoretical function for total peak (function to fit)
def LM_theo1(T, par):
    lm = 0.0
    # first sum up all the peak contributions
    for i in range(Npeaks):
        lm += LM(par[Npeak_pars * i + 0], par[Npeak_pars * i + 1],
                   par[Npeak_pars * i + 2], T)

    # then add the background
    lm += BGR(par[Npeak_pars * Npeaks + 0], par[Npeak_pars * Npeaks + 1],
              par[Npeak_pars * Npeaks + 2], T)
    return lm

def LM_theo(t, par):
    par_array = np.frombuffer(par,
                              dtype=ctypes.c_double,
                              count=Npeak_pars * Npeaks + 3)
    return LM_theo1(t[0], par_array)

# With this method, a fit_func object can be used as a function
# The function __call__ is called like fcn by Minuit repeatedly with varying parameters
class fit_func():
    def __init__(self, N, Ts, TLs):
        self.N = N
        self.T = Ts
        self.LM_exp = TLs

    def __call__(self, npar, deriv, f, par, iflag):
        # npar=3
        f.value = 0.0
        pp = np.frombuffer(par,
                           dtype=ctypes.c_double,
                           count=Npeak_pars * Npeaks + 3)
        for i in range(self.N):
            d = self.LM_exp[i] - LM_theo1(self.T[i], pp)
            f.value = f.value + d * d / self.LM_exp[i]


def LMOSL(print_all_data = False):
    # Read parameters file
    dirpara = "/pyroot/LMOSL/"
    parainfile = input("Parameters file to read :  ") # Write parameters filename with .txt
    parafile = dirpara + parainfile
    print("The filename you entered was: ", parafile)

    peaks = np.loadtxt(parafile,
                       comments='#',
                       usecols=(0, 1, 2, 3),
                       dtype=float)
    for i in range(Npeaks*Npeak_pars+3):
        print("Par line %d: %7.4f %7.4f %7.4f %7.4f" %(i+1, peaks[i][0], peaks[i][1], peaks[i][2], peaks[i][3]))

    # Read experimental file
    dirdata = "/pyroot/LMOSL/"
    datafile = input("Enter experimental data file name : ")#Enter filename with .dat
    infile = dirdata + datafile
    print("Data file to read: ", infile)

    channel, Tx, LM_exp = np.loadtxt(infile,
                                     dtype=[('', 'i4'), ('', 'f8'),
                                            ('', 'f8')],
                                     usecols=(0, 1, 2),
                                     unpack=True)
    N = len(channel)
    #Print some data lines on screen
    if print_all_data==0:
        for i in range(0,10):
            print("{} {} {}".format(channel[i], Tx[i], LM_exp[i])) # print first 10 data lines
        print("...\n")
        for i in range(10, 0, -1):
            print("{} {} {}".format(channel[i], Tx[i], LM_exp[i])) # print last 10 data lines
    else:
        for i in range(0, N):
            print("{} {} {}".format(channel[i], Tx[i], LM_exp[i])) # print all data lines   
    print("Data point = {}".format(N))
    
    #Select the Low and Upper limits of the Curve Fitting
    LL = input("Enter low limit LL: ")
    LU = input("Enter upper limit LU: ")
    L = int(LL)
    N = int(LU) - int(LL) - 1

    for i in range(N):
        LM_exp[i] = LM_exp[L]
        Tx[i] = 0.0 + Tx[L]
        L += 1

    precond = input("Enter the preconditioning: ")

    #Find min, max for the plots
    gmin = LM_exp[0]
    gmax = LM_exp[0]
    for i in range(N):
        if LM_exp[i] < gmin:
            gmin = LM_exp[i]
        if LM_exp[i] > gmax:
            gmax = LM_exp[i]

    #Create canvas
    cv = TCanvas("cv_LM_fit", "cv_LM_fit")
    cv.SetFillColor(0)
    cv.SetLogx()

    #Initialize TMinuit with maximum of :number of parameters*number of peaks+2
    ierflg = ctypes.c_int()
    arglist = arr('d', 2*[0.1]) # ser error definition
    
    mini = TMinuit(Npeaks * Npeak_pars + 2)
    fcn = fit_func(N, Tx, LM_exp)
    mini.SetFCN(fcn)  # set function to minimize
    arglist[0] = 1
    mini.mnexcm("SET ERR", arglist, 1, ierflg)

    #Set starting values and step sizes for parameters
    #1st peak
    ipar = 0
    mini.mnparm(ipar, "Im1", peaks[0][0], peaks[0][1], peaks[0][2], peaks[0][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm1", peaks[1][0], peaks[1][1], peaks[1][2], peaks[1][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "R1", peaks[2][0], peaks[2][1], peaks[2][2], peaks[2][3], ierflg)
    ipar += 1
    #2nd peak
    mini.mnparm(ipar, "Im2", peaks[3][0], peaks[3][1], peaks[3][2], peaks[3][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm2", peaks[4][0], peaks[4][1], peaks[4][2], peaks[4][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "R2", peaks[5][0], peaks[5][1], peaks[5][2], peaks[5][3], ierflg)
    ipar += 1
    #3rd peak
    mini.mnparm(ipar, "Im3", peaks[6][0], peaks[6][1], peaks[6][2], peaks[6][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "l3", peaks[7][0], peaks[7][1], peaks[7][2], peaks[7][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "R3", peaks[8][0], peaks[8][1], peaks[8][2], peaks[8][3], ierflg)
    ipar += 1
    #4th peak
    mini.mnparm(ipar, "Im4", peaks[9][0], peaks[9][1], peaks[9][2], peaks[9][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "l4", peaks[10][0], peaks[10][1], peaks[10][2], peaks[10][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "R4", peaks[11][0], peaks[11][1], peaks[11][2], peaks[11][3], ierflg)
    ipar += 1
    #5th peak
    mini.mnparm(ipar, "Im5", peaks[12][0], peaks[12][1], peaks[12][2], peaks[12][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "l5", peaks[13][0], peaks[13][1], peaks[13][2], peaks[13][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "R5", peaks[14][0], peaks[14][1], peaks[14][2], peaks[14][3], ierflg)
    ipar += 1
    #Background peak
    mini.mnparm(ipar, "Img", peaks[15][0], peaks[15][1], peaks[15][2], peaks[15][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "lg", peaks[16][0], peaks[16][1], peaks[16][2], peaks[16][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Rg", peaks[17][0], peaks[17][1], peaks[17][2], peaks[17][3], ierflg)

    #Minimization step
    arglist[0] = 2000  # max number of calls to fcn before giving up
    arglist[1] = 1e-20  # tolerance

    mini.mnexcm("SIMPLEX", arglist, 2, ierflg)
    mini.mnexcm("MIGRAD", arglist, 2, ierflg)  # execute the minimisation

    #Get result from Minuit
    par = np.ndarray(shape=1000, dtype=float)
    epar = np.ndarray(shape=1000, dtype=float)
    for i in range(Npeaks * Npeak_pars + 2):
        temp_par = ctypes.c_double()
        temp_epar = ctypes.c_double()
        # retrieve parameters and errors
        mini.GetParameter(int(i), temp_par, temp_epar)
        par[i] = temp_par.value
        epar[i] = temp_epar.value

    #Evaluating Individual glow-peaks
    P1 = np.zeros(10000, dtype=float)
    for i in range(N):
        P1[i] = LM(par[0], par[1], par[2], Tx[i])
    P2 = np.zeros(10000, dtype=float)
    if Npeaks > 1:
        for i in range(N):
            P2[i] = LM(par[3], par[4], par[5], Tx[i])
    else:
        for i in range(N):
            P2[i] = 0
    P3 = np.zeros(10000, dtype=float)
    if Npeaks > 2:
        for i in range(N):
            P3[i] = LM(par[6], par[7], par[8], Tx[i])
    else:
        for i in range(N):
            P3[i] = 0
    P4 = np.zeros(10000, dtype=float)
    if Npeaks > 3:
        for i in range(N):
            P4[i] = LM(par[9], par[10], par[11], Tx[i])
    else:
        for i in range(N): P4[i] = 0
    P5 = np.zeros(10000, dtype=float)
    if Npeaks > 4:
        for i in range(N):
            P5[i] = LM(par[12], par[13], par[14], Tx[i])
    else:
        for i in range(N): P5[i] = 0
    PG = np.zeros(10000, dtype=float)
    for i in range(0, N):
        PG[i] = BGR(par[Npeak_pars * Npeaks + 0], par[Npeak_pars * Npeaks + 1], par[Npeak_pars * Npeaks + 2], Tx[i])
    
    LMtheo = np.zeros(10000, dtype=float)
    for i in range(N):
        LMtheo[i] = P1[i]+P2[i]+P3[i]+P4[i]+P5[i]+PG[i]

    OloTheo = 0
    OloP1 = 0
    OloP2 = 0
    OloP3 = 0
    OloP4 = 0
    OloP5 = 0
    OloPg = 0
    for i in range(N):
        OloTheo += LMtheo[i]
        OloP1 += P1[i]
        OloP2 += P2[i]
        OloP3 += P3[i]
        OloP4 += P4[i]
        OloP5 += P5[i]
        OloPg += PG[i]

    #Evaluation of The Figure Of Merit, FOM value
    #DYS=0.0
    i=0
    for i in range(N):
        FOM=round(100*(np.sum(abs(LM_exp[i]-LMtheo[i]))/(np.sum(LMtheo[i]))),3)
    print("OloTheo = ", OloTheo)
    print("FOM ={}% ".format(FOM))

    #Make Graphs
    gx = np.zeros(10000)
    gy = np.zeros(10000)
    for i in range(N):
        gx[i] = Tx[i]
        gy[i] = LM_exp[i]

    g = TGraph(N, gx, gy)
    g.SetName("g")
    g.SetMarkerColor(1)
    g.SetMarkerStyle(21)
    g.SetMarkerSize(0.7)
    g.SetTitle(";T (s);LM OSL Intensity (a.u.)")
    g.GetYaxis().SetTitleOffset(1.25)

    #Plot the fitted
    fitted = TF1("fitted", LM_theo, Tx[0], Tx[N - 1], Npeaks * Npeak_pars + 2)
    fitted.SetLineColor(2)
    fitted.SetLineWidth(2)
    fitted.SetParameters(par)

    x1 = np.zeros(10000, dtype=float)
    y1 = np.zeros(10000, dtype=float)
    for i in range(N):
        x1[i] = Tx[i]
        y1[i] = P1[i]
    g1 = TGraph(N, x1, y1)
    g1.SetNameTitle("g1", "g1")
    g1.SetLineColor(3)
    g1.SetLineWidth(2)

    x2 = np.zeros(10000, dtype=float)
    y2 = np.zeros(10000, dtype=float)
    for i in range(N):
        x2[i] = Tx[i]
        y2[i] = P2[i]
    g2 = TGraph(N, x2, y2)
    g2.SetNameTitle("g2", "g2")
    g2.SetLineColor(4)
    g2.SetLineWidth(2)

    x3 = np.zeros(10000, dtype=float)
    y3 = np.zeros(10000, dtype=float)
    for i in range(N):
        x3[i] = Tx[i]
        y3[i] = P3[i]
    g3 = TGraph(N, x3, y3)
    g3.SetNameTitle("g3", "g3")
    g3.SetLineColor(5)
    g3.SetLineWidth(2)

    x4=np.zeros(10000, dtype=float) 
    for i in range(0,N): x4[i]=Tx[i]
    y4=np.zeros(10000, dtype=float) 
    for i in range(0,N): y4[i]=P4[i]
    g4 = TGraph(N,x4,y4)
    g4.SetNameTitle("g4", "g4")
    g4.SetLineColor(6)
    g4.SetLineWidth(2)

    x5=np.zeros(10000, dtype=float) 
    for i in range(0,N): x5[i]=Tx[i]
    y5=np.zeros(10000, dtype=float) 
    for i in range(0,N): y5[i]=P5[i]
    g5 = TGraph(N,x5,y5)
    g5.SetNameTitle("g5", "g5")
    g5.SetLineColor(7)
    g5.SetLineWidth(2)

    xg = np.zeros(10000, dtype=float)
    yg = np.zeros(10000, dtype=float)
    for i in range(N):
        xg[i] = Tx[i]
        yg[i] = PG[i]
    gg = TGraph(N, xg, yg)
    gg.SetNameTitle("gg", "gg")
    gg.SetLineColor(8)
    gg.SetLineWidth(2)

    gmulti = TMultiGraph()
    gmulti.SetTitle(";T (s);LM OSL Intensity (a.u.)")
    gmulti.Add(g, "p")
    gmulti.Add(g1, "l")
    gmulti.Add(g2, "l")
    gmulti.Add(g3, "l")
    gmulti.Add(g4, "l")
    gmulti.Add(g5, "l")
    gmulti.Add(gg, "l")
    gmulti.GetListOfFunctions().Add(fitted)
    gmulti.Draw("a")

    leg = TLegend(0.8, 0.7, 0.9, 0.8)
    leg.AddEntry(g, "g", "p")
    leg.AddEntry(g1, "g1", "l")
    leg.AddEntry(g2, "g2", "l")
    leg.AddEntry(g3, "g3", "l")
    leg.AddEntry(g4, "g4", "l")
    leg.AddEntry(g5, "g5", "l")
    leg.AddEntry(gg, "gg", "l")
    leg.AddEntry(fitted, "fitted", "l")
    leg.Draw()


    #Saving the results on disk
    #Three types of results are saved
    # A. The Glow-curve and its Individual TL peaks
    # B. The Curve fitting parameters for all TL peaks
    # C. A series of files are opened, each one corresponding to an individual TL peak 

    print('Run file was: ', infile)
    dirout=str('/pyroot/LMOSL/out/' )
    filep=input('Output data file name: ')
    outname1 = str(dirout + filep+'.out')
    outputFile = open(outname1, 'w')
    for i in range(N):
        outputFile.write("{} {} {} {} {} {} {} {} {}\n" .format(Tx[i], LM_exp[i], LM_theo1(Tx[i],par), P1[i], P2[i], P3[i], P4[i], P5[i], PG[i]))  
    outputFile.close()
    print("=========================================\n")

    dirpar = str('/pyroot/LMOSL/para/')
    filepar = input('Output parameter file name: ')
    outname2 = str(dirpar + filepar +'.par')
    parputFile = open(outname2,'w')
    parputFile.write( "Theo={}, FOM={}, LL={}, LU={} " .format(OloTheo, FOM, LL, LU))
    parputFile.write("\n ************************************************************ \n")
    parputFile.write("Integral   Imaximum    Tmax    E     R")
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
        dirpeaks = str("/pyroot/LMOSL/para/")
        fileindex = input("Output parameter file name: ")  
        
        outnamep1 = str(dirpeaks + fileindex + ".P1")
        P1outFile = open(outnamep1, 'w')
        P1outFile.write("FOM={}, precond={}, OloP1={}, par[0]={}, par[1]={}, par[2]={} " .format(FOM, precond, OloP1, par[0], par[1], par[2]))
        P1outFile.close()
        
        outnamep2 = str(dirpeaks + fileindex + ".P2")
        P2outFile = open(outnamep2, 'w')
        P2outFile.write("FOM={}, precond={}, OloP2={}, par[3]={}, par[4]={}, par[5]={} " .format(FOM, precond, OloP2, par[3], par[4], par[5]))
        P2outFile.close()

        outnamep3 = str(dirpeaks + fileindex + ".P3")
        P3outFile = open(outnamep3, 'w')
        P3outFile.write("FOM={}, precond={}, OloP3={}, par[6]={}, par[7]={}, par[8]={} " .format(FOM, precond, OloP3, par[6], par[7], par[8]))
        P3outFile.close()

        outnamep4 = str(dirpeaks + fileindex + ".P4")
        P4outFile = open(outnamep4, 'w')
        P4outFile.write("FOM={}, precond={}, OloP4={}, par[9]={}, par[10]={}, par[11]={}" .format(FOM, precond, OloP4, par[9], par[10], par[11]))
        P4outFile.close()

        outnamep5 = str(dirpeaks + fileindex + ".P5")
        P5outFile = open(outnamep5, 'w')
        P5outFile.write("FOM={}, precond={}, OloP5={}, par[12]={}, par[13]={}, par[14]={}" .format(FOM, precond, OloP5, par[12], par[13], par[14]))
        P5outFile.close()

        print("DONE!!!")
    
    if apofasi==2:
        print("DONE!!!")

    cv.Draw()
    input('Press Enter to close')

if __name__ == "__main__":
    LMOSL()
