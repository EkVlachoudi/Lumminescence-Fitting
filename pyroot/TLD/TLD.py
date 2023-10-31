import numpy as np
import ctypes
from array import array as arr
import scipy.special as sc
import scipy.constants
from scipy.special import lambertw
from ROOT import TGraph, TCanvas, TF1, TMultiGraph, TLegend
from ROOT import TMinuit

# Global variables
k=scipy.constants.physical_constants['Boltzmann constant in eV/K']
K=k[0]

# Theoretical function and background
def TLD(AN, TM, E, B, T):
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

def BGR(ENTA, D0, ALA, T):
    return ENTA + D0 * 10e-8 * np.exp(T / ALA)

Npeaks = 10  # number of peaks
Npeak_pars = 4  # number of parameters per peak
N = 0  # number of data point in glow curve that is read

#Define theoretical function for total peak (function to fit)
def TL_theo1(T, par):
    TL = 0.0
# first sum up all the peak contributions
    for i in range(Npeaks):
        TL += TLD(par[Npeak_pars * i + 0], par[Npeak_pars * i + 1], par[Npeak_pars * i + 2], par[Npeak_pars * i + 3], T)

  # then add the background
    TL += BGR(par[Npeak_pars * Npeaks + 0], par[Npeak_pars * Npeaks + 1], par[Npeak_pars * Npeaks + 2], T)
    return TL

def TL_theo(t, par):
    par_array = np.frombuffer(par,
                              dtype=ctypes.c_double,
                              count=Npeak_pars * Npeaks + 3)
    return TL_theo1(t[0], par_array)

# With this method, a fit_func object can be used as a function
# The function __call__ is called like fcn by Minuit repeatedly with varying parameters
class fit_func():
    def __init__(self, N, Ts, TLs):
        self.N = N
        self.T = Ts
        self.TL_exp = TLs

    def __call__(self, npar, deriv, f, par, iflag):
        # npar=4
        f.value = 0.0
        pp = np.frombuffer(par,
                           dtype=ctypes.c_double,
                           count=Npeak_pars * Npeaks + 3)
        for i in range(self.N):
            d = self.TL_exp[i] - TL_theo1(self.T[i], pp)
            f.value = f.value + d * d / self.TL_exp[i]


def TLD_700(print_all_data = False):
    # Read parameters file
    dirpara = "/pyroot/TL/"
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
    dirdata = "/pyroot/TL/"
    datafile = input("Enter experimental data file name : ")#Enter filename with .dat
    infile = dirdata + datafile
    print("Data file to read: ", infile)

    channel, Tx, TL_exp = np.loadtxt(infile,
                                     dtype=[('', 'i4'), ('', 'f8'),
                                            ('', 'f8')],
                                     usecols=(0, 1, 2),
                                     unpack=True)
    N = len(channel)
    #Print some data lines on screen
    if print_all_data==0:
        for i in range(0,10):
            print("{} {} {}".format(channel[i], Tx[i], TL_exp[i])) # print first 10 data lines
        print("...")
        for i in range(N - 10, N):
            print("{} {} {}".format(channel[i], Tx[i], TL_exp[i])) # print last 10 data lines
    else:
        for i in range(0, N):
            print("{} {} {}".format(channel[i], Tx[i], TL_exp[i])) # print all data lines   
    print("Data point = {}".format(N))
    
    #Select the Low and Upper limits of the Curve Fitting
    LL = input("Enter low limit LL: ")
    LU = input("Enter upper limit LU: ")
    L = int(LL)
    N = int(LU) - int(LL) - 1

    for i in range(N):
        TL_exp[i] = TL_exp[L]
        Tx[i] = 0.0 + Tx[L]
        L += 1

    precond = input("Enter the preconditioning: ")

    #Find min, max for the plots
    gmin = TL_exp[0]
    gmax = TL_exp[0]
    for i in range(N):
        if TL_exp[i] < gmin:
            gmin = TL_exp[i]
        if TL_exp[i] > gmax:
            gmax = TL_exp[i]
    # gmin = np.min(TL_exp)
    # gmax = np.max(TL_exp)

    #Create canvas
    cv = TCanvas("cv_TL_fit", "cv_TL_fit")
    cv.SetFillColor(0)

    #Initialize TMinuit with maximum of :number of parameters*number of peaks+2
    ierflg = ctypes.c_int()
    arglist = arr('d', 2*[0.1]) # ser error definition
    
    mini = TMinuit(Npeaks * Npeak_pars + 3)
    fcn = fit_func(N, Tx, TL_exp)
    mini.SetFCN(fcn)  # set function to minimize
    arglist[0] = 1
    mini.mnexcm("SET ERR", arglist, 1, ierflg)

    #Set starting values and step sizes for parameters
    #1st peak
    ipar = 0
    mini.mnparm(ipar, "Im1", peaks[0][0], peaks[0][1], peaks[0][2],peaks[0][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm1", peaks[1][0], peaks[1][1], peaks[1][2], peaks[1][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E1", peaks[2][0], peaks[2][1], peaks[2][2], peaks[2][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B1", peaks[3][0], peaks[3][1], peaks[3][2], peaks[3][3], ierflg)
    ipar += 1
    #2nd peak
    mini.mnparm(ipar, "Im2", peaks[4][0], peaks[4][1], peaks[4][2], peaks[4][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm2", peaks[5][0], peaks[5][1], peaks[5][2], peaks[5][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E2", peaks[6][0], peaks[6][1], peaks[6][2], peaks[6][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B2", peaks[7][0], peaks[7][1], peaks[7][2], peaks[7][3], ierflg)
    ipar += 1
    #3rd peak
    mini.mnparm(ipar, "Im3", peaks[8][0], peaks[8][1], peaks[8][2], peaks[8][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm3", peaks[9][0], peaks[9][1], peaks[9][2], peaks[9][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E3", peaks[10][0], peaks[10][1], peaks[10][2], peaks[10][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B3", peaks[11][0], peaks[11][1], peaks[11][2], peaks[11][3], ierflg)
    ipar += 1
    #4th peak
    mini.mnparm(ipar, "Im4", peaks[12][0], peaks[12][1], peaks[12][2], peaks[12][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm4", peaks[13][0], peaks[13][1], peaks[13][2], peaks[13][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E4", peaks[14][0], peaks[14][1], peaks[14][2], peaks[14][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B4", peaks[15][0], peaks[15][1], peaks[15][2], peaks[15][3], ierflg)
    ipar += 1
    #5th peak
    mini.mnparm(ipar, "Im5", peaks[16][0], peaks[16][1], peaks[16][2], peaks[16][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm5", peaks[17][0], peaks[17][1], peaks[17][2], peaks[17][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E5", peaks[18][0], peaks[18][1], peaks[18][2], peaks[18][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B5", peaks[19][0], peaks[19][1], peaks[19][2], peaks[19][3], ierflg)
    ipar += 1
    #6th peak
    mini.mnparm(ipar, "Im6", peaks[20][0], peaks[20][1], peaks[20][2], peaks[20][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm6", peaks[21][0], peaks[21][1], peaks[21][2], peaks[21][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E6", peaks[22][0], peaks[22][1], peaks[22][2], peaks[22][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B6", peaks[23][0], peaks[23][1], peaks[23][2], peaks[23][3], ierflg)
    ipar += 1
    #7th peak
    mini.mnparm(ipar, "Im7", peaks[24][0], peaks[24][1], peaks[24][2], peaks[24][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm7", peaks[25][0], peaks[25][1], peaks[25][2], peaks[25][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E7", peaks[26][0], peaks[26][1], peaks[26][2], peaks[26][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B7", peaks[27][0], peaks[27][1], peaks[27][2], peaks[27][3], ierflg)
    ipar += 1
    #8th peak
    mini.mnparm(ipar, "Im8", peaks[28][0], peaks[28][1], peaks[28][2], peaks[28][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm8", peaks[29][0], peaks[29][1], peaks[29][2], peaks[29][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E8", peaks[30][0], peaks[30][1], peaks[30][2], peaks[30][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B8", peaks[31][0], peaks[31][1], peaks[31][2], peaks[31][3], ierflg)
    ipar += 1
    #9th peak
    mini.mnparm(ipar, "Im9", peaks[32][0], peaks[32][1], peaks[32][2], peaks[32][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm9", peaks[33][0], peaks[33][1], peaks[33][2], peaks[33][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E9", peaks[34][0], peaks[34][1], peaks[34][2], peaks[34][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B9", peaks[35][0], peaks[35][1], peaks[35][2], peaks[35][3], ierflg)
    ipar += 1
    #10th peak
    mini.mnparm(ipar, "Im10", peaks[36][0], peaks[36][1], peaks[36][2], peaks[36][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "Tm10", peaks[37][0], peaks[37][1], peaks[37][2], peaks[37][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "E10", peaks[38][0], peaks[38][1], peaks[38][2], peaks[38][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "B10", peaks[39][0], peaks[39][1], peaks[39][2], peaks[39][3], ierflg)
    ipar += 1
    #background
    mini.mnparm(ipar, "ENTA", peaks[40][0], peaks[40][1], peaks[40][2], peaks[40][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "D0", peaks[41][0], peaks[41][1], peaks[41][2], peaks[41][3], ierflg)
    ipar += 1
    mini.mnparm(ipar, "ALA", peaks[42][0], peaks[42][1], peaks[42][2], peaks[42][3], ierflg)
    
    #Minimization step
    arglist[0] = 2000  # max number of calls to fcn before giving up
    arglist[1] = 1e-20  # tolerance

    mini.mnexcm("SIMPLEX", arglist, 2, ierflg)
    mini.mnexcm("MIGRAD", arglist, 2, ierflg)  # execute the minimisation

    #Get result from Minuit
    par = np.ndarray(shape=1000, dtype=float)
    epar = np.ndarray(shape=1000, dtype=float)
    for i in range(Npeaks * Npeak_pars + 3):
        temp_par = ctypes.c_double()
        temp_epar = ctypes.c_double()
        # retrieve parameters and errors
        mini.GetParameter(int(i), temp_par, temp_epar)
        par[i] = temp_par.value
        epar[i] = temp_epar.value

    #Evaluating Individual glow-peaks
    i=0 
    P1 = np.zeros(10000, dtype=float)
    for i in range(0, N): P1[i]=TLD(par[0],par[1],par[2],par[3],Tx[i])
    P2 = np.zeros(10000, dtype=float)
    if Npeaks>1:
        for i in range(0, N): P2[i]=TLD(par[4],par[5],par[6],par[7],Tx[i])
    else:
        for i in range (0,N): P2[i]=0
    P3 = np.zeros(10000, dtype=float)
    if Npeaks>2:
        for i in range(0, N): P3[i]=TLD(par[8],par[9],par[10],par[11],Tx[i])
    else:
        for i in range (0,N): P3[i]=0
    P4=np.zeros(10000, dtype=float)
    if Npeaks>3:
        for i in range(0, N): P4[i]=TLD(par[12],par[13],par[14],par[15],Tx[i])
    else:
        for i in range (0,N): P4[i]=0    
    P5=np.zeros(10000, dtype=float)
    if Npeaks>4:
        for i in range(0, N): P5[i]=TLD(par[16],par[17],par[18],par[19],Tx[i])
    else:
        for i in range (0,N): P5[i]=0     
    P6=np.zeros(10000, dtype=float)
    if Npeaks>5:
        for i in range(0, N): P6[i]=TLD(par[20],par[21],par[22],par[23],Tx[i])
    else:
        for i in range (0,N): P6[i]=0 
    P7=np.zeros(10000, dtype=float)
    if Npeaks>6:
        for i in range(0, N): P7[i]=TLD(par[24],par[25],par[26],par[27],Tx[i])
    else:
        for i in range (0,N): P7[i]=0 
    P8=np.zeros(10000, dtype=float)
    if Npeaks>7:
        for i in range(0, N): P8[i]=TLD(par[28],par[29],par[30],par[31],Tx[i])
    else:
        for i in range (0,N): P8[i]=0 
    P9=np.zeros(10000, dtype=float)
    if Npeaks>8:
        for i in range(0, N): P9[i]=TLD(par[32],par[33],par[34],par[35],Tx[i])
    else:
        for i in range (0,N): P9[i]=0 
    P10=np.zeros(10000, dtype=float)
    if Npeaks>9:
        for i in range(0, N): P10[i]=TLD(par[36],par[37],par[38],par[39],Tx[i])
    else:
        for i in range (0,N): P10[i]=0 
            
    PG = np.zeros(10000, dtype=float)
    for i in range(0, N):
        PG[i] = BGR(par[Npeak_pars * Npeaks + 0], par[Npeak_pars * Npeaks + 1], par[Npeak_pars * Npeaks + 2], Tx[i])

    TLtheo = np.zeros(10000, dtype=float)
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
        OloExp+=TL_exp[i]
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
    DYS=0.0
    i=0
    for i in range(N):
        FOM=round(100*(np.sum(abs(TL_exp[i]-TLtheo[i]))/(np.sum(TLtheo[i]))),3)
        DYS+=abs(TL_exp[i]-TLtheo[i])
    FOM=np.divide(DYS,OloTheo)   
    print("DYS = " , DYS )
    print("OloTheo = ", OloTheo)
    print("FOM ={}% ".format(FOM))

    

    #Make Graphs
    gx = np.zeros(10000)
    gy = np.zeros(10000)
    for i in range(N):
        gx[i] = Tx[i]
        gy[i] = TL_exp[i]

    g = TGraph(N, gx, gy)
    g.SetName("g")
    g.SetMarkerColor(1)
    g.SetMarkerStyle(21)
    g.SetMarkerSize(0.7)
    g.SetTitle(";T (K);TL signal (a.u.)")
    g.GetYaxis().SetTitleOffset(1.25)

    #Plot the fitted
    fitted = TF1("fitted", TL_theo, Tx[0], Tx[N - 1], Npeaks * Npeak_pars + 3)
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

    x3=np.zeros(10000, dtype=float) 
    for i in range(0,N): x3[i]=Tx[i]
    y3=np.zeros(10000, dtype=float) 
    for i in range(0,N): y3[i]=P3[i]
    g3 = TGraph(N,x3,y3)
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

    x6=np.zeros(10000, dtype=float) 
    for i in range(0,N): x6[i]=Tx[i]
    y6=np.zeros(10000, dtype=float) 
    for i in range(0,N): y6[i]=P6[i]
    g6 = TGraph(N,x6,y6)
    g6.SetNameTitle("g6", "g6")
    g6.SetLineColor(8)
    g6.SetLineWidth(2)

    x7=np.zeros(10000, dtype=float) 
    for i in range(0,N): x7[i]=Tx[i]
    y7=np.zeros(10000, dtype=float) 
    for i in range(0,N): y7[i]=P7[i]
    g7 = TGraph(N,x7,y7)
    g7.SetNameTitle("g7", "g7")
    g7.SetLineColor(9)
    g7.SetLineWidth(2)

    x8=np.zeros(10000, dtype=float) 
    for i in range(0,N): x8[i]=Tx[i]
    y8=np.zeros(10000, dtype=float) 
    for i in range(0,N): y8[i]=P8[i]
    g8 = TGraph(N,x8,y8)
    g8.SetNameTitle("g8", "g8")
    g8.SetLineColor(46)
    g8.SetLineWidth(2)

    xg = np.zeros(10000, dtype=float)
    yg = np.zeros(10000, dtype=float)
    for i in range(N):
        xg[i] = Tx[i]
        yg[i] = PG[i]
    gg = TGraph(N, xg, yg)
    gg.SetNameTitle("gg", "gg")
    gg.SetLineColor(12)
    gg.SetLineWidth(2)

    gmulti = TMultiGraph()
    gmulti.SetTitle(";T (K);TL signal (a.u.)")
    gmulti.Add(g, "p")
    gmulti.Add(g1, "l")
    gmulti.Add(g2, "l")
    gmulti.Add(g3, "l")
    gmulti.Add(g4, "l")
    gmulti.Add(g5, "l")
    gmulti.Add(g6, "l")
    gmulti.Add(g7, "l")
    gmulti.Add(g8, "l")
    gmulti.Add(gg, "l")
    gmulti.GetListOfFunctions().Add(fitted)
    gmulti.Draw("a")

    leg = TLegend(0.1, 0.7, 0.3, 0.9)
    leg.AddEntry(g, "g", "p")
    leg.AddEntry(g1, "g1", "l")
    leg.AddEntry(g2, "g2", "l")
    leg.AddEntry(g3, "g3", "l")
    leg.AddEntry(g4, "g4", "l")
    leg.AddEntry(g5, "g5", "l")
    leg.AddEntry(g6, "g6", "l")
    leg.AddEntry(g7, "g7", "l")
    leg.AddEntry(g8, "g8", "l")
    leg.AddEntry(gg, "gg", "l")
    leg.AddEntry(fitted, "fitted", "l")
    leg.Draw()


    #Saving the results on disk
    #Three types of results are saved
    # A. The Glow-curve and its Individual TL peaks
    # B. The Curve fitting parameters for all TL peaks
    # C. A series of files are opened, each one corresponding to an individual TL peak i.e 
    print('Run file was: ', infile)
    dirout=str('/pyroot/TL/out/' )
    filep=input('Output data file name: ')
    outname1 = str(dirout + filep+'.out')
    outputFile = open(outname1, 'w')
    for i in range(N):
        outputFile.write("{} {} {} {} {} {} {} {} {} {} {} \n" .format(Tx[i], TL_exp[i], TL_theo1(Tx[i],par), P1[i], P2[i], P3[i], P4[i], P5[i], P6[i], P7[i], PG[i]))  
    outputFile.close()
    print("=========================================\n")

    dirpar = str('/pyroot/TL/para/')
    filepar = input('Output parameter file name: ')
    outname2 = str(dirpar + filepar +'.par')
    parputFile = open(outname2,'w')
    parputFile.write( "Theo={}, FOM={}, LL={}, LU={} " .format(OloTheo, FOM, LL, LU))
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
        dirpeaks = str("/pyroot/TL/para/")
        fileindex = input("Output parameter file name: ")  
        
        outnamep1 = str(dirpeaks + fileindex + ".P1")
        P1outFile = open(outnamep1, 'w')
        P1outFile.write("FOM={}, precond={}, OloP1={}, par[0]={}, par[1]={}, par[2]={}, par[3]={} " .format(FOM, precond, OloP1, par[0], par[1], par[2], par[3]))
        P1outFile.close()
        
        outnamep2 = str(dirpeaks + fileindex + ".P2")
        P2outFile = open(outnamep2, 'w')
        P2outFile.write("FOM={}, precond={}, OloP2={}, par[4]={}, par[5]={}, par[6]={}, par[7]={} " .format(FOM, precond, OloP2, par[4], par[5], par[6], par[7]))
        P2outFile.close()

        outnamep3 = str(dirpeaks + fileindex + ".P3")
        P3outFile = open(outnamep3, 'w')
        P3outFile.write("FOM={}, precond={}, OloP3={}, par[8]={}, par[9]={}, par[10]={}, par[11]={}" .format(FOM, precond, OloP3, par[8], par[9], par[10], par[11]))
        P3outFile.close()

        outnamep4 = str(dirpeaks + fileindex + ".P4")
        P4outFile = open(outnamep4, 'w')
        P4outFile.write("FOM={}, precond={}, OloP4={}, par[12]={}, par[13]={}, par[14]={}, par[15]={}" .format(FOM, precond, OloP4, par[12], par[13], par[14], par[15]))
        P4outFile.close()

        outnamep5 = str(dirpeaks + fileindex + ".P5")
        P5outFile = open(outnamep5, 'w')
        P5outFile.write("FOM={}, precond={}, OloP5={}, par[16]={}, par[17]={}, par[18]={}, par[19]={}" .format(FOM, precond, OloP5, par[16], par[17], par[18], par[19]))
        P5outFile.close()

        outnamep6 = str(dirpeaks + fileindex + ".P6")
        P6outFile = open(outnamep6, 'w')
        P6outFile.write("FOM={}, precond={}, OloP6={}, par[20]={}, par[21]={}, par[22]={}, par[23]={}" .format(FOM, precond, OloP6, par[20], par[21], par[22], par[23]))
        P6outFile.close()

        outnamep7 = str(dirpeaks + fileindex + ".P7")
        P7outFile = open(outnamep7, 'w')
        P7outFile.write("FOM={}, precond={}, OloP7={}, par[24]={}, par[25]={}, par[26]={}, par[27]={}" .format(FOM, precond, OloP7, par[24], par[25], par[26], par[27]))
        P7outFile.close()

        outnamep8 = str(dirpeaks + fileindex + ".P8")
        P8outFile = open(outnamep8, 'w')
        P8outFile.write("FOM={}, precond={}, OloP8={}, par[28]={}, par[29]={}, par[30]={}, par[31]={}" .format(FOM, precond, OloP8, par[28], par[29], par[30], par[31]))
        P8outFile.close()

        outnamep9 = str(dirpeaks + fileindex + ".P9")
        P9outFile = open(outnamep9, 'w')
        P9outFile.write("FOM={}, precond={}, OloP9={}, par[32]={}, par[33]={}, par[34]={}, par[35]={}" .format(FOM, precond, OloP9, par[32], par[33], par[34], par[35]))
        P9outFile.close()

        outnamep10 = str(dirpeaks + fileindex + ".P10")
        P10outFile = open(outnamep10, 'w')
        P10outFile.write("FOM={}, precond={}, OloP10={}, par[36]={}, par[37]={}, par[38]={}, par[39]={}" .format(FOM, precond, OloP10, par[36], par[37], par[38], par[39]))
        P10outFile.close()

        print("DONE!!!")
    
    if apofasi==2:
        print("DONE!!!")

    cv.Draw()
    input('Press Enter to close')

if __name__ == "__main__":
    TLD_700()
