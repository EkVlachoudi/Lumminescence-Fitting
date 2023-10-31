import numpy as np
from scipy.integrate import solve_ivp
import iminuit
import matplotlib.pyplot as plt

kB = 8.617e-5
Hrate = 1

# Read experimental data file
dirdata = "/python/DIFF/"
datafile = input("Experimental data file name: ")
infile = dirdata + datafile
print("Data file to read:", infile)

channel, T_data, TL_data = np.loadtxt(infile,
                                     dtype=[('', 'i4'), ('', 'f8'),
                                            ('', 'f8')],
                                     usecols=(0, 1, 2),
                                     unpack=True)
N = len(channel)

# Define your ODE system
def deriv(t, y, *par):
    T, n, m, nc = y
    N0, An, Am, s, E = par

    dTdt = Hrate
    dndt = -n * s * np.exp(-E / (kB * T)) + nc * An * (N0 - n)
    dncdt = -nc * An * (N0 - n) - m * Am * nc + n * s * np.exp(-E / (kB * T))
    dmdt = -m * Am * nc
    return [dTdt, dndt, dmdt, dncdt]

# Define the range of time values based on your experimental data
t_start = min(channel)
t_end = max(channel)

# Define an objective function to compute the sum of squared residuals
def objective_function(T0, n0, m0, nc0, N0, An, Am, s, E):
    t_span = (t_start, t_end)
    y0 = [T0, n0, m0, nc0]
    params = (N0, An, Am, s, E)

    # Solve the ODE using the 'ode' solver
    result = solve_ivp(deriv, t_span, y0, args=params, method='LSODA', atol=1e-8, rtol=1e-6, t_eval=channel)

    if not result.success:
        return 1e10

    T_model = result.y[0]
    residuals = T_model - T_data
    return np.sum(residuals**2)

# Set initial conditions and measured data
T0, n0, m0, nc0 = float(100), float(1e10), float(1e10), float(0)

# Define initial values for additional parameters
N0, An, Am, s, E = float(1.0e9), float(1.7e4), float(337.8), float(0.97), float(0.028)


# Initialize Minuit
m = iminuit.Minuit(objective_function, T0=T0, n0=n0, m0=m0, nc0=nc0, N0=N0, An=An, Am=Am, s=s, E=E)

# Perform the minimization
m.migrad()

# After fitting
fitted_params = m.values

# Generate the fitting curve
params = {
    'T0': fitted_params['T0'],
    'n0': fitted_params['n0'],
    'm0': fitted_params['m0'],
    'nc0': fitted_params['nc0'],
    'N0': fitted_params['N0'],
    'An': fitted_params['An'],
    'Am': fitted_params['Am'],
    's': fitted_params['s'],
    'E': fitted_params['E']
}

# Optionally, you can also print other relevant information
print("Covariance Matrix:")
print(m.covariance)

# Calculate the sum squared residuals value
sum_squared_residuals = m.fval
# Get the number of data points
N_data = len(T_data)
# Get the number of free parameters in your model
N_parameters = len(fitted_params)
# Calculate the Figure of Merit (FOM)
FOM = 1.0 / (sum_squared_residuals / (N_data - N_parameters))
print("FOM ={}% ".format(FOM))

# Generate the fitting curve at the same time points as channel
t_eval = np.linspace(t_start, t_end, N)  # Create time points for evaluation
fitted_curve = solve_ivp(deriv, (t_start, t_end), [T0, n0, m0, nc0], args=(N0, An, Am, s, E), method='LSODA', t_eval=t_eval).y[0]

# Plot the experimental data and the fitting curve
plt.figure(figsize=(8, 6))
plt.scatter(t_eval, T_data, label='Experimental Data', color='blue', marker='o')
plt.plot(t_eval, fitted_curve, label='Fitting Curve', color='red', linestyle='-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.title('Experimental Data and Fitting Curve of solving ODE')
plt.grid()
plt.show()
