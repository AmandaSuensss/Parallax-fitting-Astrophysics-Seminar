from matplotlib import pyplot as plt
import numpy as np
import scipy
import math

# Given VLBI astrometric data
jd = np.array([2456018.36, 2456060.25, 2456144.01, 2456185.93, 2456227.79, 2456310.56, 2456396.32, 2456445.27])
ra_hms = np.array([21.2035163, 21.2039877, 21.2048741, 21.2050142, 21.2049262, 21.2046021, 21.2050618, 21.2056562])
dec_dms = np.array([38.340643, 38.336247, 38.328707, 38.323798, 38.318108, 38.305108, 38.295017, 38.290298])
ra_err = np.array([0.0000042, 0.0000072, 0.0000025, 0.0000009, 0.0000040, 0.0000093, 0.0000059, 0.0000081])
dec_err = np.array([0.000179, 0.000143, 0.000068, 0.000027, 0.000077, 0.000249, 0.000099, 0.000182])

# Convert RA and Dec to decimal degrees
ra_deg = []
dec_deg = []
for i in range(len(jd)):
    ra_deg.append((3 + 45/60 + ra_hms[i]/3600) * 15)
    dec_deg.append(23 + 43/60 + dec_dms[i]/3600)

ra_deg = np.array(ra_deg)
dec_deg = np.array(dec_deg)

# print(ra_deg)
# print(dec_deg)

# Convert Julian Dates to time since Vernal Equinox at 2000
t = jd - 2451623.815972

# Earth's orbital parameters
jd_vernal_equinox_2000 = 2451623.815972
jd_perihelion_2000 = 2451546.708333
orbital_period = 365.25
earth_obliquity = 23.44
eccentricity = 0.0167

# Calculate the correction for declination over time due to Earth's orbit
n = 2 * np.pi / orbital_period
M = n * (t - (jd_perihelion_2000 - jd_vernal_equinox_2000))
delta_dec = earth_obliquity * np.sin(M) * eccentricity

# Apply the correction to the declination
dec_deg += delta_dec

# print(t)
# print(dec_deg)

'''
# fit the data with sin function -> scipy included
def fit_sin(t, y):

    # fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"

    tt = np.array(t)
    yy = np.array(y)

    # discrete Fourier transform: assume uniform spacing
    # frequencies corresponding to the Fourier components
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   
    Fyy = abs(np.fft.fft(yy)) # absolute values of the Fourier coefficients
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    '''

# sine model
def sinfunc(t, A, w, p, c):
    return A * np.sin(w * t + p) + c

# Gauss-Newton algorithm -> fit the sine curve without using any well-written packages
def fit_sin(t, y, max_iter=10, tol=1e-6):
    tt = np.array(t)
    yy = np.array(y)

    # initial parameter guess before iteration
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess_freq = 1.0 / (tt[-1] - tt[0])
    guess_phase = 0.0
    A, w, p, c = guess_amp, 2. * np.pi * guess_freq, guess_phase, guess_offset
    
    # iterate to find the best fit
    for _ in range(max_iter):
        # Calculate the predicted values and the Jacobian matrix
        predicted = sinfunc(tt, A, w, p, c)
        J = np.column_stack(
            [np.sin(w * tt + p),
             A * tt * np.cos(w * tt + p),
             A * np.cos(w * tt + p),
             np.ones_like(tt)]
        )

        # Update the parameters using the Gauss-Newton formula
        delta = np.linalg.lstsq(J, yy - predicted, rcond=None)[0]
        A += delta[0]
        w += delta[1]
        p += delta[2]
        c += delta[3]

        if np.linalg.norm(delta) < tol: # if converges before the max iteration, break the iteration!
            break

    # Calculate other parameters
    f = w / (2. * np.pi)
    fitfunc = lambda t: sinfunc(t, A, w, p, c) 

    # result parameter: diction parameter
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc}

# fit and obtain the coefficients
cov = fit_sin(t, dec_deg)
A = cov["amp"]
w = cov["omega"]
p = cov["phase"]
c = cov["offset"]
fitfunc = cov["fitfunc"]

# print out the parameters
print(f"Amplitude: {A}")
print(f"Omega: {w}")
print(f"Phase: {p}")
print(f"Offset: {c}")
print(f"Fit function: {fitfunc}")


# Generate a smooth curve using the fitted parameters
smooth_t = np.linspace(np.min(t), np.max(t), 1000)
smooth_dec = fitfunc(smooth_t)

plt.errorbar(t, dec_deg, yerr=dec_err, fmt='o', label='Data') # plot the scatter points with error bar
plt.plot(smooth_t, smooth_dec, label='Fitted Curve') # plot the fitted sine curve
plt.xlabel('Time since Vernal Equinox at 2000 (days)')
plt.ylabel('Declination (degrees)')
plt.legend()
plt.grid()
plt.show()

# derived parameters
parallax = A / 3600  # Convert amplitude from arcseconds to degrees
distance = 1 / parallax

# Print the estimated parallax and distance
print(f"Estimated Parallax: {parallax} arcseconds")
print(f"Estimated Distance: {distance} parsecs")