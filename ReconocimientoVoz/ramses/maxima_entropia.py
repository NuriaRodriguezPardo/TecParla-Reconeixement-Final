import numpy as np
from scipy.fft import fft
from scipy.linalg import toeplitz, inv
from scipy.signal import lfilter

def maximaEntropia(x, orden):
    N = len(x)
    corr = np.correlate(x, x, mode='full')[N - 1 : N + orden] # Calculamos la autocorrelación de la señal
    Rxx = toeplitz(corr[: -1]) # Construimos la matriz de autocorrelación
    Prs = corr[1: ] # Construimos el vector de proyección referencia datos
    hlp = inv(Rxx) @ Prs #  coeficientes del predictor óptimo
    #  filtro reconstructor
    # Se trata de un filtro IIR...
    nume = [1] # ... cuyo numerador es uno ...
    deno = np.concatenate(([1], -hlp)) # ... y su denominador es la concatenación de uno seguido de los coeficientes de predicción lineal cambiados de signo.
    # Calculamos la respuesta impulsional del filtro reconstructor filtrando una delta de
    # Dirac de la misma longitud que la señal a analizar
    delta = np.zeros(N); delta[0] = 1
    hrec = lfilter(nume, deno, delta)
    # Ajustamos la potencia de la respuesta impulsional para que sea igual a la de la señal
    hrec *= np.std(x) / np.std(hrec)
    return np.abs(fft(hrec)) ** 2 # estimación de máxima entropía del espectro de la señal