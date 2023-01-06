import numpy as np
from scipy.ndimage import shift

def sim_e_r(idx_e, idx_r, irx, irz, coded_wave, delay, scatter, c):
    d_e = np.sqrt((irx[idx_e] - scatter[0]) ** 2 + (scatter[1] - irz[idx_e]) ** 2)  # distancia do emissor ao scatter
    d_r = np.sqrt((irx[idx_r] - scatter[0]) ** 2 + (scatter[1] - irz[idx_r]) ** 2)  # distancia do receptor ao scatter
    tau = (d_e + d_r + delay[idx_e]) / c  # tempo de viajem total da onda
    y = shift(coded_wave, tau) / ((2*np.pi)**2 * d_e * d_r)
    return y

def sim_pw(angle, irx, irz, Nt, scatter, coded_wave, dist_elem, c, Dx):
    '''
    angle: scalar, irx: vetor com a posição dos elementos em x, scatter: coordenada (X, Z)
    irz: vetor com a posição dos elementos em z, Nt: tempo de simulação
    coded_wave: vetor 1 por T, dist_elem: scalar, c: scalar, Dx: scalar
    '''
    N_elem = len(irx)
    delay = np.arange(len(irx)) * dist_elem * np.sin(angle) / (Dx)
    delay -= np.min(delay)
    y = np.zeros((N_elem, N_elem, Nt))
    for e in range(N_elem):
        for r in range(N_elem):
            y[e, r, :] = sim_e_r(e, r, irx, irz, coded_wave, delay, scatter, c)
    return y

def sim_pwi(angles, scatter, coded_waves, irx, irz, Nt, dist_elem, c, Dx):
    y = np.zeros((len(angles), len(irx), len(irx), Nt))
    for i in range(len(angles)):
        y[i] += sim_pw(angles[i], irx, irz, Nt, scatter, coded_waves[i], dist_elem, c, Dx)
    return y

def sim_pwi_phantom(angles, scatters, coded_waves, irx, irz, Nt, dist_elem, c, Dx):
    # scatters: matriz com as linhas contendo as coordenadas dos scatters
    y = np.zeros((len(angles), len(irx), len(irx), Nt)) # Angulos x Emissores x Receptores x Tempo
    for scatter in scatters:
        y += sim_pwi(angles, scatter, coded_waves, irx, irz, Nt, dist_elem, c, Dx)
    return y

