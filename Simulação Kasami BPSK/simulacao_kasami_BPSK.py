import numpy as np
import coded_waves as cw
from cpwc_toolbox import cpwc_kernel
from scipy.signal import gausspulse, hilbert
from PWI_SIR import sim_pwi_phantom
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import matplotlib as mpl
import time
mpl.use('TkAgg')

Dx = .1e-3
Dz = Dx
Dt = 8e-9

Lx = 50e-3   # Block width
Lz = 50e-3   # Block height
Lt = 32e-6    # Simulation time
'''
bits = 5 : Lt = 10e-6
bits = 6 : Lt = 16e-6
bits = 7 : Lt = 31e-6
bits = 8 : Lt = 54e-6
? Lt = 2**(bits-2)e-6 ?
'''

Nx = round(Lx / Dx)
Nz = round(Lz / Dz)
Nt = round(Lt / Dt)

c_som = 20e3
fc = 5e6
bwp = 0.9
c = c_som * Dt / Dx
t = np.arange(Nt)
print('Velocidade do som: ', c)

# Receiver locations
# dist_elem = 400e-06 # distancia entre elementos do transdutor
num_elementos = 128
dist_elem = Lx / num_elementos
irx = np.arange(0, Lx, dist_elem) // Dx
# print("Numero de elementos: ", len(irx))
print("lambda/dist_elem: ", c_som/(fc*dist_elem))
irx = np.int32(irx)
# irz = np.zeros_like(irx) + Np + dNp
irz = np.zeros_like(irx)

angulos = np.array([-0.8, -0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6,  0.8])
# angulos = [0]

COM_CODIGO = True

if(COM_CODIGO):
    bits = 7
    bin_code = cw.kasami_large(bits, len(angulos))  # Kasami large set codes
    coded_waves = cw.BPSK(bin_code, fc, 1 / Dt)  # BPSK
    coded_waves = np.pad(coded_waves, ((0, 0), (0, Nt - len(coded_waves[0]))), constant_values=0)  # Fill with zeros
    # CONVOLUÇÃO RESPOSTA AO IMPULSO DO TRANSDUTOR NA EMISSÃO E NA RECEPÇÃO
    h = gausspulse(Dt * t - 3 / (fc), fc, bwr=-3)
    H = np.fft.fft(h)
    X = np.fft.fftn(coded_waves, axes=[-1])
    Y = X * H * H
    coded_waves = np.fft.ifftn(Y, axes=[-1]).real
else:
    wavelet_mother = gausspulse(Dt*t-3/(fc), fc)
    coded_waves = np.zeros_like(t, dtype='float32')
    coded_waves[:len(wavelet_mother)] += wavelet_mother.astype(coded_waves.dtype)
    # plt.figure(0)
    # plt.plot(coded_waves)
    # plt.title('Sinal Emitido')
    coded_waves = np.tile(coded_waves, (len(angulos), 1))

# phantom = [[Nx//2, 50], [166, 50], [333, 50]]
phantom = [[100, 50], [400, 50], [256, 50]]
# phantom = [[0, 10]]
# numero_de_refletores = 8
# arange_Nx = np.linspace(222, Nx, numero_de_refletores)
# arange_Nx = np.arange(-500, Nx+500, dist_elem/Dx)
# phantom = np.stack((arange_Nx, np.full(np.shape(arange_Nx), 50)), axis=-1)

start = time.time()
a = sim_pwi_phantom(angulos, phantom, coded_waves, irx, irz, Nt, dist_elem, c, Dx)
end = time.time()
print("Sim time:", end-start)
a = gaussian_filter(a, 1, mode='constant')

plt.figure(0)
a1 = np.sum(a, axis = 0)
a2 = np.sum(a1, axis = 0)
plt.imshow(a2.T, aspect = 'auto', interpolation='nearest')
plt.title('Resposta ao impulso espacial')
plt.colorbar()

# pos_correlacao = cw.correlacionar_sinal(a2, coded_waves, 2.075)
pos_correlacao = cw.POC(a2, coded_waves, 3/(fc*Dt))

x = np.arange(0, Lx, Dx)
z = np.arange(0, Lx, Dx)
xt = np.arange(0, Lx, dist_elem)

ti = np.zeros_like(angulos)
for i in range(len(angulos)):
    if (angulos[i] >= 0):
        ti[i] = -3 / (fc * Dt)
    else:
        ti[i] = -3 / (fc * Dt) + Lx * np.sin(angulos[i]) / (c_som * Dt)

if(COM_CODIGO):
    sinal_analitico = np.swapaxes(pos_correlacao, 1, 2)
    sinal_analitico = hilbert(sinal_analitico, axis=1)
    plt.figure(2)
    plt.imshow(np.abs(sinal_analitico[0]), aspect='auto')
    plt.title('pré CPWC')
    f = cpwc_kernel(x, z, xt, c_som, ti * Dt, angulos, sinal_analitico, Dt)
else:
    sinal_analitico = np.sum(a, axis=1)
    sinal_analitico = np.swapaxes(sinal_analitico, 1, 2)
    sinal_analitico = hilbert(sinal_analitico, axis=1)
    f = cpwc_kernel(x, z, xt, c_som, ti * Dt, angulos, sinal_analitico, Dt)

plt.figure(1)
linex = np.arange(0, 500, 1)
linez = np.zeros_like(linex) + 50
plt.plot(linex, linez,'w--', label='Posição no Phantom z = 50')
plt.legend(loc=4)

plt.imshow(np.abs(f), aspect='auto')
plt.title('Pós CPWC')
plt.colorbar()
