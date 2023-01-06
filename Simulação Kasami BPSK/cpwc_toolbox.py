import numpy as np
import numba
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
mpl.use('TkAgg')

def cpwc_kernel(xr, zr, xt, c, tgs, theta, g, ts):
    # --- INÍCIO DO ALGORITMO CPWC, desenvolvido por Marco. ---
    # Dimensões dados
    m = g.shape[1]
    n = g.shape[2]

    # Dimensões ROI
    m_r = zr.shape[0]
    n_r = xr.shape[0]

    # Imagem
    img = np.zeros((m_r * n_r, 1), dtype='complex128')

    for k, thetak in enumerate(theta):
        data = np.vstack((g[k], np.zeros((1, n), dtype='complex128')))

        # Calcula a distância percorrida pela onda até cada ponto da ROI e de
        # volta para cada transdutor. As distâncias são convertidas em delays
        # e então em índices. Cada linha da variável j representa um pixel na
        # imagem final, contendo o índice das amostras do sinal de Ascan para
        # todos os transdutores que contribuem para esse pixel.
        j = cpwc_roi_dist(xr, zr, xt, thetak, c, ts, tgs[k]) # modificado para cada angulo ter um delay
        # j = j.reshape(m_r * n_r, n)
        j[j >= m] = -1
        j[j < 0] = -1

        # Soma as amostras de Ascan coerentemente
        aux = np.zeros(j.shape[0], dtype='complex128')
        img[:, 0] += cpwc_sum(data, aux, j)

    f = img.reshape((m_r, n_r), order='F')

    # --- FIM DO ALGORITMO CPWC. ---
    return f


@numba.njit(parallel=True)
def cpwc_roi_dist(xr, zr, xt, theta, c, ts, tgs):
    r"""Calcula os *delays* para o DAS do algoritmo CPWC.

    Os *delays* são convertidos para índices, a partir do período de
    amostragem. Os *delays* são calculados conforme a trajetória da onda
    plana, desde o transdutor até o ponto da ROI e de volta para o transdutor.

    Parameters
    ----------
    xr : :class:`np.ndarray`
        Vetor com os valores de :math:`x` da ROI, em m.

    zr : :class:`np.ndarray`
        Vetor com os valores de :math:`z` da ROI, em m.

    xt : :class:`np.ndarray`
        Vetor com os valores de :math:`x` dos elementos do transdutor, em m.

    theta : :class:`int`, :class:`float`
        Ângulo de inclinação da onda plana, em radianos.

    c : :class:`int`, :class:`float`
        Velocidade de propagação da onda no meio.

    ts : :class:`int`, :class:`float`
        Período de amostragem do transdutor.

    tgs : :class:`int`, :class:`float`
        Tempo do gate inicial.

    Returns
    -------
    :class:`np.ndarray`
        Uma matriz de números inteiros :math:`M_r \cdot N_r` por :math:`N`, em
        que :math:`M_r` é a quantidade de elementos do vetor :math:`x`,
        :math:`N_r` é a quantidade de elementos do vetor :math:`z` e :math:`N`
        é a quantidade de elementos do transdutor.

    """

    m_r = zr.shape[0]
    n_r = xr.shape[0]
    n = xt.shape[0]

    #ti_i = np.int64(tgs / ts)
    ti_i = tgs//ts

    j = np.zeros((n_r * m_r, n), dtype='complex128')
    for i in numba.prange(n_r):
        for jj in range(m_r):
            i_i = i * m_r + jj
            di = (zr[jj] * np.cos(theta) + xr[i] * np.sin(theta))
            dv = np.sqrt(zr[jj] ** 2 + (xr[i] - xt) ** 2)
            d = np.rint((di + dv) / (c * ts))
            j[i_i, :] = d - ti_i

    return j



@numba.njit(parallel=True)
def cpwc_sum(data, img, j):
    r"""Realiza a soma para o DAS do algoritmo CPWC.

    Parameters
    ----------
    data : :class:`np.ndarray`
        Matriz :math:`M` por :math:`N` contendo os dados de aquisição.

    img : :class:`np.ndarray`
        Vetor :math:`N_r` para acumular os dados.

    j : :class:`np.ndarray`
        Matriz com os *delays* para cada ponto da ROI. Deve ser uma matriz
        :math:`M_r \cdot N_r` por :math:`N`, em que :math:`M_r` é a quantidade
        de elementos do vetor :math:`x`, :math:`N_r` é a quantidade de
        elementos do vetor :math:`z` e :math:`N` é a quantidade de elementos
        do transdutor.

    Returns
    -------
    :class:`np.ndarray`
        Vetor 1 por :math:`M_r \cdot N_r` contendo a soma no eixo 1 da matriz.

    """

    i = np.arange(j.shape[1])
    # img = np.zeros(j.shape[0])
    for jj in numba.prange(j.shape[0]):
        idx = j[jj, :]
        for ii in range(i.shape[0]):
            img[jj] += data[int(idx[ii].real), ii]

    return img