import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, max_len_seq, firls, firwin2, firwin, fftconvolve, unit_impulse, gausspulse
from scipy.ndimage import shift
import scipy.io
from scipy.linalg import hadamard


def delta(n):
    return 1. * (n == 0)


def Golay_sequences_recursive(n):
    # Gera uma sequencia de Golay de acordo a orden n
    # as sequencias terão tamanho 2^n
    A = np.zeros(2 ** n)
    B = np.zeros(2 ** n)
    len_a = 2 ** n
    a = np.zeros(len_a)
    b = np.zeros(len_a)

    def b_i(n, i):
        b0 = 0
        if (n == 0):
            b0 = delta(i)
        else:
            b0 = a_i(n - 1, i) - b_i(n - 1, i - 2 ** (n - 1))
        return b0

    def a_i(n, i):
        a0 = 0
        if (n == 0):
            a0 = delta(i)
        else:
            a0 = a_i(n - 1, i) + b_i(n - 1, i - 2 ** (n - 1))
        return a0

    for i in range(len_a):
        a[i] = a_i(n, i)
        b[i] = b_i(n, i)
    return a, b


def gold_codes(n, length=None):
    code = []
    for i in range(n):
        np.random.seed(i)
        seq1 = max_len_seq(n, state=np.random.randint(0, 2, n), taps=[2], length=length)[0]
        seq2 = max_len_seq(n, state=np.random.randint(0, 2, n), taps=[1, 2, 3], length=length)[0]
        code.append(np.logical_xor(seq1, seq2) * 2 - 1)
    return code


def kasami_small(n, qnt_codigos):
    q = int(1 + 2 ** (n / 2))
    codes = []
    for k in range(qnt_codigos):
        np.random.seed(k)
        seq1 = max_len_seq(n, state=np.random.randint(0, 2, n), taps=[1, 2, 3])[0]
        seq2 = seq1[::q]
        code = np.zeros_like(seq1)
        for i in range(len(seq1)):
            code[i] = np.logical_xor(seq1[i], seq2[i % len(seq2)]) * 2 - 1
        codes.append(code)
    return codes


def kasami_large(bits, qnt_codigos):
    n = bits
    q = int(1 + 2 ** (n / 2))
    codes = []
    for k in range(qnt_codigos):
        np.random.seed(k)
        seq1 = max_len_seq(n, state=np.random.randint(0, 2, n), taps=[2])[0]
        seq2 = seq1[::q]
        seq3 = max_len_seq(n, state=np.random.randint(0, 2, n), taps=[1, 2, 3])[0]
        code = np.zeros_like(seq1)
        for i in range(len(seq1)):
            aux = np.logical_xor(seq1[i], seq2[i % len(seq2)])
            code[i] = np.logical_xor(aux, seq3[i]) * 2 - 1
            # code[i] = np.logical_xor(aux, seq3[i])*-1
        codes.append(code)
    return codes


def BPSK(codes, fc, fs):
    """
    codifica a fase de uma senoide de acordo com um código
    codes: matriz com os códigos em cada linha dela
    fc: frequencia central da onda
    fs: frequencia de amostragem
    """
    Na = int(np.round(fs / fc))
    t0 = np.linspace(0, 1, Na, endpoint=False)
    y = np.zeros((len(codes), len(t0)*len(codes[0])))
    for i in range(len(codes)):
        t = np.tile(t0, len(codes[i]))
        bin = np.repeat(codes[i], Na)
        y[i] = np.sin(2 * np.pi * t) * bin
    return y

def correlacionar_sinal(sinalRecebido, ondas_codificadas, time_shift):
    """
    Correlaciona a matriz do sinal de eco com os diversos códigos no tempo
    Parameters
    ----------
    sinalRecebido: sinal de eco nas dimensões: Elementos x Tempo
    ondas_codificadas: Códigos x Tempo
    time_shift: deslocamento temporal aplicado após a correlação

    Returns
    -------
    sinaisFiltrados: resultado nas dimensões: Códigos x Elementos x Tempo
    """
    tf = np.shape(sinalRecebido)[1]
    sinaisFiltrados = np.zeros((len(ondas_codificadas), len(sinalRecebido), tf))
    for k in range(len(ondas_codificadas)):
        for i in range(len(sinalRecebido)):
            sinaisFiltrados[k, i, :] = correlate(sinalRecebido[i], ondas_codificadas[k], mode='same')[:tf]
            sinaisFiltrados[k, i, :] = shift(sinaisFiltrados[k, i], -tf / time_shift)
    return sinaisFiltrados

if __name__ == '__main__':
    n = 4
    # max_len_seq = linear-feedback shift register (LFSR)
    bits = kasami_large(5, 4)  # dim(n, 2^n-1)
    y = BPSK(bits, 5e6, 120e6)
    # y = filtro_binario(bits, 12e-6, 6.67e-9, 5e6, 0.9)
    # amostra_por_bit = round(1/(2*5e6*6.67e-9))
    # bits = (np.repeat(bits, amostra_por_bit, axis=1)-1)/2
    # plt.show()
    # exit()
    # y = []
    # for k in range(0, n):
    #     y.append(binary_phase_shift_keying(H[k]))
    #     # y.append(H[k])
    v = np.zeros(n * len(y[0]))
    for i in range(n):
        v[i * len(y[0]):(i + 1) * len(y[0])] = y[i]

    fig = plt.figure(figsize=(5, 3), layout="constrained")
    fig.suptitle('Kasami Large Set n=' + str(n))
    spec = fig.add_gridspec(5, 2)

    # ax5 = fig.add_subplot(spec[5, :])
    # ax5.set_title(str(15)+" amostra por bit")
    # ax5.plot(bits[0], 'bo')
    # ax5.plot(bits[0])

    ax0 = fig.add_subplot(spec[0, :])
    ax0.set_title('Sinal codificado com ' + str(n) + ' códigos em sequência')
    ax0.plot(v)

    ax10 = fig.add_subplot(spec[1, 0])
    ax10.set_title('Onda Código 1')
    ax10.plot(y[0])
    ax11 = fig.add_subplot(spec[1, 1])
    ax11.set_title('Correlação Sinal com Onda 1')
    ax11.plot(correlate(v, y[0]))

    ax20 = fig.add_subplot(spec[2, 0])
    ax20.set_title('Onda Código 2')
    ax20.plot(y[1])
    ax21 = fig.add_subplot(spec[2, 1])
    ax21.set_title('Correlação Sinal com Onda 2')
    ax21.plot(correlate(v, y[1]))

    ax30 = fig.add_subplot(spec[3, 0])
    ax30.set_title('Onda Código 3')
    ax30.plot(y[2])
    ax31 = fig.add_subplot(spec[3, 1])
    ax31.set_title('Correlação Sinal com Onda 3')
    ax31.plot(correlate(v, y[2]))

    ax40 = fig.add_subplot(spec[4, 0])
    ax40.set_title('Onda Código 4')
    ax40.plot(y[3])
    ax41 = fig.add_subplot(spec[4, 1])
    ax41.set_title('Correlação Sinal com Onda 4')
    ax41.plot(correlate(v, y[3]))

    fig.set_size_inches(12, 6)
    plt.savefig('test.png', dpi=fig.dpi)
    # scipy.io.savemat("excitations_hadamard_"+str(n)+"_bitskk.mat", mdict={"excitations_hadamard_"+str(n)+"_bits": y})
