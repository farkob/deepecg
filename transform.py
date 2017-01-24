import numpy as np
import linecache
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
import pywt

def from_bytes (data, big_endian = False):
    if isinstance(data, str):
        data = bytearray(data)
    if big_endian:
        data = reversed(data)
    num = 0
    for offset, byte in enumerate(data):
        num += byte << (offset * 8)
    return num

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class Transformer:

    def __init__(self, prefix):
        self.prefix = prefix

    def get_data(self, patientPath, lead = None):
        headerFile = open(self.prefix + patientPath + ".hea","r")
        datFile = open(self.prefix + patientPath + ".dat","rb")

        signalLength = int(headerFile.readline().split()[3])

        # Loop over all 12 measurements from dat file
        data = np.zeros((12,signalLength))
        for sampleIdx in range(signalLength):
            # Read 12 signal from dat file
            for varIdx in range(12):
                myBytes = datFile.read(2)
                data[varIdx, sampleIdx] = int.from_bytes(myBytes, byteorder='little', signed=True)/2000.

        # Close ressources
        headerFile.close()
        datFile.close()

        return data

    def get_diagnosis(self, patientPath):
        """
        Get the patient's diagnosis from the header file.
        """
        line = linecache.getline(self.prefix + patientPath + ".hea", 23)
        diagnosis = ' '.join(line.split()[4:])
        return diagnosis


    def prepare(self, data, n):
        approx = []
        detail1 = []
        detail2 = []
        for datum in data:
            filtered = butter_bandpass_filter(datum[n*4000:(n+1)*4000], 1, 49.2, 1000)
            a4, d4, d3, d2, d1 = pywt.wavedec(filtered, 'sym7', level=4)
            approx.append(a4)
            detail1.append(d4)
            detail2.append(d3)

        return approx, detail1, detail2

if __name__ == "__main__":
    db = Transformer('/home/farkob/ptbdb/')
    data = db.get_data('patient004/s0020are')
    plt.subplot(4, 1, 1)
    plt.plot(data[0,:4000])
    #plt.pcolormesh(t, f, Sxx*1000)
    #plt.colorbar()

    filtered = butter_bandpass_filter(data[2,:4000], 1, 49.2, 1000)
    a, b, c, d, e = pywt.wavedec(filtered, 'sym7', level=4)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    print(e.shape)
    plt.subplot(4,1,2)
    plt.plot(filtered)
    plt.subplot(4,1,3)
    plt.plot(a)
    plt.subplot(4,1,4)
    plt.plot(b)

    plt.show()
