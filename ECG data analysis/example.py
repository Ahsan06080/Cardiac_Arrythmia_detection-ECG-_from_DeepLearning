import bwr
import matplotlib.pyplot as plt
import numpy as np

# Read input csv file from physionet
f = open("samples/1.csv", "r")
lines = f.readlines()
f.close()

# Discard the first two lines because of header. Takes either column 1 or 2 from each lines (different signal lead)
signal = np.zeros((len(lines) - 2))
for i in range(len(signal)):
    signal[i] = float(lines[i + 2].split(",")[1])

baseline = bwr.calc_baseline(signal)

# Remove baseline from orgianl signal
ecg_out = signal - baseline

plt.subplot(2, 1, 1)
plt.plot(signal, "b-", label="signal")
plt.plot(baseline, "r-", label="baseline")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(ecg_out, "b-", label="signal - baseline")
plt.legend()
plt.show()
