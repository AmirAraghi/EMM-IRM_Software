# EMM-IRM_Software
#1. Importing the Libraries
#1.1 Aquisition Libraries
import RPi.GPIO as GPIO
import board
import busio
import time
import adafruit_ads1x15.ads1115 as ADS
import numpy as np
from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn
#1.2 Processing Libraries
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks
from tkinter.filedialog import askopenfilename
#1.3 Regression Libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from tkinter.filedialog import askopenfilename
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import sympy as sp
from scipy.misc import derivative

#2. Input data

NC=4                     # Insert the the number of channels (range from 1 to 4)
d= 5                     # Insert the duration of each acquisition package [s]
di= 600                  # Insert the interval duration between each package (di [s])
D = 14                   # Insert the total duration of the trial (D [days])
phi_i= 16.75             # Insert the internal diameter of the tube (Øi [mm])
t= 3600                  # Insert the time that passed from t0 (t [s])
fc = 50                  # low cut-off frequency filter [Hz]
fc_High = 0              # high cut-off frequency filter [Hz]

Lt1= 400                  # Insert the total length of the beam 1 (Lt1 [mm])
Lt2= 400                  # Insert the total length of the beam 2(Lt2 [mm])
Lt3= 400                  # Insert the total length of the beam 3(Lt3 [mm])
Lt4= 400                  # Insert the total length of the beam 4(Lt4 [mm])

L1= 350                   # Insert the free span length of the beam 1 (L1 [mm])
L2= 350                   # Insert the free span length of the beam 2 (L2 [mm])
L3= 350                   # Insert the free span length of the beam 3 (L3 [mm])
L4= 350                   # Insert the free span length of the beam 4 (L4 [mm])

Mt1= 4.1                  # Insert the mass at the tip of the beam 1 (Mt1 [g])
Mt2= 4.1                  # Insert the mass at the tip of the beam 2 (Mt2 [g])
Mt3= 4.1                  # Insert the mass at the tip of the beam 3 (Mt3 [g])
Mt4= 4.1                  # Insert the mass at the tip of the beam 4 (Mt4 [g])


Mf1= 300                  # Insert the total mass of the full tube 1 (Mf1 [g])
Mf2= 300                  # Insert the total mass of the full tube 2 (Mf2 [g])
Mf3= 300                  # Insert the total mass of the full tube 3 (Mf3 [g])
Mf4= 300                  # Insert the total mass of the full tube 4 (Mf4 [g])

#2.1 Calculated data

M1= Mt1+ ((33/140)*(Mf1*L1)/Lt1) # Total equivalent mass at the free end of the full cantilever beam 1 [g]
M2= Mt2+ ((33/140)*(Mf2*L2)/Lt2) # Total equivalent mass at the free end of the full cantilever beam 1 [g]
M3= Mt3+ ((33/140)*(Mf3*L3)/Lt3) # Total equivalent mass at the free end of the full cantilever beam 1 [g]
M4= Mt4+ ((33/140)*(Mf4*L4)/Lt4) # Total equivalent mass at the free end of the full cantilever beam 1 [g]

pi=np.pi
I_cement= ((pi*(phi_i**4)) / 64)/(10**12)   # The moment of inertia of the beam cross-section [m^4]

#3. Acquisition and post-processing

from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn

i2c = busio.I2C (board.SCL, board.SDA)

ads1 = ADS.ADS1115(i2c, data_rate=860, mode=Mode.CONTINUOUS, address=0x48)
ads2 = ADS.ADS1115(i2c, data_rate=860, mode=Mode.CONTINUOUS, address=0x49)
ads3 = ADS.ADS1115(i2c, data_rate=860, mode=Mode.CONTINUOUS, address=0x4a)
ads4 = ADS.ADS1115(i2c, data_rate=860, mode=Mode.CONTINUOUS, address=0x4b)

canal1 = AnalogIn(ads1, ADS.P1) #EixoY
canal2 = AnalogIn(ads2, ADS.P1) #EixoY
canal3 = AnalogIn(ads3, ADS.P1) #EixoY
canal4 = AnalogIn(ads4, ADS.P1) #EixoY


lista1 = []
lista2 = []
lista3 = []
lista4 = []

i=1

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)



while (i <= (3600/di)*24*14):

    GPIO.output(4, 0)
    time.sleep(0.02)
    GPIO.output(4, 1)
    time.sleep(0.05)

    t_end = time.time() + d  # Sampling Duration

    while time.time() < t_end:
        lista1.append((canal1.voltage - 1.65) / 0.3)  # acceleration [g]
        time.sleep(0.0005)  # Sampling frequency

    filename = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("/home/pi/Desktop/EMM-ARM4_PLUS/Sample 1/" + filename + " - Captura " + str(i),
              "w") as file:  # insert the file address where you want to save the acquisition data
        file.write("This is file number " + str(i) + "\n")

        c1 = len(lista1)
        freq1 = c1 / d
        f1 = str(round(freq1, 1))
        file.write(str(c1) + " catches" "\n")
        file.write("Frequency = " + f1 + " Hz" "\n")
        file.write("Sampling duration = " + str(d) + 's \n')
        file.write(str(t_end) + "\n")

        for item in lista1:
            file.write("%s\n" % item)

#Post-processing

    list_EIcomposite=[]
    list_T = []
    list_f = []

    for file in os.listdir("Sample 1"):
        data = np.loadtxt(os.path.join("Sample 1", file), skiprows=5)
        time = np.loadtxt(os.path.join("Sample 1", file), skiprows=4)


        N = len(data)  # Number of data
        SAMPLE_RATE = N / d  # Sampling frequency [Hz] (LENGTH OF VECTOR)


        #Defining the time variables

        x = np.linspace(0, d, N, endpoint=False)
        y = data

        #Welch method (Frequency domain variables)

        f, S = signal.welch(y, SAMPLE_RATE, nperseg=500)

        S = S * 10 ** 8

        #Peak piking
        peaks = find_peaks(S, height=1, threshold=1, distance=1, prominence=1)
        height = peaks[1]['peak_heights']
        peaks_pos = f[peaks[0]]

        h = [i for i in height]
        p = [i for i in peaks_pos]

        #Identify the max

        fp_limits = [i for i in p if fc_High <= i <= fc]
        Sp_limits = [h[i] for i in range(p.index(fp_limits[0]), len(fp_limits) + p.index(fp_limits[0]))]

        max_S = max(Sp_limits)  # Find the maximum y value
        fn = fp_limits[Sp_limits.index(max_S)]  # Find the x value corresponding to the maximum y value

        #Calculate the Modulus of Elasticity

        EI_composite = ((2 * pi * fn) ** 2) * M1 * (L1 ** 3) / 3 / (10 ** 12)
        list_EIcomposite.append(EI_composite)
        list_T.append(time[0])
        list_f.append(fn)

        #Add time to the Modulus of Elasticity

    list_T = (list_T - list_T[0]+ t) / 3600
    list_E = (list_EIcomposite - list_EIcomposite[0] )/ I_cement / (10 ** 9) #[GPa]

    #Salvar os resultados em um arquivo txt

    with open("Results1.txt", "w") as file:
        file.write("E [Gpa]     T [h]     fn [Hz]\n")

        for index in range(len(list_E)):
            file.write(str(list_E[index]) + "   " + str(list_T[index]) + "  " + str(list_f[index]) + "\n")
        file.close()


#4. Regression and plotting

    with open('Results1.txt', 'r') as file:

        x = np.loadtxt(file, dtype=float, skiprows=1)[:, 0]
        y = np.loadtxt(file, dtype=float, skiprows=1)[:, 1]


    # Equation

    def E(t, a1, T1, B1, a2, T2, B2, a3):
        return a1 * np.exp(-(T1 / t) ** B1) + a2 * np.exp(-(T2 / t) ** B2) + a3


    # guess
    g = [10.8, 18.32, 1.67, 277, 3.8526e+15, 0.05, -0.84]

    n = len(x)
    e = np.empty(n)

    # Fit

    c, cov = curve_fit(E, x, y, g, maxfev=200000)
    for i in range(n):
        e[i] = E(x[i], c[0], c[1], c[2], c[3], c[4], c[5], c[6])
    R2 = r2_score(e, y)

    t = np.linspace(0, 500, 1000)
    # t=[24,72,168,336]

    M = E(t, c[0], c[1], c[2], c[3], c[4], c[5], c[6])



    # Plot

    ## Axes configuration
    axes = plt.subplot(1, 1, 1)

    ### Limits
    # axes.axis([0, 500, 0, 13])
    plt.margins(x=0, y=0)

    ### major ticks
    axes.xaxis.set_major_locator(MultipleLocator(1))
    axes.yaxis.set_major_locator(MultipleLocator(1))

    ### minor ticks
    # axes.xaxis.set_minor_locator(MultipleLocator(10))
    # axes.yaxis.set_minor_locator(MultipleLocator(0.2))

    ### major grids
    axes.grid(which="major", axis='x', linewidth=1, linestyle=':', color='0.75', alpha=0.5)
    axes.grid(which="major", axis='y', linewidth=1, linestyle=':', color='0.75', alpha=0.5)

    ### major grids
    # axes.grid(which="minor", axis='x', linewidth=1, linestyle='-', color='green', alpha=0.5)
    # axes.grid(which="minor", axis='y', linewidth=1, linestyle='-', color='red', alpha=0.5)

    ### Titles and Lables
    plt.xscale("log")
    plt.ylabel('Modulus of elasticity [GPa]', size=14)
    plt.xlabel('Time [h]', size=14)

    ## Graphs

    plt.scatter(x, y, s=3, c="slateblue", marker="o", label="Sample1", cmap=None,
                vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors="slateblue")

    plt.plot(t, M, "black", linewidth=2, linestyle='--', label="Sample1_Regression, "
                                                               #                          " a1="+ str(c[0])+'\n' 
                                                               #                          " T1="+ str(c[1])+'\n'  
                                                               #                         " B1="+ str(c[2])+'\n'
                                                               #                         " a2=" + str(c[3])+'\n' 
                                                               #                         " T2=" + str(c[4])+'\n'
                                                               #                         " B2=" + str(c[5])+'\n'
                                                               #                          " a3=" + str(c[6])+'\n'
                                                               "R" + "\u00b21=" + str(round(R2, 3)) + '\n')
    # show
    plt.legend()
    plt.show()



    time.sleep(3)

 #_________________________________________________________________________________________
    if NC>1:
        GPIO.output(17, 0)
        time.sleep(0.02)
        GPIO.output(17, 1)

        t_end = time.time() + d

        while time.time() < t_end:
            lista2.append((canal2.voltage - 1.65) / 0.3)
            time.sleep(0.0005)

        filename = time.strftime("%Y-%m-%d-%H-%M-%S")
        with open("/home/pi/Desktop/EMM-ARM4_PLUS/Sample 2/" + filename + " - Captura " + str(i),
                  "w") as file:  # inserir nome do arquivo no primeiro argumento
            file.write("This is file number " + str(i) + "\n")

            c2 = len(lista2)
            freq2 = c2 / d
            f2 = str(round(freq2, 1))
            file.write(str(c2) + " catches" "\n")
            file.write("Frequency = " + f2 + " Hz" "\n")
            file.write("Sampling duration = " + str(d) + 's \n')
            file.write(str(t_end) + "\n")

            for item in lista2:
                file.write("%s\n" % item)

    # Post-processing

        list_EIcomposite = []
        list_T = []
        list_f = []

        for file in os.listdir("Sample 2"):
            data = np.loadtxt(os.path.join("Sample 2", file), skiprows=5)
            time = np.loadtxt(os.path.join("Sample 2", file), skiprows=4)

            N = len(data)  # Number of data
            SAMPLE_RATE = N / d  # Sampling frequency [Hz] (LENGTH OF VECTOR)

            # Defining the time variables

            x = np.linspace(0, d, N, endpoint=False)
            y = data

            # Welch method (Frequency domain variables)

            f, S = signal.welch(y, SAMPLE_RATE, nperseg=500)

            S = S * 10 ** 8

            # Peak piking
            peaks = find_peaks(S, height=1, threshold=1, distance=1, prominence=1)
            height = peaks[1]['peak_heights']
            peaks_pos = f[peaks[0]]

            h = [i for i in height]
            p = [i for i in peaks_pos]

            # Identify the max

            fp_limits = [i for i in p if fc_High <= i <= fc]
            Sp_limits = [h[i] for i in range(p.index(fp_limits[0]), len(fp_limits) + p.index(fp_limits[0]))]

            max_S = max(Sp_limits)  # Find the maximum y value
            fn = fp_limits[Sp_limits.index(max_S)]  # Find the x value corresponding to the maximum y value

            # Calculate the Modulus of Elasticity

            EI_composite = ((2 * pi * fn) ** 2) * M2 * (L2 ** 3) / 3 / (10 ** 12)
            list_EIcomposite.append(EI_composite)
            list_T.append(time[0])
            list_f.append(fn)

            # Add time to the Modulus of Elasticity

        list_T = (list_T - list_T[0] + t) / 3600
        list_E = (list_EIcomposite - list_EIcomposite[0]) / I_cement / (10 ** 9)  # [GPa]

        # Salvar os resultados em um arquivo txt

        with open("Results2.txt", "w") as file:
            file.write("E [Gpa]     T [h]     fn [Hz]\n")

            for index in range(len(list_E)):
                file.write(str(list_E[index]) + "   " + str(list_T[index]) + "  " + str(list_f[index]) + "\n")
            file.close()

    # 4. Regression and plotting

        with open('Results2.txt', 'r') as file:

            x = np.loadtxt(file, dtype=float, skiprows=1)[:, 0]
            y = np.loadtxt(file, dtype=float, skiprows=1)[:, 1]


        # Equation

        def E(t, a1, T1, B1, a2, T2, B2, a3):
            return a1 * np.exp(-(T1 / t) ** B1) + a2 * np.exp(-(T2 / t) ** B2) + a3


        # guess
        g = [10.8, 18.32, 1.67, 277, 3.8526e+15, 0.05, -0.84]

        n = len(x)
        e = np.empty(n)

        # Fit

        c, cov = curve_fit(E, x, y, g, maxfev=200000)
        for i in range(n):
            e[i] = E(x[i], c[0], c[1], c[2], c[3], c[4], c[5], c[6])
        R2 = r2_score(e, y)

        t = np.linspace(0, 500, 1000)
        # t=[24,72,168,336]

        M = E(t, c[0], c[1], c[2], c[3], c[4], c[5], c[6])

        # Plot

        ## Axes configuration
        axes = plt.subplot(1, 1, 1)

        ### Limits
        # axes.axis([0, 500, 0, 13])
        plt.margins(x=0, y=0)

        ### major ticks
        axes.xaxis.set_major_locator(MultipleLocator(1))
        axes.yaxis.set_major_locator(MultipleLocator(1))

        ### minor ticks
        # axes.xaxis.set_minor_locator(MultipleLocator(10))
        # axes.yaxis.set_minor_locator(MultipleLocator(0.2))

        ### major grids
        axes.grid(which="major", axis='x', linewidth=1, linestyle=':', color='0.75', alpha=0.5)
        axes.grid(which="major", axis='y', linewidth=1, linestyle=':', color='0.75', alpha=0.5)

        ### major grids
        # axes.grid(which="minor", axis='x', linewidth=1, linestyle='-', color='green', alpha=0.5)
        # axes.grid(which="minor", axis='y', linewidth=1, linestyle='-', color='red', alpha=0.5)

        ### Titles and Lables
        plt.xscale("log")
        plt.ylabel('Modulus of elasticity [GPa]', size=14)
        plt.xlabel('Time [h]', size=14)

        ## Graphs

        plt.scatter(x, y, s=3, c="slateblue", marker="o", label="Sample2", cmap=None,
                    vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors="slateblue")

        plt.plot(t, M, "black", linewidth=2, linestyle='--', label="Sample2_Regression, "
                                                                   #                          " a1="+ str(c[0])+'\n' 
                                                                   #                          " T1="+ str(c[1])+'\n'  
                                                                   #                         " B1="+ str(c[2])+'\n'
                                                                   #                         " a2=" + str(c[3])+'\n' 
                                                                   #                         " T2=" + str(c[4])+'\n'
                                                                   #                         " B2=" + str(c[5])+'\n'
                                                                   #                          " a3=" + str(c[6])+'\n'
                                                                   "R" + "\u00b21=" + str(round(R2, 3)) + '\n')
        # show
        plt.legend()
        plt.show()

        time.sleep(3)
    # ________________________________________________________________________________________________
    if NC>2:
        GPIO.output(24, 0)
        time.sleep(0.03)
        GPIO.output(24, 1)
        time.sleep(0.02)

        t_end = time.time() + d
        while time.time() < t_end:
            lista3.append((canal3.voltage - 1.65) / 0.3)
            time.sleep(0.0005)

        filename = time.strftime("%Y-%m-%d-%H-%M-%S")
        with open("/home/pi/Desktop/EMM-ARM4_PLUS/Sample 3/" + filename + " - Captura " + str(i),
                  "w") as file:
            file.write("This is file number " + str(i) + "\n")

            c3 = len(lista3)
            freq3 = c3 / d
            f3 = str(round(freq3, 1))
            file.write(str(c3) + " catches" "\n")
            file.write("Frequency = " + f3 + " Hz" "\n")
            file.write("Sampling duration = " + str(d) + 's \n')
            file.write(str(t_end) + "\n")

            for item in lista3:
                file.write("%s\n" % item)

    # Post-processing

        list_EIcomposite = []
        list_T = []
        list_f = []

        for file in os.listdir("Sample 3"):
            data = np.loadtxt(os.path.join("Sample 3", file), skiprows=5)
            time = np.loadtxt(os.path.join("Sample 3", file), skiprows=4)

            N = len(data)  # Number of data
            SAMPLE_RATE = N / d  # Sampling frequency [Hz] (LENGTH OF VECTOR)

            # Defining the time variables

            x = np.linspace(0, d, N, endpoint=False)
            y = data

            # Welch method (Frequency domain variables)

            f, S = signal.welch(y, SAMPLE_RATE, nperseg=500)

            S = S * 10 ** 8

            # Peak piking
            peaks = find_peaks(S, height=1, threshold=1, distance=1, prominence=1)
            height = peaks[1]['peak_heights']
            peaks_pos = f[peaks[0]]

            h = [i for i in height]
            p = [i for i in peaks_pos]

            # Identify the max

            fp_limits = [i for i in p if fc_High <= i <= fc]
            Sp_limits = [h[i] for i in range(p.index(fp_limits[0]), len(fp_limits) + p.index(fp_limits[0]))]

            max_S = max(Sp_limits)  # Find the maximum y value
            fn = fp_limits[Sp_limits.index(max_S)]  # Find the x value corresponding to the maximum y value

            # Calculate the Modulus of Elasticity

            EI_composite = ((2 * pi * fn) ** 2) * M3 * (L3 ** 3) / 3 / (10 ** 12)
            list_EIcomposite.append(EI_composite)
            list_T.append(time[0])
            list_f.append(fn)

            # Add time to the Modulus of Elasticity

        list_T = (list_T - list_T[0] + t) / 3600
        list_E = (list_EIcomposite - list_EIcomposite[0]) / I_cement / (10 ** 9)  # [GPa]

        # Salvar os resultados em um arquivo txt

        with open("Results3.txt", "w") as file:
            file.write("E [Gpa]     T [h]     fn [Hz]\n")

            for index in range(len(list_E)):
                file.write(str(list_E[index]) + "   " + str(list_T[index]) + "  " + str(list_f[index]) + "\n")
            file.close()

    # 4. Regression and plotting

        with open('Results3.txt', 'r') as file:

            x = np.loadtxt(file, dtype=float, skiprows=1)[:, 0]
            y = np.loadtxt(file, dtype=float, skiprows=1)[:, 1]


        # Equation

        def E(t, a1, T1, B1, a2, T2, B2, a3):
            return a1 * np.exp(-(T1 / t) ** B1) + a2 * np.exp(-(T2 / t) ** B2) + a3


        # guess
        g = [10.8, 18.32, 1.67, 277, 3.8526e+15, 0.05, -0.84]

        n = len(x)
        e = np.empty(n)

        # Fit

        c, cov = curve_fit(E, x, y, g, maxfev=200000)
        for i in range(n):
            e[i] = E(x[i], c[0], c[1], c[2], c[3], c[4], c[5], c[6])
        R2 = r2_score(e, y)

        t = np.linspace(0, 500, 1000)
        # t=[24,72,168,336]

        M = E(t, c[0], c[1], c[2], c[3], c[4], c[5], c[6])

        # Plot

        ## Axes configuration
        axes = plt.subplot(1, 1, 1)

        ### Limits
        # axes.axis([0, 500, 0, 13])
        plt.margins(x=0, y=0)

        ### major ticks
        axes.xaxis.set_major_locator(MultipleLocator(1))
        axes.yaxis.set_major_locator(MultipleLocator(1))

        ### minor ticks
        # axes.xaxis.set_minor_locator(MultipleLocator(10))
        # axes.yaxis.set_minor_locator(MultipleLocator(0.2))

        ### major grids
        axes.grid(which="major", axis='x', linewidth=1, linestyle=':', color='0.75', alpha=0.5)
        axes.grid(which="major", axis='y', linewidth=1, linestyle=':', color='0.75', alpha=0.5)

        ### major grids
        # axes.grid(which="minor", axis='x', linewidth=1, linestyle='-', color='green', alpha=0.5)
        # axes.grid(which="minor", axis='y', linewidth=1, linestyle='-', color='red', alpha=0.5)

        ### Titles and Lables
        plt.xscale("log")
        plt.ylabel('Modulus of elasticity [GPa]', size=14)
        plt.xlabel('Time [h]', size=14)

        ## Graphs

        plt.scatter(x, y, s=3, c="slateblue", marker="o", label="Sample3", cmap=None,
                    vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors="slateblue")

        plt.plot(t, M, "black", linewidth=2, linestyle='--', label="Sample3_Regression, "
                                                                   #                          " a1="+ str(c[0])+'\n' 
                                                                   #                          " T1="+ str(c[1])+'\n'  
                                                                   #                         " B1="+ str(c[2])+'\n'
                                                                   #                         " a2=" + str(c[3])+'\n' 
                                                                   #                         " T2=" + str(c[4])+'\n'
                                                                   #                         " B2=" + str(c[5])+'\n'
                                                                   #                          " a3=" + str(c[6])+'\n'
                                                                   "R" + "\u00b21=" + str(round(R2, 3)) + '\n')
        # show
        plt.legend()
        plt.show()

        time.sleep(3)
    # ____________________________________________________________________________________________________________
    if NC>3:
        GPIO.output(22, 0)
        time.sleep(0.03)
        GPIO.output(22, 1)
        time.sleep(0.05)

        t_end = time.time() + d
        while time.time() < t_end:
            lista4.append((canal4.voltage - 1.65) / 0.3)
            time.sleep(0.0005)

        filename = time.strftime("%Y-%m-%d-%H-%M-%S")
        with open("/home/pi/Desktop/EMM-ARM4_PLUS/Sample 4/" + filename + " - Captura " + str(i),
                  "w") as file:
            file.write("This is file number " + str(i) + "\n")

            c4 = len(lista4)
            freq4 = c4 / d
            f4 = str(round(freq4, 1))
            file.write(str(c4) + " catches" "\n")
            file.write("Frequency = " + f4 + " Hz" "\n")
            file.write("Sampling duration = " + str(d) + 's \n')
            file.write(str(t_end) + "\n")

            for item in lista4:
                file.write("%s\n" % item)

    # Post-processing

        list_EIcomposite = []
        list_T = []
        list_f = []

        for file in os.listdir("Sample 2"):
            data = np.loadtxt(os.path.join("Sample 2", file), skiprows=5)
            time = np.loadtxt(os.path.join("Sample 2", file), skiprows=4)

            N = len(data)  # Number of data
            SAMPLE_RATE = N / d  # Sampling frequency [Hz] (LENGTH OF VECTOR)

            # Defining the time variables

            x = np.linspace(0, d, N, endpoint=False)
            y = data

            # Welch method (Frequency domain variables)

            f, S = signal.welch(y, SAMPLE_RATE, nperseg=500)

            S = S * 10 ** 8

            # Peak piking
            peaks = find_peaks(S, height=1, threshold=1, distance=1, prominence=1)
            height = peaks[1]['peak_heights']
            peaks_pos = f[peaks[0]]

            h = [i for i in height]
            p = [i for i in peaks_pos]

            # Identify the max

            fp_limits = [i for i in p if fc_High <= i <= fc]
            Sp_limits = [h[i] for i in range(p.index(fp_limits[0]), len(fp_limits) + p.index(fp_limits[0]))]

            max_S = max(Sp_limits)  # Find the maximum y value
            fn = fp_limits[Sp_limits.index(max_S)]  # Find the x value corresponding to the maximum y value

            # Calculate the Modulus of Elasticity

            EI_composite = ((2 * pi * fn) ** 2) * M2 * (L2 ** 3) / 3 / (10 ** 12)
            list_EIcomposite.append(EI_composite)
            list_T.append(time[0])
            list_f.append(fn)

            # Add time to the Modulus of Elasticity

        list_T = (list_T - list_T[0] + t) / 3600
        list_E = (list_EIcomposite - list_EIcomposite[0]) / I_cement / (10 ** 9)  # [GPa]

        # Salvar os resultados em um arquivo txt

        with open("Results2.txt", "w") as file:
            file.write("E [Gpa]     T [h]     fn [Hz]\n")

            for index in range(len(list_E)):
                file.write(str(list_E[index]) + "   " + str(list_T[index]) + "  " + str(list_f[index]) + "\n")
            file.close()

    # 4. Regression and plotting

        with open('Results4.txt', 'r') as file:

            x = np.loadtxt(file, dtype=float, skiprows=1)[:, 0]
            y = np.loadtxt(file, dtype=float, skiprows=1)[:, 1]


        # Equation

        def E(t, a1, T1, B1, a2, T2, B2, a3):
            return a1 * np.exp(-(T1 / t) ** B1) + a2 * np.exp(-(T2 / t) ** B2) + a3


        # guess
        g = [10.8, 18.32, 1.67, 277, 3.8526e+15, 0.05, -0.84]

        n = len(x)
        e = np.empty(n)

        # Fit

        c, cov = curve_fit(E, x, y, g, maxfev=200000)
        for i in range(n):
            e[i] = E(x[i], c[0], c[1], c[2], c[3], c[4], c[5], c[6])
        R2 = r2_score(e, y)

        t = np.linspace(0, 500, 1000)
        # t=[24,72,168,336]

        M = E(t, c[0], c[1], c[2], c[3], c[4], c[5], c[6])

        # Plot

        ## Axes configuration
        axes = plt.subplot(1, 1, 1)

        ### Limits
        # axes.axis([0, 500, 0, 13])
        plt.margins(x=0, y=0)

        ### major ticks
        axes.xaxis.set_major_locator(MultipleLocator(1))
        axes.yaxis.set_major_locator(MultipleLocator(1))

        ### minor ticks
        # axes.xaxis.set_minor_locator(MultipleLocator(10))
        # axes.yaxis.set_minor_locator(MultipleLocator(0.2))

        ### major grids
        axes.grid(which="major", axis='x', linewidth=1, linestyle=':', color='0.75', alpha=0.5)
        axes.grid(which="major", axis='y', linewidth=1, linestyle=':', color='0.75', alpha=0.5)

        ### major grids
        # axes.grid(which="minor", axis='x', linewidth=1, linestyle='-', color='green', alpha=0.5)
        # axes.grid(which="minor", axis='y', linewidth=1, linestyle='-', color='red', alpha=0.5)

        ### Titles and Lables
        plt.xscale("log")
        plt.ylabel('Modulus of elasticity [GPa]', size=14)
        plt.xlabel('Time [h]', size=14)

        ## Graphs

        plt.scatter(x, y, s=3, c="slateblue", marker="o", label="Sample4", cmap=None,
                    vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors="slateblue")

        plt.plot(t, M, "black", linewidth=2, linestyle='--', label="Sample4_Regression, "
                                                                   #                          " a1="+ str(c[0])+'\n' 
                                                                   #                          " T1="+ str(c[1])+'\n'  
                                                                   #                         " B1="+ str(c[2])+'\n'
                                                                   #                         " a2=" + str(c[3])+'\n' 
                                                                   #                         " T2=" + str(c[4])+'\n'
                                                                   #                         " B2=" + str(c[5])+'\n'
                                                                   #                          " a3=" + str(c[6])+'\n'
                                                                   "R" + "\u00b21=" + str(round(R2, 3)) + '\n')
        # show
        plt.legend()
        plt.show()

    print("Please wait for the next section")

    lista1 = []
    lista2 = []
    lista3 = []
    lista4 = []

    i = i + 1

    time.sleep(di)  # Gap duration
