# -------SUBSONIC

    # import csv from digitiser
    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/NACA0012M0.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        xc = np.array([float(item[0]) for item in data[1:]])*0.99
        cpuc = np.array([float(item[1]) for item in data[1:]])
        cpul = np.array([float(item[2]) for item in data[1:]])

    # import csv from digitiser
    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/NACA0012M0expl.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        xpl = np.array([float(item[0]) for item in data[1:]])*0.99
        expl = np.array([float(item[2]) for item in data[1:]])

    # import csv from digitiser
    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/NACA0012M0expu.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        xpu = np.array([float(item[0]) for item in data[1:]])*0.99
        expu = np.array([float(item[1]) for item in data[1:]])

    ax.scatter(xpl, expl, s= 8, label= "Experiment (Harris, 1981)", Color = "r")
    ax.scatter(xpu, expu, s= 8, label= "", Color = "r")
    ax.plot(xc, cpuc, lw=1, label= "M.K. Singh et al", Color = "b")
    ax.plot(xc, cpul, lw=1, label= "", Color = "b", linestyle = ":")

    ax.text(0.3, -0.8, 'NACA 0012, M=0.65, α=1.86°\n CFL 20, 1E+05 iterations',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})


    # -------TRANSONIC

       # import csv from digitiser
    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/transoniccpl.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        xc = np.array([float(item[0]) for item in data[1:]])
        cpl = np.array([float(item[2]) for item in data[1:]])

    # import csv from digitiser
    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/transoniccpu.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        xc2 = np.array([float(item[0]) for item in data[1:]])
        cpu = np.array([float(item[1]) for item in data[1:]])

    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/JK_trans_pls_flo82.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        x3 = np.array([float(item[0]) for item in data[1:]])
        cpl3 = np.array([float(item[2]) for item in data[1:]])
        cpu3 = np.array([float(item[1]) for item in data[1:]])

    ax.plot(xc2, -cpu, lw=1, label= "S upper", Color = "b")
    ax.plot(xc , -cpl, lw=1, label= "S lower", Color = "b", linestyle = "--")
    ax.plot(x3 , cpu3, lw=1, label= "JK upper", Color = "r")
    ax.plot(x3 , cpl3, lw=1, label= "JK lower", Color = "r", linestyle = ":")
    ax.text(0.6, 1.25, 'NACA 0012, M=0.85, α=1.00°\n CFL 0.001, 2E+06 iterations',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})


    # -------SUPERSONIC

        # import csv from digitiser
    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/NACA0012M1.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        xc = np.array([float(item[0]) for item in data[1:]])
        cpuc = np.array([float(item[1]) for item in data[1:]])
        cpul = np.array([float(item[2]) for item in data[1:]])

    # import csv from digitiser
    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/NACA0012M1expl.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        xpl = np.array([float(item[0]) for item in data[1:]])
        expl = np.array([float(item[2]) for item in data[1:]])

    # import csv from digitiser
    with open("/home/rinaldo/Desktop/BEng_Proj/Digitising/NACA0012M1expu.csv") as f:
        data = list(csv.reader(f, delimiter=","))
        xpu = np.array([float(item[0]) for item in data[1:]])
        expu = np.array([float(item[1]) for item in data[1:]])

    ax.scatter(xpu, expu, s= 8, label= "AGARD upper", Color = "r")
    ax.scatter(xpl, expl, s= 8, label= "AGARD lower", Color = "r", linestyle = "--")
    ax.plot(xc ,cpuc, lw=1, label= "S upper", Color = "k", linestyle = "--")
    ax.plot(xc, cpul, lw=1, label= "S lower", Color = "k", linestyle = "--")
    ax.text(0.6, 1.25, 'NACA 0012, M=0.85, α=1.00°\n CFL 0.001, 2E+06 iterations',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})