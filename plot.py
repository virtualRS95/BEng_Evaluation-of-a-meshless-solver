"""
This Script takes points and results from the University of Glasgow meshless 
solver. It then processes those to be output as plots and data values.

Requires following external modules:
matplotlib, pandas, numpy, scipy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math
import csv
import os
import numpy as np
import scipy.integrate as integ
import scipy.spatial as ssp
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
from matplotlib import ticker
from matplotlib.patches import Polygon
from operator import itemgetter


global gflowdataframe # to allow the transfer of point dataframes between functions
global gsurfdataframe # to allow the transfer of surface dataframes between functions
global gdataframe # to allow the transfer of other dataframes between functions
global gamma

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# -------------------------- VARIABLES FOR USER INPUT --------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
"""
Comment out lines which should not be used
## marked comments are non-functional comments
"""
## --- Testcase: ---

flowmodel = 0 # choose number: 0 (Euler), 1 (Laminar) or 2 (RANS)
three_dim = 0 # False: 2D, True 3D analyser
Linux = 1
gamma = 1.401

if Linux:
    if three_dim:
        #resultsdir = "/home/rinaldo/Desktop/TC2/TestCase2_A0_M0.85/Goland"
        #resultsdir = "/home/rinaldo/Desktop/TC2/M0.1_A0/Goland"
        resultsdir = "/home/rinaldo/Desktop/TC2/M0.1_A5_2/Goland"
    else:
        #resultsdir = "/home/rinaldo/Desktop/TC1/M0.2_10/test"
        #resultsdir = "/home/rinaldo/Desktop/TC1/transonic_katz/test"
        resultsdir = "/home/rinaldo/Desktop/TC1/subsonic/test"
        #resultsdir = "/home/rinaldo/Desktop/TC1/transonic/test"
else:
    if three_dim:
        resultsdir = "/home/rinaldo/Desktop/TC2/TestCase2_A0_M0.85/Goland"
    else:
        #resultsdir = "/home/rinaldo/Desktop/TC1/TestCase1_Katz/test"
        resultsdir = "D:\\Beng_Proj Linux\\TC1_archive\\TestCase1_subsonic\\test"

# debug options
viscid = False  # True of False depending on TestCase setting, Flow model 0 -> False
RANS = False # if above is True, specify if RANS (2) or Euler (1), OTHERWISE SET AS FALSE!

## --- 2D Testcase: ---


## --- 3D Testcase: ---
slices = [0.05, 0.115, 0.20, 0.307, 0.44, 0.602, 0.794, 1.011, 1.25, 1.5, 1.75,
            1.995, 2.225, 2.434, 2.617, 2.902, 3.008, 3.092, 3.158, 3.21, 3.25]
#slices2 = [0.05, 0.115, 0.20, 0.307, 0.44, 0.602, 0.794, 1.011, 1.25, 1.5, 1.75,
#            1.995, 2.225, 2.434, 2.617, 2.902, 3.008, 3.092, 3.158, 3.21, 3.25,
#            3.27, 3.29, 3.31, 3.32, 3.33, 3.34, 3.35]
slices2 = [0.05, 0.115, 0.20, 0.307, 0.44, 0.602, 0.794, 1.011, 1.25, 1.5, 1.75,
            1.995, 2.225, 2.434, 2.617, 2.902, 3.008, 3.092, 3.158, 3.21, 3.25, 3.31,
            3.32, 3.33, 3.34, 3.35]
            # at which points the program searches for 3D surface file slices
wingspan = 6.7 #*c

## --- 2D Processing options: ---
#plot_grid = True
#plot_cpx = True

## --- 3D Processing options: ---
#calc_coefs = True


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ---------------- BEGINNING OF NON-PLOTTING FUNCTION DEFINITIONS --------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def getFlowinfo(resultsdir,surface=0, residual=0, RANS=0):
    """
    Reads the tecplot files of the surface or general point clouds and 
    returns the values inside the file as lists of all points and a dataframe containing all

    surface: open surface file and only extract BC GROUP 30 (solid wing surface)
    residual: 
    """
    
    global gflowdataframe # to allow the transfer of point dataframes between functions
    global gsurfdataframe # to allow the transfer of surface dataframes between functions

    x = []
    y = []
    z = []
    r = []
    u = []
    v = []
    w = []
    p = []
    cp = []     # coeff of pressure (p_env-p_local/q), higher local pressure = negative cp
    cf = []     # coeff of friction (surface only)
    nut = []    # not used yet
    flag = False    # found BC 30 (solid surface)
    flag2 = False   # executing BC 30 script
    count = 0   # line count
    
    # finding correct file
    if residual:
        inpath = resultsdir + ".residual.tec"   # 4 header lines
    elif surface:
        inpath = resultsdir + ".surface.tec"    # 5 header lines
    else:
        inpath = resultsdir + ".tec"            # 4 header lines 

    # opening correct file and read all lines
    with open(inpath) as file:
        data = file.readlines()

    # ignoring unnecessary header files
    if RANS: 
        del data[0:5] #delete header lines, RANS has 1 more variable, so longer header
    else:
        del data[0:4] #delete header lines

    # data extraction for surface
    if surface:
        for item in data:
            count += 1  # count lines
            
            if "bc group 30" in item:
                #print("found")
                #print(count)
                flag = True
            #print(item)
            
            if flag2:
                #print(data)
                #print(item)
                if RANS:
                    try:
                        a, b, c, d, e, f, g, h, i, j, k = item.split()
                    except(ValueError):
                        #print("SURFACE algorithm terminated before file end")
                        print("length of array: "+ str(len(x)))
                        gsurfdataframe = pd.DataFrame({"X":x, "Y":y, "Z":z, "R":r, "U":u,
                         "V":v, "W":w, "P":p, "CP":cp, "CF":cf, "NUT":nut})
                        #print(gsurfdataframe)
                        return x, y, z, r, u, v, w, p, cp, cf, nut
                else:    
                    try:
                        a, b, c, d, e, f, g, h, i, j = item.split()
                    except(ValueError):
                        #print("SURFACE algorithm terminated before file end")
                        print("length of array: "+ str(len(x)))
                        gsurfdataframe = pd.DataFrame({"X":x, "Y":y, "Z":z, "R":r, "U":u,
                         "V":v, "W":w, "P":p, "CP":cp, "CF":cf})
                        #print(gsurfdataframe)
                        return x, y, z, r, u, v, w, p, cp, cf
                if True:
                    x.append(float(a))
                    y.append(float(b))
                    z.append(float(c))
                    r.append(float(d))
                    u.append(float(e))
                    v.append(float(f))
                    w.append(float(g))
                    p.append(float(h))
                    cp.append(float(i))
                    cf.append(float(j))
                    if RANS:
                        nut.append(float(k))
                flag = False
            
            # execute code from next line iteration after deleting headers
            if flag:
                flag2 = True    
                #print(item)
                #print(data)
                del data[0:2]
                
        else:
            print("Flag False, file does not contain BC GROUP 30")
            return()
    
    # non surface data extraction (free flow or residuals)
    else:
        for item in data:
            if RANS:
                try:
                    a, b, c, d, e, f, g, h, i = item.split()
                except(ValueError):
                    #print(item)
                    print("length of array: "+ str(len(x)))
                    gflowdataframe = pd.DataFrame({"X":x, "Y":y, "Z":z, "R":r, "U":u,
                         "V":v, "W":w, "P":p, "NUT":nut})
                    #print(gflowdataframe)
                    return x, y, z, r, u, v, w, p, nut
            else:
                try:
                    a, b, c, d, e, f, g, h = item.split()
                except(ValueError):
                    #print(item)
                    print("length of array: "+ str(len(x)))
                    gflowdataframe = pd.DataFrame({"X":x, "Y":y, "Z":z, "R":r, "U":u,
                         "V":v, "W":w, "P":p})
                    #print(gflowdataframe)
                    return x, y, z, r, u, v, w, p
            if True:
                x.append(float(a))
                y.append(float(b))
                z.append(float(c))
                r.append(float(d))
                u.append(float(e))
                v.append(float(f))
                w.append(float(g))
                p.append(float(h))
                if RANS:
                    nut.append(float(i))
        
        # if file ends without incompatible data (connectivity info)
        print("no connectivity data warning")
        gflowdataframe = pd.DataFrame({"X":x, "Y":y, "Z":z, "R":r, "U":u,
                                       "V":v, "W":w, "P":p})           
        return x, y, z, r, u, v, w, p


def getLoadsConvergence(resultsdir):
    """
    returns list of loads at iteration numbers
    """

    inpath = resultsdir + ".loads.convergence"
    itr, cdp, clp, cmp, cdv, clv, cmv = [], [], [], [], [], [], []
    
    with open(inpath) as file:
        data = file.readlines()
        del data[0:2] # delete header
        
        for line in data:
            a, b, c, d, e, f, g = line.split()
            itr.append(float(a))
            cdp.append(float(b))
            clp.append(float(c))
            cmp.append(float(d))
            cdv.append(float(e))
            clv.append(float(f))
            cmv.append(float(g))

    return itr, cdp, clp, cmp, cdv, clv, cmv


def getReport(resultsdir):
    """
    returns report lists from report file
    """
    itr = []
    cputime = []
    rmean = []
    rturb = []
    
    
    inpath = resultsdir + ".report"
    with open(inpath) as file:
        data = file.readlines()
        del data[0:2] #delete header lines
        for item in data:
            a, b, c, d = item.split()
            itr.append(float(a))
            cputime.append(float(b))
            rmean.append(float(c))
            rturb.append(float(d))

    return itr, cputime, rmean, rturb


def calcCoeffFromLoads(resultsdir):
    """
    takes .loads file and calculates load coefficients from pressure and
    viscous contribution
    returns t, cdp, clp, cmp, cdv, clv, cmv, cd, cl, cm
    """

    inpath = resultsdir + ".loads"

    with open(inpath) as file:
        data = file.readlines()
        t, cdp, clp, cmp, cdv, clv, cmv = data[2].split()
        cd = float(cdp) + float(cdv)
        cl = float(clp) + float(clv)
        cm = float(cmp) + float(cmv)
        
    return t, cdp, clp, cmp, cdv, clv, cmv, cd, cl, cm


def printCoeffFromLoads(resultsdir):
    """
    Prints load coefficients and solver runtime
    """

    t, cdp, clp, cmp, cdv, clv, cmv, cd, cl, cm = calcCoeffFromLoads(resultsdir)
    print("time: " + t)
    print("Coeff of Drag: " + str(float(cdp) + float(cdv)))
    print("Coeff of Lift: " + str(float(clp) + float(clv)))
    print("Coeff of Moment: " + str(float(cmp) + float(cmv)))
    return()


def checkGlobalDf(resultsdir):
    """
    orders to get the flow and surface dataframes if they do not exist yet
    """
    
    # import dataframe
    if "gflowdataframe" not in globals():
        getFlowinfo(resultsdir)
        
    if "gsurfdataframe" not in globals():
        getFlowinfo(resultsdir, surface=1)

    return()


def getYSlice(resultsdir, y=0, tolerance=0.01, surface=0):
    """
    returns Y-slice DataFrame of data between y-tolerance and y+tolerance
    sorted in circular fashion starting at trailing edge -> counter clockwise
    for 2D: y=0, all data
    recommended tolerance: 0.01

    for getXSlice() look into 3D-only functions
    """
    
    # import dataframe
    if "gsurfdataframe" not in globals():
        checkGlobalDf(resultsdir)

    # creating a copy
    if surface:
        dfy = gsurfdataframe.copy()
    else:
        dfy = gflowdataframe.copy()
        
    # choosing a slice
    mask = dfy["Y"] >= (y-abs(tolerance))
    dfy = dfy[mask]
    mask = dfy["Y"] <= (y+abs(tolerance))
    dfy = dfy[mask]
    
    # ordering dataframe in circular fashion, TE and LE at Z=0
    # only for surface points
    if surface:
        mask = dfy["Z"] > 0
        dfu = dfy[mask]     # upper surface
        dfl = dfy[~mask]    # lower surface

        dfu = dfu.sort_values("X", ascending=False)
        dfl = dfl.sort_values("X", ascending=True)
        points = dfu.append(dfl)
        points.reset_index(inplace=True) # new indexing in df
        del points['index']
    else:
        points = dfy.copy(deep = False)
    return points # returns dataframe of all surface points in slice


def orderCoord(x,z):
    """
    takes coordinates of a symmetric aerofoil and sorts them in circular
    fashion from trailing edge to trailing edge
    returns x, z sorted as numpy array
    """

    # using dataframe to sort
    df = pd.DataFrame({"X": x, "Z": z})

    mask = df["Z"] >= 0 #sorting above and below z=0 separately
    dfu = df[mask]
    dfl = df[~mask]
    dfu = dfu.sort_values("X", ascending=False)
    dfl = dfl.sort_values("X", ascending=True)
    df = dfu.append(dfl)
    x = df.T.to_numpy()[0] # conversion to array
    z = df.T.to_numpy()[1]

    return x, z


def plotLoadsConvergence(resultsdir):
    """
    plots a logarithmic plot of the loads over iteration number
    """

    # import loads convergence data
    itr, cdp, clp, cmp, cdv, clv, cmv = getLoadsConvergence(resultsdir)

    # plot data
    fig, ax = plt.subplots(figsize=(4,2.8))
    if viscid:
        plt.plot(itr, cdp, label="Coeff. of Drag (p)", marker="x", markersize=7)
        plt.plot(itr, clp, label="Coeff. of Lift (p)", marker="x", markersize=7)
        plt.plot(itr, cmp, label="Coeff. of Moment (p)", marker="x", markersize=7)
        plt.plot(itr, cdv, label="Coeff. of Drag (v)", marker="o", markersize=7)
        plt.plot(itr, clv, label="Coeff. of Lift (v)", marker="o", markersize=7)
        plt.plot(itr, cmv, label="Coeff. of Moment (v)", marker="o", markersize=7)
    else:
        plt.plot(itr, cdp, label="Coeff. of Drag (p)")
        plt.plot(itr, clp, label="Coeff. of Lift (p)")
        plt.plot(itr, cmp, label="Coeff. of Moment (p)")
    
    # plot formatting
    #ax.text(2000, -2.5, 'NACA 0012, M=0.5,\nα=3.0°, CFL-impl. 20',
    #    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    ax.text(7000, -2.8, 'Goland Wing, M=0.2,\nα=6.0°, CFL-impl. 0.1',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    plt.subplots_adjust(
        left=0.16, bottom=0.18, right=0.97, top=0.97)
    #ax.set_xscale("log")
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(100000))
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0E'))
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(100000))
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    ax.legend()
    ax.set_xlabel("iteration count")
    ax.set_ylabel("Load Coefficients")
    #plt.grid()
    ax.grid(b=True, which='major', color='0.20', linestyle='-')
    ax.grid(b=True, which='minor', color='0.80')
    ax.set_xlim(xmin=0) #,xmax=max(itr)
    title = resultsdir + "_loadsConv.png"
    #plt.savefig(title,bbox_inches='tight')
    plt.show()
    return()


def plotReport(resultsdir, viscid):
    """
    uses report file; 
    plots the mean and turbulent residuals
    report file residual values are base 10 logarithmic (value 2 = 10^2)
    """

    # import report file lists as array
    itr, cputime, rmean, rturb = np.array(getReport(resultsdir))
    # calculate non logarithmic values
    rmean, rturb = 10**rmean, 10**rturb
    
    # logarithmic plot wihtout cputime twinx
    fig1, ax1 = plt.subplots(figsize=(4,2.8))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual")
    ax1.plot(itr, rmean, label="Mean residual")
    if viscid:
        ax1.plot(itr, rturb, label="Turb. residual")
    # plot formatting
    ax1.set_yscale("log")
    #ax1.set_xscale("log")
    ax1.grid(b=True, which='major', color='0.20', linestyle='-')
    ax1.grid(b=True, which='minor', color='0.80')
    
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(100000))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0E'))
    #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    #ax1.ticklabel_format(axis='x', style='sci', scilimits=None, useOffset=None, useLocale=None, useMathText=None)
    

    """ #------------ TWINX WITH CPUTIME CODE ---------
    color=("tab:red")
    fig2, ax2 = plt.subplots(figsize=(4,2.8))
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("Residual", color = color)
    ax2.plot(itr, rmean, label="Mean residual", color = color)
    if viscid:
        ax2.plot(itr, rturb, label="Turb. residual", color = "tab:orange")
    ax2.set_yscale("log")
    ax2.grid()

    ax3 = ax2.twinx()
    color = ("tab:blue")
    ax3.set_ylabel("cumulative CPU time [s]", color = color)
    ax3.plot(itr, cputime, label="cpu time", color = color, linestyle = "--")
    #ax3.plot(itr, np.abs(cputime), label="cpu time", color = color)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylim(0)
    """
    #ax1.text(100500, 0.005, 'NACA 0012, M=0.5, α=3.0°\n CFL-impl. 20',
    #    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    #ax1.text(1, 0.005, 'NACA 0012, M=0.85,\nα=1.0°, CFL-impl. 20',
    #    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    #ax1.text(1, 0.003, 'Goland Wing, M=0.2,\nα=6.0°, CFL-impl. 0.1',
    #    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})

    lgd = fig1.legend(loc = 4, bbox_to_anchor=(0.92,0.80))
    plt.subplots_adjust(
        left=0.16, bottom=0.18, right=0.97, top=0.97)
    
    #fig.tight_layout()
    #title = resultsdir + "/report.svg"
    #plt.savefig(title,bbox_inches='tight')
    plt.show()
    return()
    

def plotPressureContours(resultsdir, coefficient = 0):
    """
    plots the pressure field around the aerofoil
    coefficient = True: Plots pressure coefficient (experimental)
    """
    
    # importing and converting flow dataframe
    if three_dim:
        #coord = float(input("type slice y-coordinate: "))
        coord = 0.05        # lazy option
        #tolerance = float(input("type tolerance: "))
        tolerance = 0.01    # lazy option
    else:
        coord = 0
        tolerance = 0.01
        
    #checkGlobalDf(resultsdir)
    flowdf = getYSlice(resultsdir, coord, tolerance, False)
    surfdf = getYSlice(resultsdir, coord, tolerance, True)

    x = flowdf['X'].values.tolist() # freeflow point data
    z = flowdf['Z'].values.tolist()
    p = flowdf['P'].values.tolist()
    xs = surfdf['X'].values.tolist() # surface coordinates of aerofoil
    zs = surfdf['Z'].values.tolist()

    # calculating pressure coefficients (experimental)
    if coefficient:
        cp = []

        #dataframe to find pressure at far field point
        params = flowdf.iloc[(flowdf['X']+48).abs().argsort()[:1]]
        pfree = params['P'].values[0]
    
        # determining free flow pressure
        for idx in range(len(p)):
            vext = 1
            dens = 1
            ccp = (p[idx]-pfree)*2/(dens*(vext*vext))
            cp.append(ccp)

    # plotting the pressure (coefficient)
    fig, ax1 = plt.subplots(figsize=(4,2.8))
    
    if coefficient:
        dcp = plt.tricontour(x, z, cp, 30, linewidths=0.5, colors='k', zorder=2) #lines
        #dcp = plt.tricontour(x, z, cp, 20, linewidths=0.5, zorder=2, cmap='RdGy') #lines
        #dp = plt.tricontourf(x, z, cp, 112, cmap=cm.rainbow, zorder=1) #fill
    else:
        dcp = plt.tricontour(x, z, p, 30, linewidths=0.5, colors='k', zorder=2) #lines
        #dp = plt.tricontourf(x, z, p, 112, cmap=cm.rainbow, zorder=1) #fill
    #plt.colorbar(dp)
    
    # plot aerofoil shape 
    x1, z1 = orderCoord(xs,zs)    # order surface points
    print(x1)
    print(z1)
    poly = Polygon(np.column_stack([x1,z1]), color = "w", fill =True, lw = 1, zorder=4)
    ax1.add_patch(poly)
    ax1.plot(x1,z1, color = "k",lw = 1, zorder=4)

    #fig.set_tight_layout(True)
    ax1.text(1, 0.5, 'NACA 0012, Re = 4.8e6\nM=0.85, α=1.0°',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3}) 

    ax1.set_aspect("equal", adjustable='box')
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")

    ax1.set_ylim(-0.7,0.7)
    ax1.set_xlim(-0.5,1.5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.subplots_adjust(
        left=0.16, bottom=0.15, right=0.97, top=0.97)
    plt.clabel(dcp, inline=True, fontsize=8,rightside_up=True, manual=True)
    
    plt.show()

    return()


def plotStreamlines(resultsdir, coefficient = 0):
    """
    plots the pressure field around the aerofoil
    coefficient = True: Plots pressure coefficient (experimental)
    """
    
    # importing and converting flow dataframe
    if three_dim:
        #coord = float(input("type slice y-coordinate: "))
        coord = 0.05        # lazy option
        #tolerance = float(input("type tolerance: "))
        tolerance = 0.01    # lazy option
    else:
        coord = 0
        tolerance = 0.01
        
    #checkGlobalDf(resultsdir)
    flowdf = getYSlice(resultsdir, coord, tolerance, False)
    surfdf = getYSlice(resultsdir, coord, tolerance, True)

    # constrain flowdf to 10x10c
    mask = flowdf["X"] >= (-8)
    df = flowdf[mask]
    mask = df["X"] <= (8)
    df = df[mask]
    mask = df["Z"] >= (-8)
    df = df[mask]
    mask = df["Z"] <= (8)
    flowdf = df[mask]

    print(flowdf)

    x = flowdf.T.to_numpy()[0] # freeflow point data
    z = flowdf.T.to_numpy()[2]
    r = flowdf.T.to_numpy()[3]
    u = flowdf.T.to_numpy()[4]
    w = flowdf.T.to_numpy()[6]
    p = flowdf.T.to_numpy()[7]
    speed = (u**2+w**2)**0.5
    xs = surfdf['X'].values.tolist() # surface coordinates of aerofoil
    zs = surfdf['Z'].values.tolist()


    # creating an interpolated meshgrid
    xg = np.linspace(-2.5, 3, 800)
    zg = np.linspace(-1, 1, 400)
    xi, zi = np.meshgrid(xg,zg)

    print(x)
    px = x.flatten()
    pz = z.flatten()
    pr = r.flatten()
    pu = u.flatten()
    pw = w.flatten()
    pspeed = speed.flatten()
    print(px)
    va = calcSound(1.401, p, r)
    Mach = [a/b for a,b in zip(pspeed,va)]
    

    gu = griddata((px,pz), pu, (xi,zi),fill_value=0.99)
    gw = griddata((px,pz), pw, (xi,zi),fill_value=0.1)
    gr = griddata((px,pz), pr, (xi,zi),fill_value=0.1)
    gspeed = griddata((px,pz), pspeed, (xi,zi),fill_value=1)
    gmach = griddata((px,pz), Mach, (xi,zi),fill_value=1)

    

    # plotting the
    #fig, ax1 = plt.subplots(figsize=(8,2.8))
    fig, ax1 = plt.subplots()
    #fig, ax1 = plt.subplots(figsize=(4,2.8))

    strm = ax1.streamplot(xi, zi, gu, gw, color=gmach, linewidth=2, cmap='rainbow')
    fig.colorbar(strm.lines, shrink = 0.80,ax = ax1)
    #ax1.scatter(x,z, s = 0.3)
    dcp = plt.tricontour(x, z, r, 60, linewidths=0.5, colors='k', zorder=2)
    
    
    # plot aerofoil shape 
    x1, z1 = orderCoord(xs,zs)    # order surface points
    poly = Polygon(np.column_stack([x1,z1]), color = "w", fill =True, lw = 1, zorder=4)
    ax1.add_patch(poly)
    ax1.plot(x1,z1, color = "k",lw = 1, zorder=4)

    #fig.set_tight_layout(True)
    ax1.text(-2.0, 0.6, 'NACA 0012, M=1.20, α=7.00°\nMach contours and streamlines',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3}) 

    ax1.set_aspect("equal", adjustable='box')
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")

    ax1.set_ylim(-0.7,0.7)
    ax1.set_xlim(-1.0,2.0)
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    #ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.subplots_adjust(
        left=0.16, bottom=0.15, right=0.97, top=0.97)
    
    plt.show()

    return()


def velMagn(u,v,w):
    """
    returns list of velocity magnitude for each point
    """
    vel = []
    for i in range(len(u)):
        vel.append((u[i]**2+v[i]**2+w[i]**2)**0.5)
    return vel


def calcSound(gamma, p, r):
    """
    returns local speed of sound for every point (not Mach number)
    """
    a = []
    for i in range(len(p)):
        a.append(math.sqrt(gamma*p[i]/r[i]))
    return a


def plotMachContours(resultsdir):
    """
    plots the local Mach number field around the aerofoil at chosen slice
    """
    
    # importing and converting flow dataframe
    if three_dim:
        #coord = float(input("type slice y-coordinate: "))
        coord = 0.05        # lazy option
        #tolerance = float(input("type tolerance: "))
        tolerance = 0.01    # lazy option
    else:
        coord = 0
        tolerance = 0.01
        
    #checkGlobalDf(resultsdir)
    flowdf = getYSlice(resultsdir, coord, tolerance, False)
    surfdf = getYSlice(resultsdir, coord, tolerance, True)

    x = flowdf['X'].values.tolist() # freeflow point data
    z = flowdf['Z'].values.tolist()
    p = flowdf['P'].values.tolist() # pressure
    r = flowdf['R'].values.tolist() # density (roh)
    u = flowdf['U'].values.tolist() # x-dir velocity
    v = flowdf['V'].values.tolist()
    w = flowdf['W'].values.tolist()
    xs = surfdf['X'].values.tolist() # surface coordinates of aerofoil
    zs = surfdf['Z'].values.tolist()

    gamma = 1.401 #specific heat ratio of air for standard conditions
    cp = []

    # calculate velocity magnitude list
    vel = velMagn(u, v, w)

    # calc local speed of sound
    va = calcSound(gamma, p, r)

    # calc local mach number
    Mach = [a/b for a,b in zip(vel,va)]
    print(max(Mach)) # prints point at highest Mach number
    
    # checking inflow mach number, dataframe to find far field point
    checkpoint = flowdf.iloc[(flowdf['X']+48).abs().argsort()[:1]].values[0]
    pc = checkpoint[7] # pressure
    rc = checkpoint[3] # density (roh)
    uc = checkpoint[4] # x-dir velocity
    vc = checkpoint[5]
    wc = checkpoint[6]
        # calculate velocity magnitude list
    velc = velMagn(u, v, w)
    vac = calcSound(gamma, p, r)
    Machc = velc[0]/vac[0]
    print(Machc) # prints point at highest Mach number

    fig, ax1 = plt.subplots(figsize=(4,2.8))
    dmach = plt.tricontour(x, z, Mach, 16, linewidths=0.5, colors='k', zorder=2) #lines
    #dmachf = plt.tricontourf(x, z, Mach, 112, cmap=cm.rainbow, zorder=1) #fill
    #plt.colorbar(dmachf)
    
    # plot aerofoil shape 
    x1, z1 = orderCoord(xs,zs)    # order surface points
    poly = Polygon(np.column_stack([x1,z1]), color = "w", fill =True, lw = 1, zorder=4)
    ax1.add_patch(poly)
    ax1.plot(x1,z1, color = "k",lw = 1, zorder=4)

    #fig.set_tight_layout(True)
    ax1.set_aspect("equal", adjustable='box')
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")

    ax1.set_ylim(-0.7,0.7)
    ax1.set_xlim(-0.5,1.5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.subplots_adjust(
        left=0.22, bottom=0.15, right=0.97, top=0.97)
    
    ax1.text(1, 0.5, 'NACA 0012, Re = 4.8e6\nM=0.85, α=1.0°',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3}) 
    
    plt.clabel(dmach, inline=True, fontsize=8,rightside_up=True, manual=True)

    #title = resultsdir + "/MachContours.svg"
    #plt.savefig(title,bbox_inches='tight')
    plt.show()
    return()


def plotCpx(resultsdir):
    """
    Plots a surface CP over x/c
    optional: add experiment data or other
    """

    # importing and converting flow dataframe
    if three_dim:
        coord = float(input("type slice y-coordinate: "))
        #coord = 0.05        # lazy option
        tolerance = float(input("type tolerance: "))
        #tolerance = 0.01    # lazy option
    else:
        coord = 0
        tolerance = 0.01
        
    #checkGlobalDf(resultsdir)
    df = getYSlice(resultsdir, coord, tolerance, True)
    #print(df)
    # splitting at Z=0 and sorting in X-ascending order
    mask = df["Z"] >= 0
    dfu = df[mask]
    dfl = df[~mask]
    
    dfu = dfu.sort_values("X", ascending=True)
    #print(dfu)
    dfl = dfl.sort_values("X", ascending=True)
    xu = dfu.T.to_numpy()[0]    # x upper
    xl = dfl.T.to_numpy()[0]    # x lower
    cpu = dfu.T.to_numpy()[8]   #cp upper
    cpl = dfl.T.to_numpy()[8]
    
    # plot the whole thing
    #fig, ax = plt.subplots(figsize=(3.13, 2.5)) #halfsize (too small)
    fig, ax = plt.subplots(figsize=(4.0, 2.8)) #compromise 0.7=AR
    #fig, ax = plt.subplots(figsize=(8.0, 3.8)) #compromise x=AR
    #fig, ax = plt.subplots(figsize=(6.27, 4)) #fullsize 0.62=AR
    ax.plot(xu, cpu, lw=1, label= "Glasgow Meshless", Color = "k")
    ax.plot(xl, cpl, lw=1, label= "", Color = "k", linestyle = '--')
    """
    ----SECTION BELOW TO ADD EXPERIMENTAL DATA AND DIGITISED DATA----
    """

    """
           # import csv from digitiser
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

    ax.scatter(xpl, expl, s= 8, label= "Exp. (Harris, 1981)", Color = "r")
    ax.scatter(xpu, expu, s= 8, label= "", Color = "r")
    ax.plot(xc, cpuc, lw=1, label= "M.K. Singh et al", Color = "b")
    ax.plot(xc, cpul, lw=1, label= "", Color = "b", linestyle = "--")


    ax.text(0.3, -0.8, 'NACA 0012, M=0.65, α=1.86°\n CFL 20, 1E+05 iterations',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})

    """    
    """
    ----SECTION ABOVE TO ADD EXPERIMENTAL DATA AND DIGITISED DATA----
    """

    ax.text(0.2, -0.8, 'Goland wing, AR=6.7, Inviscid\nM=0.2, α=6.0°, slice at y=3c',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    
    ax.set_ylabel("-Cp")
    ax.set_xlabel("x/c")
    ax.legend(loc = 1)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlim(-0.00,1)
    ax.set_ylim(-1.2,1.6)
 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # adding half aerofoil shape
    xs = df['X'].values.tolist() # freeflow point data
    zs = df['Z'].values.tolist()

    ax2 = ax.twinx()
    ax2.set_xlim(0,1)
    #ax2.set_ylim(0,0.62) #full
    ax2.set_ylim(0,0.70) #compromise
    #ax2.set_ylim(0,0.315) #half

    x1, z1 = orderCoord(xs,zs)    # order surface points
    poly = Polygon(np.column_stack([x1,z1]), color = "w", fill =True, lw = 1, zorder=4)
    ax2.add_patch(poly)
    ax2.plot(x1,z1, color = "k",lw = 1, zorder=4)
    ax2.get_yaxis().set_ticks([])

    #plt.subplots_adjust(
    #    left=0.10, bottom=0.11, right=0.98, top=0.97) #full
    #plt.subplots_adjust(
    #    left=0.15, bottom=0.15, right=0.97, top=0.97)
    plt.subplots_adjust(
        left=0.07, bottom=0.15, right=0.97, top=0.97)
    #title = resultsdir + "/CPx.svg"
    #plt.savefig(title,bbox_inches='tight')
    plt.show()
    return()


def getCurl(resultsdir):
    """
    calculates and possibly plots the velocity curl (field) of a slice of data

    FUNCTION NOT WORKING YET
    CURL CALCULATION WORKS ONLY IN MATLAB
    WILL IMPLEMENT A WAY TO RUN MATLAB FROM INSIDE THIS PROGRAM
    USING CSV TO TRANSFER DATA
    """

    # importing and converting flow dataframe
    if three_dim:
        #coord = float(input("type slice y-coordinate: "))
        coord = 0.05        # lazy option for debug
        #tolerance = float(input("type tolerance: "))
        tolerance = 0.01    # lazy option
    else:
        coord = 0
        tolerance = 0.01
        
    if three_dim:        
        flowdf = getYSlice(resultsdir, coord, tolerance, surface=False)
        surfdf = getYSlice(resultsdir, coord, tolerance, surface=True)
    else:
        flowdf = getYSlice(resultsdir, surface=False)
        surfdf = getYSlice(resultsdir, surface=True)

    # export as CSV    
    flowdf.to_csv(resultsdir+"_flowdf0.05.csv")
    surfdf.to_csv(resultsdir+"_surfdf0.05.csv")
    
    #x = flowdf['X'].to_numpy() # freeflow point data
    #y = flowdf['Y'].to_numpy()
    #z = flowdf['Z'].to_numpy()
    #u = flowdf['U'].to_numpy() # x-dir velocity
    #v = flowdf['V'].to_numpy()
    #w = flowdf['W'].to_numpy()

    #points = np.hstack((x,z))
    #points = np.array([[1, 2],[2, 3],[3, 4], [6,5], [1,6]])
    
    #triangulate
    #tri = ssp.Delaunay(points)
    #hull = ssp.ConvexHull(points)
    #print(tri)

    return()


def plotLiftslope():
    directory = "/home/rinaldo/Desktop/liftslope.csv"
    data = np.genfromtxt(directory, delimiter=',')

    fig, ax = plt.subplots(figsize=(4.0, 2.8)) #compromise 0.7=AR
    #fig, ax = plt.subplots(figsize=(8.0, 3.8)) #compromise x=AR
    #fig, ax = plt.subplots(figsize=(6.27, 4)) #fullsize 0.62=AR

    ax.plot(data[:,0], data[:,2], lw=1, label= "Glasgow M",color ="b")
    ax.plot(data[:,0], data[:,4], lw=1, label= "XFoil",color ="b", linestyle = "--")
    ax2=ax.twinx()
    ax2.plot(data[:,0], data[:,3], lw=1, label= "Cm GM",color ="r")
    ax2.plot(data[:,0], data[:,5], lw=1, label= "Cm xfoil",color ="r", linestyle = "--")
    #ax.plot(xl, cpl, lw=1, label= "", Color = "k", linestyle = '--')
    ax2.set_ylabel("Cm",color ="r")
    ax2.ticklabel_format(style='sci',scilimits=(-3,-2))
    #ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1E'))
    #ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    
    ax.text(7, 0.28, 'NACA 0012\nM=0.20\ninviscid',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})

    ax.set_ylabel("Cl",color ="b")
    ax.set_xlabel("AoA [°]")
    ax.legend(loc = 2)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    #ax.set_xlim(-0.00,1)
    #ax.set_ylim(-1.2,1.6)
 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    #plt.subplots_adjust(
    #    left=0.15, bottom=0.16, right=0.97, top=0.97)
    plt.subplots_adjust(
        left=0.15, bottom=0.16, right=0.82, top=0.92)
    plt.show()
    return()


def plotLiftslope2():
    directory = "/home/rinaldo/Desktop/Goland.csv"
    data = np.genfromtxt(directory, delimiter=',')

    fig, ax = plt.subplots(figsize=(4.0, 2.8)) #compromise 0.7=AR
    #fig, ax = plt.subplots(figsize=(8.0, 3.8)) #compromise x=AR
    #fig, ax = plt.subplots(figsize=(6.27, 4)) #fullsize 0.62=AR

    ax.plot(data[:,0], data[:,1]/3.35, lw=1, label= "Glasgow M",color ="b")
    ax.plot(data[:,0], data[:,3], lw=1, label= "LLT",color ="b", linestyle = "--")
    #ax.scatter(data[:,0], data[:,1]/3.35, lw=1,color ="b", s=14, marker="x")
    #ax.scatter(data[:,0], data[:,3], lw=1,color ="b", s=14, marker="o")

    ax2=ax.twinx()
    ax2.plot(data[:,0], data[:,2]/3.35, lw=1, label= "Cdi GM",color ="r")
    ax2.plot(data[:,0], data[:,4], lw=1, label= "Cdi LLT",color ="r", linestyle = "--")
    #ax.plot(xl, cpl, lw=1, label= "", Color = "k", linestyle = '--')
    ax2.set_ylabel("Cdi",color ="r")
    #ax2.ticklabel_format(style='sci',scilimits=(-3,-2))
    #ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1E'))
    #ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    
    ax.text(7, 0.28, 'NACA 0012\nM=0.20\ninviscid',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})

    ax.set_ylabel("CL",color ="b")
    ax.set_xlabel("AoA [°]")
    ax.legend(loc = 2)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlim(-0.00,5)
    ax.set_ylim(-0.1,0.45)
    ax2.set_ylim(-0.01,0.045)
 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    #plt.subplots_adjust(
    #    left=0.15, bottom=0.16, right=0.97, top=0.97)
    plt.subplots_adjust(
        left=0.15, bottom=0.16, right=0.82, top=0.92)
    plt.show()
    return()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ----------------------------- 2D ONLY FUNCTIONS -----------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def getDmin(resultsdir):
    """
    returns grid points and dmin lists
    ACTUALLY NOT 2D ONLY BUT UNTESTED ON 3D
    """

    x = []
    y = []
    z = []
    dmin = []
    
    # imports dmin file and splits into lists
    inpath = resultsdir + ".dmin.tec"
    with open(inpath) as file:
        data = file.readlines()
        del data[0] #delete header line
        for item in data:
            a, b, c, d = item.split()
            x.append(float(a))
            y.append(float(b))
            z.append(float(c))
            dmin.append(float(d))
    
    return x, y, z, dmin


def plotDmin(resultsdir):
    """
    visualisation of the minimal distance of a node to the aerofoil
    also returns x,y,z,dmin lists
    """
    x, y, z, dmin = getDmin(resultsdir)

    # sorting by distance:
    data = np.array([x,y,z,dmin])
    dataT = np.transpose(data)
    res = sorted(dataT, key = itemgetter(3))

    datasorted = np.transpose(dataT)
    datasorted = np.fliplr(datasorted) #print high dmin values above others
    x = datasorted[0]
    y = datasorted[1]
    z = datasorted[2]
    dmin = datasorted[3]

    print("maximum distance of a point: " + str(max(dmin)))

    # rainbow colourmap of points
    fig = plt.figure(figsize=(4.0, 2.8))
    dp = plt.scatter(x,z,c=dmin, marker = "o", cmap=cm.rainbow)
    plt.title("Minimum distance to aerofoil (dmin)")
    plt.colorbar(dp)

    # rainbow colourmap, triangulation
    fig2 = plt.figure(figsize=(4.0, 2.8))
    plt.tricontour(x, z, dmin, 14, linewidths=0.5, colors='k') #lines
    dp = plt.tricontourf(x, z, dmin, 112, cmap=cm.rainbow) #fill
    plt.title("Minimum distance to aerofoil (dmin)")
    plt.colorbar(dp)

    # histogram of dmin
    fig3 = plt.figure(figsize=(4.0, 2.8))
    cmp = plt.cm.get_cmap("rainbow")
    n, bins, patches = plt.hist(dmin, 14, color='green', log=True, edgecolor='black', linewidth=0.5)
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) #normed=1, 
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmp(c))

    #plt.hist(dmin, 15, log=True) #enable logarithmic values
    
    plt.show()
    return x, y, z, dmin


def plotCloud(resultsdir):
    """
    plots the point cloud, where surface points are red x markers
    """
    # get free flow (outer) points
    xo, yo, zo, dmin = getDmin(resultsdir)

    # get surface points
    checkGlobalDf(resultsdir)
    xs = gsurfdataframe['X'].values.tolist()
    zs = gsurfdataframe['Z'].values.tolist()
    
    # plotting values
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 2.8))
    ax1 = plt.subplot(121)
    plt.scatter(xo, zo, marker='.', color = "k", s = 1)
    plt.scatter(xs, zs, marker='x', color = "r", s = 7)
    ax2 = plt.subplot(122)
    plt.scatter(xo, zo, marker='.', color = "k", s = 1)
    plt.scatter(xs, zs, marker='x', color = "r", s = 7)
    
    # plot formatting
    fig.set_tight_layout(True)
    ax1.set_aspect("equal", adjustable='box')
    ax2.set_aspect("equal", adjustable='box')
    
    plt.show()
    return()


def plotDens(resultsdir):
    """
    calculates and plots the density of the point cloud
    very resource intensive through weighted gaussian function (NP or exponential or so)
    """

    # import point cloud
    #x, y, z, dmin = getDmin(resultsdir)
    df = getYSlice(resultsdir, y=0, tolerance=0.01, surface=0)
    surfdf = getYSlice(resultsdir, y=0, tolerance=0.01, surface=1)
    x = df['X'].values.tolist() # freeflow point data
    z = df['Z'].values.tolist()
    xo = surfdf['X'].values.tolist()
    zo = surfdf['Z'].values.tolist()
    xz = np.vstack([x,z])      

    # Calculate the point density
    v = gaussian_kde(xz)(xz)

    # Sort the points by density, so that the densest points are plotted last
    data = np.array([x,z,v])
    dataT = np.transpose(data)
    res = sorted(dataT, key = itemgetter(2))
    datasorted = np.transpose(dataT)
    #datasorted = np.fliplr(datasorted) # outervalues on top
    x = datasorted[0]
    z = datasorted[1]
    v = datasorted[2]

    # two logarithmic density plots, tri, RdGy
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.8))
    ax1 = plt.subplot(121)
    plt.tricontour(x, z, abs(np.log10(v)), 12, linewidths=0.5, cmap='RdGy')
    ax2 = plt.subplot(122)
    plt.tricontour(x, z, abs(np.log10(v)), 12, linewidths=0.5, cmap='RdGy')
    
    fig.set_tight_layout(True)
    ax1.set_aspect("equal", adjustable='box')
    ax2.set_aspect("equal", adjustable='box')
    
    # logarithmic density plot
    fig2, ax3 = plt.subplots(figsize=(4, 2.8))
    ax3.scatter(x, z, c=v, s=50, edgecolor='', cmap=cm.rainbow)
    
    # logarithmic density plot, tri with colour fill and stuff
    fig3, ax4 = plt.subplots(figsize=(8, 3.8))
    #plt.tricontour(x, z, abs(np.log10(v)), 6, linewidths=0.5, cmap='RdGy')
    plt.tricontour(x, z, abs(np.log10(v)), 6, linewidths=0.5, colors='k') #lines
    dp = plt.tricontourf(x, z, -abs(np.log10(v)), levels=112, cmap=cm.viridis_r) #fill
    ax4.scatter(x,z, s= 0.5, zorder=1)
    cbar = plt.colorbar(dp, ticks=[np.log10(1), np.log10(0.1), np.log10(0.01), np.log10(0.001), np.log10(0.0001), np.log10(0.00001), np.log10(0.000001)])
    cbar.ax.set_yticklabels(['10⁰', '10⁻¹', '10⁻²', '10⁻³', '10⁻⁴', '10⁻⁵', '10⁻⁶'])
    axes = plt.gca()
    ax4.set_ylim([-30,30])
    ax4.set_xlim([-20,50])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.draw()
    
    # adding the aerofoil shape
    # sorting x, z
    x1, z1 = orderCoord(xo,zo)
    poly = Polygon(np.column_stack([x1,z1]), color = "w", fill =True, lw = 1, zorder=3)
    #ax1.add_patch(poly)
    #ax2.add_patch(poly)
    #ax3.add_patch(poly)
    ax4.add_patch(poly)

    plt.show()
    return v


def testFunction(resultsdir):
    """
    no use
    """
    checkGlobalDf(resultsdir)
    df = gsurfdataframe.loc[:,("X", "Z")]

    print(df)
    return()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ----------------------------- 3D ONLY FUNCTIONS -----------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def getXSlice(resultsdir, x=0, tolerance=0.01, surface=0):
    """
    returns X-slice DataFrame of data between y-tolerance and y+tolerance
    surface points (if applicable) sorted in circular fashion starting at trailing edge -> counter clockwise
    recommended tolerance: 0.01
    """
    
    # import dataframe
    if "gsurfdataframe" not in globals():
        checkGlobalDf(resultsdir)

    # creating a copy
    if surface:
        dfx = gsurfdataframe.copy()
    else:
        dfx = gflowdataframe.copy()
        
    # choosing a slice
    mask = dfx["X"] >= (x-abs(tolerance))
    dfx = dfx[mask]
    mask = dfx["X"] <= (x+abs(tolerance))
    dfx = dfx[mask]
    
    # ordering dataframe in circular fashion, TE and LE at Z=0
    # only for surface points
    if surface:
        mask = dfx["Z"] > 0
        dfu = dfx[mask]     # upper surface
        dfl = dfx[~mask]    # lower surface

        dfu = dfu.sort_values("Y", ascending=False)
        dfl = dfl.sort_values("Y", ascending=True)
        points = dfu.append(dfl)
        points.reset_index(inplace=True) # new indexing in df
        del points['index']
    else:
        points = dfx.copy(deep = False)
    return points # returns dataframe of all surface points in slice


def getZSlice(resultsdir, x=0, lower=0, surface=0):
    """
    returns X-slice DataFrame of data between y-tolerance and y+tolerance
    surface points (if applicable) sorted in circular fashion starting at trailing edge -> counter clockwise
    recommended tolerance: 0.01
    """
    
    # import dataframe
    if "gsurfdataframe" not in globals():
        checkGlobalDf(resultsdir)

    # creating a copy
    if surface:
        dfx = gsurfdataframe.copy()
    else:
        dfx = gflowdataframe.copy()
        
    # choosing a slice
    if lower:
        mask = dfx["Z"] <= x
    else:
        mask = dfx["Z"] >= x
    dfx = dfx[mask]

    points = dfx.copy(deep = False)
    return points # returns dataframe of all surface points in slice


def plotMultiCpx(resultsdir, slices, skip = 4):
    """
    Plots a the surface CP over x/c for multiple y slices;
    One dataframe per slice, x0u, x1u,... and x0l, x1l... as values, where the index is slice index
    Skip: number of slices per plotted slice
    """

    #tolerance = float(input("type tolerance: "))
    tolerance = 0.001   # lazy option

    # pre-allocating lists
    no_slices = (len(slices))
    print("number of slices :" + str(no_slices))
    df_list = [0]*no_slices
    dfu = [0]*no_slices
    dfl = [0]*no_slices
    xu = [0]*no_slices
    xl = [0]*no_slices
    cpu = [0]*no_slices
    cpl = [0]*no_slices
    
    # splitting at Z=0 and sorting in X-ascending order
    for index, value in enumerate(slices):
        if index % skip == 0 or index == no_slices-1:
            df_list[index] = getYSlice(resultsdir, value, tolerance, True)

            mask = df_list[index]["Z"] >= 0
            dfu[index] = df_list[index][mask]
            dfl[index] = df_list[index][~mask]
        
            dfu[index] = dfu[index].sort_values("X", ascending=False)
            dfl[index] = dfl[index].sort_values("X", ascending=True)
            xu[index] = dfu[index].T.to_numpy()[0]    # x upper
            xl[index] = dfl[index].T.to_numpy()[0]    # x lower
            cpu[index] = dfu[index].T.to_numpy()[8]   #cp upper
            cpl[index] = dfl[index].T.to_numpy()[8]

    # plot the whole thing
    #fig, ax = plt.subplots(figsize=(3.13, 2.5)) #halfsize (too small)
    fig, ax = plt.subplots(figsize=(4.0, 2.8)) #compromise 0.7=AR
    #fig, ax = plt.subplots(figsize=(6.27, 4)) #fullsize 0.62=AR
    #fig, ax = plt.subplots(figsize=(8, 4.5)) #fullsize ?=AR

    for index, value in enumerate(slices):
        if index % skip == 0 or index == no_slices-1:
            # normalise slice value for colour
            normval = value/max(slices)
            cmap = mpl.cm.get_cmap('viridis')
            
            ax.plot(xu[index], cpu[index], lw=0.5, label= ("y/c: "+ str("{:.3f}".format(normval)))
                    , Color = "k", alpha = 0.5)
            ax.plot(xl[index], cpl[index], lw=0.5, label= "", Color = "b", alpha = 0.5, ls="--")
            x1 = np.concatenate([xu[index], xl[index]])
            z1 = np.concatenate([cpu[index],cpl[index]])
            poly = Polygon(np.column_stack([x1,z1]), alpha=0.2, facecolor=cmap(normval), fill=True, lw=None)
            #p = mpl.collections.PatchCollection(poly,cmap=mpl.cm.jet,alpha=0.3)
            #ax.add_collection(p)
            ax.add_patch(poly)
            print(index)
            ## IMPLEMENT CHANGING COLOUR. ALSO PLOT LARGER

    """
    """
    #----SECTION BELOW TO ADD EXPERIMENTAL DATA AND DIGITISED DATA----
    """
    #ax.scatter(xpl, expl, s= 8, label= "Experiment (Harris, 1981)", Color = "r")
    #ax.scatter(xpu, expu, s= 8, label= "", Color = "r")
    #ax.plot(xc, cpuc, lw=1, label= "M.K. Singh et al", Color = "k", linestyle = "--")
    #ax.plot(xc, cpul, lw=1, label= "", Color = "k", linestyle = "--")
    """
    #----SECTION ABOVE TO ADD EXPERIMENTAL DATA AND DIGITISED DATA----
    
    #ax.text(0.2, -0.5, 'NACA 0012, Re = 4.8e8\nM=0.65, α=1.86°\n fix aspect ratio',
    #    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    ax.text(0.2, -0.8, 'Goland wing, AR=6.7, Inviscid\nM=0.2, α=6.0°',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    ax.set_ylabel("-Cp")
    ax.set_xlabel("x/c")
    #ax.legend()
    
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlim(-0.00,1)
    ax.set_ylim(-1.2,1.2)

 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # adding half aerofoil shape
    xs = dfu[0]['X'].values.tolist() # freeflow point data
    zs = dfu[0]['Z'].values.tolist()

    ax2 = ax.twinx()
    ax2.set_xlim(0,1)
    #ax2.set_ylim(0,0.62) #full
    ax2.set_ylim(0,0.70) #compromise
    #ax2.set_ylim(0,0.315) #half

    x1, z1 = orderCoord(xs,zs)    # order surface points
    poly = Polygon(np.column_stack([x1,z1]), color = "w", fill =True, lw = 1, zorder=4)
    ax2.add_patch(poly)
    ax2.plot(x1,z1, color = "k",lw = 1, zorder=4)
    ax2.get_yaxis().set_ticks([])

    #plt.subplots_adjust(
    #    left=0.10, bottom=0.11, right=0.98, top=0.97) #full
    plt.subplots_adjust(
        left=0.08, bottom=0.10, right=0.97, top=0.97)
    #title = resultsdir + "/CPx.svg"
    #plt.savefig(title,bbox_inches='tight')
    ax.annotate("towards tip", xy=(0.15,0.18), xycoords='data', xytext=(0.22,0.85), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    ax.annotate("tip dome", xy=(0.5,0.4), xycoords='data', xytext=(0.61,0.85), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    #fig.canvas.draw()
                
    plt.show()
    
    return()


def integSliceCoefs(resultsdir, y, t):
    """
    BROKEN NEEDS TO BE FIXED (SIGNS) BECAUSE SWITCH FROM CW TO CCW!
    returns coefficient of lift, drag and moment@0.25c for the current
    aerofoil slice, using getYSlice
    """
    # import points dataframe (sorted ccw from trailing edge)
    points = getYSlice(resultsdir, y, t, True)  #print(points.to_string())

    # number of points (rows)
    rows, collumns = points.shape

    # integrate coefficients, clockwise!!!!! trapezoidal (-):
    Cl = 0
    Cd = 0
    Cm = 0
    #print(points.loc[0]['CP'])
    for index in range(rows-1):
        localCl = 0.5* (points.loc[index]['CP']+points.loc[index+1]['CP'])* (points.loc[index+1]['X']-points.loc[index]['X'])
        localCd = 0.5* (points.loc[index]['CP']-points.loc[index+1]['CP'])* (points.loc[index+1]['Z']-points.loc[index]['Z'])
        localCml = 0.5* ((points.loc[index]['CP']+points.loc[index+1]['CP'])* (points.loc[index+1]['X']-points.loc[index]['X'])\
                 *((points.loc[index]['X']+points.loc[index+1]['X'])/2-0.25))
        localCmd = 0.25*((points.loc[index]['CP']+points.loc[index+1]['CP'])* (points.loc[index+1]['Z']-points.loc[index]['Z'])\
                 *(points.loc[index]['X']-points.loc[index+1]['X']))
        localCm = localCml + localCmd
        #print(localCl)#print(localCd)#print(localCm)
        Cl -= localCl
        Cd -= localCd
        Cm -= localCm
        
    localCl = 0.5* (points.loc[index+1]['CP']+points.loc[1]['CP'])* (points.loc[index+1]['X']-points.loc[1]['X'])
    localCd = 0.5* (points.loc[index+1]['CP']-points.loc[1]['CP'])* (points.loc[index+1]['Z']-points.loc[1]['Z'])
    localCml = 0.5* ((points.loc[index+1]['CP']+points.loc[1]['CP'])* (points.loc[1]['X']-points.loc[index+1]['X'])\
            *((points.loc[index+1]['X']+points.loc[1]['X'])/2-0.25))
    localCmd = 0.25*((points.loc[index+1]['CP']-points.loc[1]['CP'])* (points.loc[1]['Z']-points.loc[index+1]['Z'])\
            *(points.loc[index+1]['X']+points.loc[1]['X']))
    localCm = localCml + localCmd
    #print(localCl)#print(localCd)#print(localCm)
    Cl -= localCl
    Cd += localCd
    Cm -= localCm

    #print("")
    #print("Cl: "+ str(Cl))
    #print("Cd: "+ str(Cd))
    #print("Cm: "+ str(Cm))
    return Cl, Cd, Cm


def printSliceCoefs(resultsdir, y, t):
    Cl, Cd, Cm = integSliceCoefs(resultsdir, y, t)
    print("Cl: "+ str(Cl))
    print("Cd: "+ str(Cd))
    print("Cm: "+ str(Cm))
    return()


def coefsFromSlices(resultsdir, slices):
    #slices = [0.05, 0.115, 0.20, 0.307, 0.44, 0.602, 0.794, 1.011, 1.25, 1.5, 1.75, 1.995, 2.225, 2.434, \
    #          2.617, 2.902, 3.008, 3.092, 3.158, 3.21, 3.25]
    #wingspan = 6.5 #*c
    Cl=[0]*len(slices)
    Cd=[0]*len(slices)
    Cm=[0]*len(slices)

    # calculating coefficients lists
    for index, value in enumerate(slices):
        #print(value)
        Cl[index], Cd[index], Cm[index] = integSliceCoefs(resultsdir, value, 0.005)  #print(points.to_string())

    # integrating with simspons rule
    CL = integ.simps(Cl,slices)*2
    CD = integ.simps(Cd,slices)*2
    CM = integ.simps(Cm,slices)*2

    # print values
    print("CL: "+ str(CL))
    print("CD: "+ str(CD))
    print("CM: "+ str(CM))
    return()


def plotSliceCoefs(resultsdir, slices):
    """
    Plots the loads over spanwise distance for finite wing
    """
    
    # setting up the calculation with point slices
    #slices = [0.05, 0.115, 0.20, 0.307, 0.44, 0.602, 0.794, 1.011, 1.25, 1.5, 1.75, 1.995, 2.225, 2.434, \
    #          2.617, 2.902, 3.008, 3.092, 3.158, 3.21, 3.25]
    #wingspan = 6.5 #*c
    print(len(slices))
    Cl=[0]*len(slices)
    Cd=[0]*len(slices)
    Cm=[0]*len(slices)

    # calculating coefficients lists
    for index, value in enumerate(slices):
        #print(value)
        Cl[index], Cd[index], Cm[index] = integSliceCoefs(resultsdir, value, 0.001)  #print(points.to_string())

    # plotting coefficients over y
    fig, ax = plt.subplots(figsize=(4.0, 2.8)) #compromise 0.7=AR
    ax.plot(slices, Cl, label= "Cl")
    ax.plot(slices, Cd, label= "Cd")
    ax.plot(slices, Cm, label= "Cm 0.25")

    # add LLT
    LLT = np.genfromtxt('Cl.csv', delimiter=',')#*0.9905 #M0.2
    LLTz = np.genfromtxt('z.csv', delimiter=',')*3.35
    no_points=len(LLT)
    #t = np.linspace(0, 3.35, no_points)
    


    #plt.plot( 3.35*np.cos(t) , 0.53*np.sin(t) ,label="ellipt. Cl-fit")
    plt.plot(LLTz,LLT,label="LLT")
    

    ax.text(0.2, -0.12, 'Goland wing, AR=6.7, Inviscid\nM=0.10, α=5.0°',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    ax.set_ylabel("Force Coefficients")
    ax.set_xlabel("Spanwise coordinate/chord y/c")

    ax.legend()
    ax.grid(color='lightgray',linestyle='--')

    ax.set_xlim(0,3.5)
    plt.subplots_adjust(
        left=0.22, bottom=0.15, right=0.97, top=0.97)
    plt.show()
    
    return


def plotVectors(resultsdir, coord, tolerance):
    """
    vector plot x slice
    """

    flowdf = getXSlice(resultsdir, x=coord, tolerance=tolerance, surface=0)

    x = flowdf['X'].to_numpy() # freeflow point data
    y = flowdf['Y'].to_numpy()
    z = flowdf['Z'].to_numpy()
    u = flowdf['U'].to_numpy() # x-dir velocity
    v = flowdf['V'].to_numpy()
    w = flowdf['W'].to_numpy()
    p = flowdf['P'].to_numpy()
    r = flowdf['R'].to_numpy()
    speed = (u**2+v**2+w**2)**0.5

    w = w-math.sin(6*math.pi/180) #assuming free flow: |V|(u,w)=1, v=0

    # creating an interpolated meshgrid
    #yg = np.linspace(0.1, 10, 20)
    #zg = np.linspace(-5, 5, 20)
    yg = np.linspace(0, 15, 200)
    zg = np.linspace(-15, 15, 200)
    yi, zi = np.meshgrid(yg,zg)
    
    px = x.flatten()
    py = y.flatten()
    pz = z.flatten()
    pr = r.flatten()
    pu = u.flatten()
    pv = v.flatten()
    pw = w.flatten()
    pspeed = speed.flatten()
    va = calcSound(1.401, p, r)
    Mach = [a/b for a,b in zip(pspeed,va)]
    

    gu = griddata((py,pz), pu, (yi,zi)) #,fill_value=0.99
    gv = griddata((py,pz), pv, (yi,zi),fill_value=0.1)
    gw = griddata((py,pz), pw, (yi,zi)) #,fill_value=0.1
    gr = griddata((py,pz), pr, (yi,zi),fill_value=0.1)
    gspeed = griddata((py,pz), pspeed, (yi,zi),fill_value=1)
    gmach = griddata((py,pz), Mach, (yi,zi),fill_value=1)


    np.savetxt("TE1.csv",yi , delimiter=",")
    np.savetxt("TE2.csv",zi , delimiter=",")
    np.savetxt("TE3.csv",gv , delimiter=",")
    np.savetxt("TE4.csv",gw , delimiter=",")

    rect = mpl.patches.Rectangle((0,-0.02),3.35,0.04,color="k")
    
    fig, ax = plt.subplots(figsize=(4,4))
    ax.add_patch(rect)
    Q = plt.quiver(yi, zi, gv, gw, gspeed, cmap="rainbow", scale=2)
    qk = ax.quiverkey(Q, 0.85, 0.15, 0.1, r'$0.1 \frac{c}{s}$', labelpos='E',
                   coordinates='figure')
    ax.set_xlabel("spanwise coord. (y)")
    ax.set_ylabel("normal coord. (z)")
    ax.set_xlim(2,5)
    
    ax.text(3.0, 1.1, 'Goland wing, AR=6.7, Inviscid\nM=0.20, α=6.00°, 2E+4 iter\nflow velocity trailing edge',
        bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3})
    plt.subplots_adjust(
        left=0.13, bottom=0.11, right=0.96, top=0.94)
    
    plt.show()
    
    return()


def plotXYCPContours(resultsdir):
    """
    y-x plot of projected surface pressure contours
    """
    df = getZSlice(resultsdir, 0, 0, 1) #upper
    df2 = getZSlice(resultsdir, 0, 1, 1) #lower

    #fig, (ax,ax2) = plt.subplots(2,1,figsize=(8.0, 2.8)) #compromise 0.7=AR
    fig, (ax,ax2) = plt.subplots(2,1,figsize=(8.0, 4.8)) #compromise 0.7=AR
    x = df['X'].to_numpy() # freeflow point data
    y = df['Y'].to_numpy()
    z = df['Z'].to_numpy()
    cp= df['CP'].to_numpy()
    print(df.to_string())
    print(x)
    print(cp)
    x2 = df2['X'].to_numpy() # freeflow point data
    y2 = df2['Y'].to_numpy()
    z2 = df2['Z'].to_numpy()
    cp2= df2['CP'].to_numpy()
    

    #dcc = ax.tricontour( y, x, cp, 20, linewidths=0.5, colors='k', zorder=2, alpha = 0.5) #lines
    dcp = ax.tricontourf(y, x, cp, 20, zorder=1, cmap='rainbow') #lines
    #dcb = plt.colorbar(dcp, aspect=5, shrink=0.65)
    dcp2 = ax2.tricontourf(y2, x2, cp2, 20, zorder=1, cmap='rainbow') #lines
    #dcc = ax2.tricontour( y, x, cp, 20, linewidths=0.5, colors='k', zorder=2, alpha = 0.5) #lines
    #dcb = plt.colorbar(dcp2, aspect=5, shrink=0.65)

    plt.subplots_adjust(
        left=0.07, bottom=0.00, right=0.86, top=1.0)
    
    cbar_ax = fig.add_axes([0.88, 0.55, 0.05, 0.4])
    cbar = fig.colorbar(dcp, cax=cbar_ax)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.05, 0.4])
    cbar = fig.colorbar(dcp2, cax=cbar_ax)

    ax.set_aspect("equal", adjustable='box')
    #ax.set_xlabel("spanwise coordinate (y)")
    ax.set_ylabel("chordwise coordinate (x)")
    #ax.set_title("Upper surface -Cp distribution")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax2.set_aspect("equal", adjustable='box')
    ax2.set_xlabel("spanwise coordinate (y)")
    ax2.set_ylabel("chordwise coordinate (x)")
    #ax2.set_title("Upper surface -Cp distribution")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax.text(2.0, 0.8, 'Goland wing, AR=6.7, Inviscid\nM=0.10, α=5.00°, 2E+4 iter\nupper surface',
        bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3}, fontsize = 6)
    ax2.text(2.0, 0.8, 'Goland wing, AR=6.7, Inviscid\nM=0.10, α=5.00°, 2E+4 iter\nlower surface',
        bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3}, fontsize = 6)
    #ax.text(2.0, 0.8, 'Goland wing, AR=6.7, Inviscid\nM=0.85, α=2.00°, 3E+5 iter\nlower surface',
    #    bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3}, fontsize = 6)
    plt.tight_layout()


    plt.show()
    return()


def plotTipCPContours(resultsdir):
    """
    y-x plot of projected surface pressure contours
    """
    df = getYSlice(resultsdir, 3.4, 0.09, 1)

    #fig, ax = plt.subplots(figsize=(8.0, 2.8)) #compromise 0.7=AR
    fig, (ax,ax2) = plt.subplots(1,2,figsize=(8.0, 2.8),sharey=True) #compromise 0.7=AR
    x = df['X'].to_numpy() # freeflow point data
    y = df['Y'].to_numpy()
    z = df['Z'].to_numpy()
    cp= df['CP'].to_numpy()

    #dcc = ax.tricontour( y, x, cp, 20, linewidths=0.5, colors='k', zorder=2, alpha = 0.5) #lines
    dcp = ax.tricontourf(x, z, cp, 20,cmap='rainbow') #lines
    dcp = ax2.tricontourf(x, z, cp, 20, cmap='rainbow') #lines
    #dcb = plt.colorbar(dcp, aspect=5, shrink=0.65)
    #dcb = plt.colorbar(dcp, aspect=5)
    #ax.scatter(x,z)

    plt.subplots_adjust(
        left=0.07, bottom=0.17, right=0.97, top=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.05, 0.7])
    fig.colorbar(dcp, cax=cbar_ax)

    #ax.set_aspect("equal", adjustable='box')
    ax.set_xlabel("chordwise coordinate (x)")
    ax.set_xlim(-0.0005,0.006)
    ax2.set_xlim(0.994,1.0005)
    ax.set_ylim(-0.0007,0.0007)
    ax.set_ylabel("normal coordinate (z)")
    ax2.set_xlabel("chordwise coordinate (x)")

    #ax2.set_ylabel("normal coordinate (z)")
    #ax.set_title("Wing tip -Cp distribution")
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))


    ax.text(0.0, 0.0004, 'Goland wing, AR=6.7, Inviscid\nM=0.20, α=6.00°, 2E+4 iter\nnot to scale',
        bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3}, fontsize = 6)
    #ax2.text(2.0, 0.8, 'Goland wing, AR=6.7, Inviscid\nM=0.20, α=6.00°, 2E+4 iter\nlower surface',
    #    bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3}, fontsize = 6)
    #ax.text(2.0, 0.8, 'Goland wing, AR=6.7, Inviscid\nM=0.85, α=2.00°, 3E+5 iter\nlower surface',
    #    bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3}, fontsize = 6)
    plt.tight_layout()

    plt.subplots_adjust(
        left=0.14, bottom=0.17, right=0.86, top=0.85)
    plt.show()
    return()


def plotSurface(resultsdir, upper=1):
    """
    plots mesh surface
    """
    
    df = getZSlice(resultsdir, 0, 0, upper)

    
    mask = df["X"] <= 0.5
    df = df[mask]

    fig = plt.figure(figsize=(4.0, 2.8))
    ax = fig.add_subplot(111, projection='3d')
    #fig, ax1 = plt.subplots(figsize=(4.0, 2.8)) #compromise 0.7=AR
    #ax1.gca(projection='3d')
    
    x = df['X'].to_numpy() #freeflow point data
    y = df['Y'].to_numpy()
    z = df['Z'].to_numpy()
    points = np.array([x,y,z])

    #hull = ssp.ConvexHull(points)
    #X,Y = np.meshgrid(x,y)
    triang = tri.Triangulation(x, y)
    #ax.plot_trisurf(triang, z, cmap=plt.cm.CMRmap)

    #ax.plot_trisurf(x,y,z,cmap=plt.cm.Spectral)
    ax.plot_trisurf(triang,z)
    #ax.scatter(x,y,z,color = "k",s=3)
    #ax.plot_surface(x,y,z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.auto_scale_xyz([0, 1], [0, 3.5], [0, 1])
    """
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')
    """
    plt.grid()
    
    plt.show()
    return()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------- EXPERIMENTAL ------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def shockCompare(resultsdir,p1a,p1b,p1c,p2a,p2b,p2c):
    """
    compares points on a single streamline for transonic case
    """

    df1 = getYSlice(resultsdir,0,0.01,0)
    df2 = getYSlice(resultsdir,0,0.01,0)

    mask = df1["X"] >= (p1a)
    df1 = df1[mask]
    mask = df1["X"] <= (p1b)
    df1 = df1[mask]

    mask = df2["X"] >= (p2a)
    df2 = df2[mask]
    mask = df2["X"] <= (p2b)
    df2 = df2[mask]

    df1.sort_values("X", ascending=False)
    df2.sort_values("X", ascending=False)
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    del df1['index']
    del df2['index']

    print("\npoint 1:")    
    df1.iloc[(df1['Z']-p1c).abs().argsort()[:1]]
    x1 = df1['X'].values[0]
    z1 = df1['Z'].values[0]
    r1 = df1['R'].values[0]
    u1 = df1['U'].values[0]
    w1 = df1['W'].values[0]
    p1 = df1['P'].values[0]
    U1 = (u1**2+w1**2)**0.5
    mdot1 = r1*U1
    ptot1 = p1+mdot1*U1
    E1 = p1/(gamma-1)/r1+0.5+U1
    H1 = E1 + p1/r1
    print("x: "+str(x1)+"    z: "+str(z1))
    print("p: "+str(p1)+"    vel: "+str(U1)+"    dens: "+str(r1))
    print("mass continuity: "+str(mdot1))
    print("stagn. Pressure: "+str(ptot1))
    print("energy:          "+str(E1))
    print("enthalpy:        "+str(H1))

    print("\npoint 2:")
    df2.iloc[(df2['Z']-p2c).abs().argsort()[:1]]
    x2 = df2['X'].values[0]
    z2 = df2['Z'].values[0]
    r2 = df2['R'].values[0]
    u2 = df2['U'].values[0]
    w2 = df2['W'].values[0]
    p2 = df2['P'].values[0]
    U2 = (u2**2+w2**2)**0.5
    mdot2 = r2*U2
    ptot2 = p2+mdot2*U2
    E2 = p2/(gamma-1)/r2+0.5+U2
    H2 = E2 + p2/r2
    print("x: "+str(x2)+"    z: "+str(z2))
    print("p: "+str(p2)+"    vel: "+str(U2)+"    dens: "+str(r2))
    print("mass continuity: "+str(mdot2))
    print("stagn. Pressure: "+str(ptot2))
    print("energy:          "+str(E2))
    print("enthalpy:        "+str(H2))

    print("\n--differences--")
    print("mass continuity: "+str(abs(mdot2-mdot1)))
    print("stagn. Pressure: "+str(abs(ptot2-ptot1)))
    print("enthalpy:        "+str(abs(H2-H1)))
    return()


def plotCp(resultsdir,coefficient = 0):
    """
    Plots a surface CP over x/c
    optional: add experiment data or other
    """

    # importing and converting flow dataframe
    if three_dim:
        coord = float(input("type slice y-coordinate: "))
        #coord = 0.05        # lazy option
        tolerance = float(input("type tolerance: "))
        #tolerance = 0.01    # lazy option
    else:
        coord = 0
        tolerance = 0.01
        
    #checkGlobalDf(resultsdir)
    df = getYSlice(resultsdir, coord, tolerance, True)
    #print(df)
    # splitting at Z=0 and sorting in X-ascending order
    mask = df["Z"] >= 0
    dfu = df[mask]
    dfl = df[~mask]
  
    dfu = dfu.sort_values("X", ascending=True)
    print(dfu)
    dfl = dfl.sort_values("X", ascending=True)
    xu = dfu.T.to_numpy()[0]    # x upper
    print(xu)
    xl = dfl.T.to_numpy()[0]    # x lower
    cpu = dfu.T.to_numpy()[8]   #cp upper
    print(cpu)
    cpl = dfl.T.to_numpy()[8]

    # plot the whole thing
    fig, (ax,ax1) = plt.subplots(1,2,figsize=(8.0, 2.8)) #compromise 0.7=AR
    #fig, ax = plt.subplots(figsize=(8.0, 3.8)) #compromise x=AR

    print(max(cpu))
    ax.plot(xu, cpu, lw=1, label= "Glasgow Meshless", Color = "k")
    ax.plot(xl, cpl, lw=1, label= "", Color = "k", linestyle = '--')
    """
    ----SECTION BELOW TO ADD EXPERIMENTAL DATA AND DIGITISED DATA----
    """
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
    print(max(cpuc))
    print(max(expu))
    
    ax.scatter(xpu, expu, s= 8, label= "AGARD", Color = "r")
    ax.scatter(xpl, expl, s= 8, label= "", Color = "r")
    ax.plot(xc ,cpuc, lw=1, label= "M.K. Singh et al", Color = "b")
    ax.plot(xc, cpul, lw=1, label= "", Color = "b", linestyle = "--")
    ax.text(0.5, -0.8, 'NACA 0012, M=1.20,\nα=7.00°, CFL 0.5,\n2.5E+05 iterations',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    """
    ----SECTION ABOVE TO ADD EXPERIMENTAL DATA AND DIGITISED DATA----
    """

    #ax.text(0.2, -0.8, 'Goland wing, AR=6.7, Inviscid\nM=0.85, α=2.00°, not converged\nslice at midspan',
    #    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3})
    
    ax.set_ylabel("-Cp")
    ax.set_xlabel("x/c")
    #ax.legend(loc = 2, fontsize = 8)
    ax.legend(loc = 2)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlim(-0.00,1)
    ax.set_ylim(-1.2,1.6)
 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # adding half aerofoil shape
    xs = df['X'].values.tolist() # freeflow point data
    zs = df['Z'].values.tolist()

    ax2 = ax.twinx()
    ax2.set_xlim(0,1)
    #ax2.set_ylim(0,0.62) #full
    ax2.set_ylim(0,0.70) #compromise
    #ax2.set_ylim(0,0.315) #half

    x1, z1 = orderCoord(xs,zs)    # order surface points
    poly = Polygon(np.column_stack([x1,z1]), color = "k", fill =True, lw = 1, alpha=0.5, zorder=4)
    ax2.add_patch(poly)
    ax2.plot(x1,z1, color = "k",lw = 1, zorder=4)
    ax2.get_yaxis().set_ticks([])

    #plt.subplots_adjust(
    #    left=0.10, bottom=0.11, right=0.98, top=0.97) #full
    #plt.subplots_adjust(
    #    left=0.15, bottom=0.15, right=0.97, top=0.97)
    #title = resultsdir + "/CPx.svg"
    #plt.savefig(title,bbox_inches='tight')
    

    """
    plots the pressure field around the aerofoil
    coefficient = True: Plots pressure coefficient (experimental)
    """
    
    # importing and converting flow dataframe
    if three_dim:
        #coord = float(input("type slice y-coordinate: "))
        coord = 0.05        # lazy option
        #tolerance = float(input("type tolerance: "))
        tolerance = 0.01    # lazy option
    else:
        coord = 0
        tolerance = 0.01
        
    #checkGlobalDf(resultsdir)
    flowdf = getYSlice(resultsdir, coord, tolerance, False)
    surfdf = getYSlice(resultsdir, coord, tolerance, True)

    x = flowdf['X'].values.tolist() # freeflow point data
    z = flowdf['Z'].values.tolist()
    p = flowdf['P'].values.tolist()
    xs = surfdf['X'].values.tolist() # surface coordinates of aerofoil
    zs = surfdf['Z'].values.tolist()

    # calculating pressure coefficients (experimental)
    if coefficient:
        cp = []

        #dataframe to find pressure at far field point
        params = flowdf.iloc[(flowdf['X']+48).abs().argsort()[:1]]
        pfree = params['P'].values[0]
    
        # determining free flow pressure
        for idx in range(len(p)):
            vext = 1
            dens = 1
            ccp = (p[idx]-pfree)*2/(dens*(vext*vext))
            cp.append(ccp)

    # plotting the pressure (coefficient)
    #fig, ax1 = plt.subplots(1,2,2,figsize=(4,2.8))
    ax1.scatter(x,z, s=12, color='k', marker = "x",zorder = 7)
    
    if coefficient:
        dcp = ax1.tricontour(x, z, cp, 20, linewidths=0.5, colors='k', zorder=2) #lines
        #dcp = plt.tricontour(x, z, cp, 20, linewidths=0.5, zorder=2, cmap='RdGy') #lines
        dp = ax1.tricontourf(x, z, cp, 100, cmap=cm.rainbow, alpha = 0.3, zorder=1) #fill
    else:
        dcp = ax1.tricontour(x, z, p, 30, linewidths=0.5, colors='k', zorder=2) #lines
        dp = ax1.tricontourf(x, z, p, 112, cmap=cm.rainbow, zorder=1) #fill
    #plt.colorbar(dp)
    
    # plot aerofoil shape 
    x1, z1 = orderCoord(xs,zs)    # order surface points
    poly = Polygon(np.column_stack([x1,z1]), color = "w", fill =True, lw = 1, zorder=4)
    ax1.add_patch(poly)
    ax1.plot(x1,z1, color = "k",lw = 1, zorder=4)

    #fig.set_tight_layout(True)
    #ax1.text(-0.45, 0.58, 'Cp contours',
    #    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 3}) 

    ax1.set_aspect("equal", adjustable='box')
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")

    ax1.set_ylim(-0.7,0.7)
    ax1.set_xlim(-0.5,1.5)
    """
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    """

    #plt.subplots_adjust(
    #    left=0.16, bottom=0.15, right=0.97, top=0.97)
    plt.subplots_adjust(
        left=0.08, bottom=0.16, right=0.97, top=0.97)
    
    plt.clabel(dcp, inline=True, fontsize=8,rightside_up=True, manual=True)
    
    plt.show()

    return()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------- FUNCTION AND PLOT EXECUTION ------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

#plotDens(resultsdir)
#plotCloud(resultsdir)
#plotLiftslope()
#plotLiftslope2()
#printCoeffFromLoads(resultsdir)
plotLoadsConvergence(resultsdir)
plotReport(resultsdir, viscid)
#plotStreamlines(resultsdir, coefficient = 0)
#shockCompare(resultsdir,0.4215,0.422,0.197,0.4692,0.4694,0.1942)
#plotPressureContours(resultsdir, coefficient = 0)
#plotMachContours(resultsdir)
#plotCp(resultsdir,1)
#plotCpx(resultsdir)
#getCurl(resultsdir)
if three_dim:
    #plotTipCPContours(resultsdir)
    #plotMultiCpx(resultsdir, slices2, skip = 5)
    #coefsFromSlices(resultsdir, slices2)
    plotSliceCoefs(resultsdir, slices2)
    #plotVectors(resultsdir, 5, 0.5)
    plotXYCPContours(resultsdir)
    #plotSurface(resultsdir,1)


# NEXT UP:
# SURFACE PRESSURE 3D PLOT -- quite hard, but might be doable in matlab
# ENTROPY -- so far don't know how...
# OTHER 3D PLOTS --
# OTHER PLOTS
