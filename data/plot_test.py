from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import *

import matplotlib.pyplot as plt
import numpy as np

def plotCell(cell,electrodes):
    fig = plt.figure()
    ax = Axes3D(fig)

    zips = []
    xmin = 0;xmax=1;
    ymin = 0;ymax=1;
    zmin = 0;zmax=1;
    n = 0
    for i in range(len(cell.x3d)):
        n += cell.x3d[i].size
        zips.append(zip(cell.x3d[i], cell.y3d[i], cell.z3d[i]))
        if min(cell.x3d[i]) < xmin: xmin = min(cell.x3d[i])
        if max(cell.x3d[i]) > xmax: xmax = max(cell.x3d[i])
        if min(cell.y3d[i]) < ymin: ymin = min(cell.y3d[i])
        if max(cell.y3d[i]) > ymax: ymax = max(cell.y3d[i])
        if min(cell.z3d[i]) < zmin: zmin = min(cell.z3d[i])
        if max(cell.z3d[i]) > zmax: zmax = max(cell.z3d[i])
    print n
    ax.add_collection3d(Line3DCollection(zips, color='black',))

    ax.scatter(electrodes.x, electrodes.y, electrodes.z, facecolor = 'blue', edgecolor = 'black')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_zlim([zmin,zmax])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plotMidpoints(cell,electrodes , t_ind, rang = [-np.inf,np.inf,-np.inf,np.inf,-np.inf,np.inf]):
    art3d.zalpha = lambda *args:args[0]
    fig = plt.figure()
    ax = Axes3D(fig)

    indis = np.logical_and(np.logical_and(np.logical_and(cell.xmid>=rang[0], cell.xmid<=rang[1]),
							np.logical_and(cell.ymid>=rang[2], cell.ymid<=rang[3])),
							np.logical_and(cell.zmid>=rang[4], cell.zmid<=rang[5]))

    p = ax.scatter(cell.xmid[indis], cell.ymid[indis], cell.zmid[indis], c = cell.imem[indis,t_ind],
					vmin=np.min(cell.imem[indis,t_ind]), vmax=np.max(cell.imem[indis,t_ind]), s = 50)

    plt.colorbar(p)

    indis = np.logical_and(np.logical_and(np.logical_and(electrodes.x>=rang[0], electrodes.x<=rang[1]),
							np.logical_and(electrodes.y>=rang[2], electrodes.y<=rang[3])),
							np.logical_and(electrodes.z>=rang[4], electrodes.z<=rang[5]))
    ax.scatter(electrodes.x[indis], electrodes.y[indis], electrodes.z[indis], facecolor = 'black', edgecolor = 'black', s = 50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



def saveEAP(electrode, filename = 'EAP.csv'):
	output = np.concatenate((np.reshape(electrode.x,(electrode.x.size,1)), 
							np.reshape(electrode.y,(electrode.y.size,1)),
							np.reshape(electrode.z,(electrode.z.size,1)), 
							electrode.LFP), axis = 1)

	np.savetxt(filename, output, delimiter=',')


def saveCAP(cell, filename = 'CAP.csv'):
	output = np.concatenate((np.reshape(cell.xmid,(cell.xmid.size,1)), 
							np.reshape(cell.ymid,(cell.ymid.size,1)),
							np.reshape(cell.zmid,(cell.zmid.size,1)), 
							cell.imem), axis = 1)

	np.savetxt(filename ,output, delimiter=',')







