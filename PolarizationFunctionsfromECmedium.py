#from colorsys import hsv_to_rgb

import numpy as np

from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
u = np.linspace(0, 2 * np.pi, 1000)
#print('Line9',u[0:3])

def Jones2EllipseParameters(Ex,Ey):
# Calculate ellipse orientation angle
    a = np.abs(Ex)**2 - np.abs(Ey)**2
    b = 2 * np.real(Ex * np.conjugate(Ey))
    OrientationAngle = np.arctan2(b,a)

    xhi=np.angle(np.conj(Ey)*Ex)
    sin2Epsilon=2*np.abs(Ex)*np.abs(Ey)/(np.abs(Ex)**2+np.abs(Ey)**2)*np.sin(xhi)

    S=np.sqrt(np.abs(Ex)**2+np.abs(Ey)**2)
    return OrientationAngle, sin2Epsilon, S

def TraceEllipseSpatial(Ex,Ey):
    #print('Line24')
    import numpy
    # Generate angles for the ellipse trace
    #print('Line27')
    u = numpy.linspace(0, 2 * numpy.pi, 1000)
    print('Line29',u[0:3])
    # compute MAX SPAN
    SPAN=numpy.sqrt(np.max(np.abs(Ex)**2+np.abs(Ey)**2))

    nx,ny=numpy.shape(Ex)

    fig, axes = plt.subplots(nrows=nx, ncols=ny, figsize=(ny, nx), sharex=True, sharey=True)

    if nx==1:
        axes = np.array([axes])  # Ensures axes is a 2D array
    if ny==1:
        axes = np.array([axes])  # Ensures axes is a 2D array

    for i in range(nx):
        for j in range(ny):
            a, b = Ex[i, j], Ey[i, j]
            OrientationAngle, sin2Xhi, span=Jones2EllipseParameters(a,b)
            OrientationAngle = (OrientationAngle + np.pi) / (2 * np.pi) # Normalize the orientation angle between 0 and 1

            result = np.dstack((OrientationAngle,(1-np.abs(sin2Xhi)),span/SPAN))
            color = hsv_to_rgb(result)[0][0]

            delta=np.angle(a*np.conj(b))
            # Parametric equations for the ellipse
            X=np.abs(a)*np.cos(u)
            Y=np.abs(b)*np.cos(u+delta)

            axes[i, j].plot(X/SPAN, Y/SPAN, '.k', markersize=1)
            axes[i, j].set_facecolor(color)

            axes[i, j].set_xlim(-1, 1)
            axes[i, j].set_ylim(-1, 1)
            axes[i, j].set_adjustable('box')
            axes[i, j].set_aspect('equal')
            axes[i, j].grid(False)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.show()
    
def ColorTraceEllipseSpatial(Ex,Ey):
    
    import numpy
    # Generate angles for the ellipse trace
    u = numpy.linspace(0, 2 * numpy.pi, 1000)

    # compute MAX SPAN
    SPAN=numpy.sqrt(np.max(np.abs(Ex)**2+np.abs(Ey)**2))

    nx,ny=numpy.shape(Ex)

    fig, axes = plt.subplots(nrows=nx, ncols=ny, figsize=(ny, nx), sharex=True, sharey=True)

    if nx==1:
        axes = np.array([axes])  # Ensures axes is a 2D array
    if ny==1:
        axes = np.array([axes])  # Ensures axes is a 2D array

    for i in range(nx):
        for j in range(ny):
            a, b = Ex[i, j], Ey[i, j]
            OrientationAngle, sin2Xhi, span=Jones2EllipseParameters(a,b)
            OrientationAngle = (OrientationAngle + np.pi) / (2 * np.pi) # Normalize the orientation angle between 0 and 1

            result = np.dstack((OrientationAngle,(1-np.abs(sin2Xhi)),span/SPAN))
            color = hsv_to_rgb(result)[0][0]
            #print("type(color)","color.shape",type(color))
            
            delta=np.angle(a*np.conj(b))
            # Parametric equations for the ellipse
            #X=np.abs(a)*np.cos(u)
            #Y=np.abs(b)*np.cos(u+delta)

            #axes[i, j].plot(X/SPAN, Y/SPAN, '.k', markersize=1)
            axes[i, j].set_facecolor(color)

            axes[i, j].set_xlim(-1, 1)
            axes[i, j].set_ylim(-1, 1)
            #axes[i, j].set_adjustable('box')
            axes[i, j].set_aspect('equal')
            axes[i, j].grid(False)
            #axes[i, j].set_xticks([])
            #axes[i, j].set_yticks([])
    #print("type(color)","color.shape",type(color))
    ###plt.subplots_adjust(wspace=0, hspace=0.2)
    ###plt.show()

def TestTime(Ex,Ey):
    
    import numpy
    # Generate angles for the ellipse trace
    u = numpy.linspace(0, 2 * numpy.pi, 1000)

    # compute MAX SPAN
    SPAN=numpy.sqrt(np.max(np.abs(Ex)**2+np.abs(Ey)**2))

    nx,ny=numpy.shape(Ex)

    fig, axes = plt.subplots(nrows=nx, ncols=ny, figsize=(ny, nx), sharex=True, sharey=True)

    if nx==1:
        axes = np.array([axes])  # Ensures axes is a 2D array
    if ny==1:
        axes = np.array([axes])  # Ensures axes is a 2D array

    for i in range(nx):
        for j in range(ny):
            a, b = Ex[i, j], Ey[i, j]
            OrientationAngle, sin2Xhi, span=Jones2EllipseParameters(a,b)
            OrientationAngle = (OrientationAngle + np.pi) / (2 * np.pi) # Normalize the orientation angle between 0 and 1

            result = np.dstack((OrientationAngle,(1-np.abs(sin2Xhi)),span/SPAN))
            color = hsv_to_rgb(result)[0][0]

            delta=np.angle(a*np.conj(b))
            # Parametric equations for the ellipse
            #X=np.abs(a)*np.cos(u)
            #Y=np.abs(b)*np.cos(u+delta)
        
    print("type(color),color.shape",type(color),color.shape)
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.show()
