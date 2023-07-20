
# Solution of 2D Stokes continuity and advection equations 
# with finite differences and marker-in-cell technique
# on a regular grid using pressure-velocity formulation
# for a medium with variable viscosity
# using of the first order accurate in space and time 
# marker advection scheme


import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numpy import unravel_index as u_i

from numpy import unravel_index,ravel_multi_index
#np.ravel_multi_index(arr, (7,6), order='F')

from matplotlib import pyplot as plt
import time

from joblib import Parallel, delayed
from joblib import Memory
import multiprocessing


num_cores = multiprocessing.cpu_count() #number of cores

def mm12rc(mm1):
    # Gives rc indices
    return u_i(mm1, (mynum, mxnum), order='F')


def interpolate_mark_nodes(mn,rho,wtnodes,etas,wtetas,etan,wtetan):
    #time.sleep(1);
    #for mm1 in range(1,marknum):
    #global xn,yn,rho,wtnodes,etas,wtetas,etan,wtetan
    # Check markers inside the grid
    if (MX[mn]>=gridx[0] and MX[mn]<=gridx[xnum-1] and MY[mn]>=gridy[0] and MY[mn]<=gridy[ynum-1]): 

        #  xn    rho(xn,yn)--------------------rho(xn+1,yn)
        #           ?           **                  ?
        #           ?           ?                  ?
        #           ?          dy                  ?
        #           ?           ?                  ?
        #           ?           v                  ?
        #           ?<----dx--->o Mrho(xm,ym)       ?
        #           ?                              ?
        #           ?                              ?
        #  xn+1  rho(xn,yn+1)-------------------rho(xn+1,yn+1)
        #
        #
        # Define indexes for upper left node in the cell where the marker is
        # !!! SUBTRACT 0.5 since int16(0.5)=1
        xn=int(MX[mn]/xstp-0.5) #+1;
        yn=int(MY[mn]/ystp-0.5) #+1;
        
        
        if (xn<0):
            xn=0;
        
        if (xn>xnum-2):
            xn=xnum-2;
        
        if (yn<0):
            yn=0;
        
        if (yn>ynum-2):
            yn=ynum-2;
        
        # Define normalized distances from marker to the upper left node;
        dx=(MX[mn]-gridx[xn])/xstp;
        dy=(MY[mn]-gridy[yn])/ystp;

        # Define material density and viscosity from marker type
        MRHOCUR=MRHO[MI[mn]]; # Density
        METACUR=MFLOW[MI[mn]]; # Viscosity


        # Add density to 4 surrounding basic nodes
        # Upper-Left node
        rho[yn,xn]=rho[yn,xn]+(1.0-dx)*(1.0-dy)*MRHOCUR;
        wtnodes[yn,xn]=wtnodes[yn,xn]+(1.0-dx)*(1.0-dy);
        # Lower-Left node
        rho[yn+1,xn]=rho[yn+1,xn]+(1.0-dx)*dy*MRHOCUR;
        wtnodes[yn+1,xn]=wtnodes[yn+1,xn]+(1.0-dx)*dy;
        # Upper-Right node
        rho[yn,xn+1]=rho[yn,xn+1]+dx*(1.0-dy)*MRHOCUR;
        wtnodes[yn,xn+1]=wtnodes[yn,xn+1]+dx*(1.0-dy);
        # Lower-Right node
        rho[yn+1,xn+1]=rho[yn+1,xn+1]+dx*dy*MRHOCUR;
        wtnodes[yn+1,xn+1]=wtnodes[yn+1,xn+1]+dx*dy;

        # Add viscosity etas() to 4 surrounding basic nodes
        # only using markers located at <=0.5 gridstep distances from nodes
        # Upper-Left node
        if(dx<=0.5 and dy<=0.5):
            etas[yn,xn]=etas[yn,xn]+(1.0-dx)*(1.0-dy)*METACUR;
            wtetas[yn,xn]=wtetas[yn,xn]+(1.0-dx)*(1.0-dy);
        
        # Lower-Left node
        if(dx<=0.5 and dy>=0.5):
            etas[yn+1,xn]=etas[yn+1,xn]+(1.0-dx)*dy*METACUR;
            wtetas[yn+1,xn]=wtetas[yn+1,xn]+(1.0-dx)*dy;
        
        # Upper-Right node
        if(dx>=0.5 and dy<=0.5):
            etas[yn,xn+1]=etas[yn,xn+1]+dx*(1.0-dy)*METACUR;
            wtetas[yn,xn+1]=wtetas[yn,xn+1]+dx*(1.0-dy);
        
        # Lower-Right node
        if(dx>=0.5 and dy>=0.5):
            etas[yn+1,xn+1]=etas[yn+1,xn+1]+dx*dy*METACUR;
            wtetas[yn+1,xn+1]=wtetas[yn+1,xn+1]+dx*dy;
        
        # Add viscosity etan() to the center of current cell (pressure node)
        etan[yn+1,xn+1]=etan[yn+1,xn+1]+(1.0-abs(0.5-dx))*(1.0-abs(0.5-dy))*METACUR;
        wtetan[yn+1,xn+1]=wtetan[yn+1,xn+1]+(1.0-abs(0.5-dx))*(1.0-abs(0.5-dy));


def marker_cycle(mm1):
    # Marker cycle (displacement of markers according to their velocity)
    #for mm1 in range(1,marknum):

    # Check markers inside the grid
    if (MX[mm1]>=gridx[0] and MX[mm1]<=gridx[xnum-1] and MY[mm1]>=gridy[0] and MY[mm1]<=gridy[ynum-1]): 

        #  xn    V(xn,yn)--------------------V(xn+1,yn)
        #           ?           **                  ?
        #           ?           ?                  ?
        #           ?          dy                  ?
        #           ?           ?                  ?
        #           ?           v                  ?
        #           ?<----dx--->o Mrho(xm,ym)      ?
        #           ?                              ?
        #           ?                              ?
        #  xn+1  V(xn,yn+1)-------------------V(xn+1,yn+1)

        # Define indexes for upper left node in the Vx-cell where the marker is
        # Vx-cells are displaced downward for 1/2 of vertical gridstep
        xn=int(MX[mm1]/xstp-0.5);#+1;
        yn=int(MY[mm1]/ystp-0.5);#+1;
        
       
        # Check indexes:
        # vertical index for upper left Vx-node must be between 1 and ynum-2
        if (xn<0):
            xn=0;
        
        if (xn>xnum-2):
            xn=xnum-2;
        
        if (yn<0):
            yn=0;
        
        if (yn>ynum-2):
            yn=ynum-2;
        
        # Define and check normalized distances from marker to the upper left Vx-node;
        dx=(MX[mm1]-gridx[xn])/xstp;
        dy=(MY[mm1]-gridcy[yn])/ystp;

        # Calculate Marker velocity from four surrounding Vx nodes
        vxm=0;
        vxm=vxm+(1.0-dx)*(1.0-dy)*vx[yn,xn];
        vxm=vxm+(1.0-dx)*dy*vx[yn+1,xn];
        vxm=vxm+dx*(1.0-dy)*vx[yn,xn+1];
        vxm=vxm+dx*dy*vx[yn+1,xn+1];

        # Define indexes for upper left node in the Vy-cell where the marker is
        xn=int(MX[mm1]/xstp-0.5);#+1;
        yn=int(MY[mm1]/ystp-0.5);#+1;
        
        # Check indexes:
        # horizontal index for upper left Vy-node must be between 1 and xnum-2
        if (xn<0):
            xn=0;
        
        if (xn>xnum-2):
            xn=xnum-2;
        
        if (yn<0):
            yn=0;
        
        if (yn>ynum-2):
            yn=ynum-2;
        
        # Define and check normalized distances from marker to the upper left VY-node;
        dx=(MX[mm1]-gridcx[xn])/xstp;
        dy=(MY[mm1]-gridy[yn])/ystp;

        # Calculate Marker velocity from four surrounding nodes
        vym=0;
        vym=vym+(1.0-dx)*(1.0-dy)*vy[yn,xn];
        vym=vym+(1.0-dx)*dy*vy[yn+1,xn];
        vym=vym+dx*(1.0-dy)*vy[yn,xn+1];
        vym=vym+dx*dy*vy[yn+1,xn+1];

        # Displacing Marker according to its velocity
        MX[mm1]=MX[mm1]+timestep*vxm;
        MY[mm1]=MY[mm1]+timestep*vym;

location = 'cache_dir'
#memory = Memory(location, verbose=0)
#interpolate_mark_nodes_cached = memory.cache(interpolate_mark_nodes)

starttime=time.time();

# Numerical model parameters
# Model size, m
xsize   =   1000000; # Horizontal
ysize   =   1500000; # Vertical

# Numbers of nodes
xnum    =   31; # Horizontal
ynum    =   21; # Vertical
# Grid step
xstp    =   xsize/(xnum-1); # Horizontal
ystp    =   ysize/(ynum-1); # Vertical

# Model viscosity
etaleft =   1e+20; # Left Layer
etaright=   1e+22; # Right Layer

# Pressure condition in one cell (i==2 and j==3)
p0cell  =   0;

# Gravity acceleration directed downward
gy      =   10; # m/s**2

# Making vectors for nodal points positions (basic nodes)
x       =   np.arange(0,xsize+xstp,xstp); # Horizontal
y       =   np.arange(0,ysize+ystp,ystp); # Vertical

# Making vectors for cell centers positions (staggered nodes)
xc      =    np.arange(xstp/2,xsize+xstp/2,xstp); # Horizontal
yc      =    np.arange(ystp/2,ysize+ystp/2,ystp); # Vertical


# Maximal timestep, s
timemax=1e+6*365.25*24*3600; #1 Myr
# Maximal marker displacement step, number of gridsteps
markmax=0.5;
# Amount of timesteps
stepmax=50;

# Material properties
# Viscosity, Pa s
MFLOW=[1e+20, # 1 = Left Layer
    1e+22]; # 2 = Right Layer
# Density, kg/m**3
MRHO=[3200, # 1 = Weak Medium
    3300]; # 2 = Right Layer


# Pressure condition in one cell (i==2 and j==3)
p0cell=0;

# Gravity acceleration directed downward
gy=10; # m/s**2

# Making vectors for nodal points positions (basic nodes)
gridx=np.arange(0,xsize+xstp,xstp); # Horizontal
gridy=np.arange(0,ysize+ystp,ystp); # Vertical

# Making vectors for cell centers positions (staggered nodes)
gridcx=np.arange(xstp/2,xsize+xstp/2,xstp); # Horizontal
gridcy=np.arange(ystp/2,ysize+ystp/2,ystp); # Vertical

#gridcx=xstp/2:xstp:xsize-xstp/2; # Horizontal
#gridcy=ystp/2:ystp:ysize-ystp/2; # Vertical

# Defining number of markers and steps between them in the horizontal and vertical direction
mxnum=200; #total number of markers in horizontal direction
mynum=300;  #total number of markers in vertical direction
mxstep=xsize/mxnum; #step between markers in horizontal direction   
mystep=ysize/mynum; #step between markers in vertical direction

# Creating markers arrays
MX=np.zeros(mynum*mxnum,dtype=np.float64);   # X coordinate, m
MY=np.zeros(mynum*mxnum,dtype=np.float64);   # Y coordinate, m
MI=np.zeros(mynum*mxnum,int);   # Type

# Defining intial position of markers
# Defining lithological structure of the model
# Marker counter
mm1=0;
for xm in range(1,mxnum+1):
    for ym in range(1,mynum+1):
        
         
        # Coordinates with small random displacement
        #MX[mm1]=xm*mxstep-mxstep/2+(np.random.rand()-0.5)*mxstep;
        #MY[mm1]=ym*mystep-mystep/2+(np.random.rand()-0.5)*mystep;
        MX[mm1]=xm*mxstep-mxstep/2+(np.random.rand()-0.5)*mxstep;
        MY[mm1]=ym*mystep-mystep/2+(np.random.rand()-0.5)*mystep;
        
        # Defining initial rock distribution (marker type)
        if(MX[mm1]<xsize/2):
            # 1 = Left layer
            MI[mm1]=0;
        else:
            # 2 = Right layer
            MI[mm1]=1;
        
        # Update marker counter:
        mm1+=1;
            
# Save Number of markers
marknum=mm1

# Density, viscosity arrays
etas = np.zeros([ynum,xnum],float);    # Viscosity for shear stress
etan = np.zeros([ynum,xnum],float);    # Viscosity for normal stress
rho = np.zeros([ynum,xnum],float);     # Density

# Initial time, s
timesum=0;


alert=0;

# Main Time cycle
for ntimestep in range(1,stepmax+1):
#for ntimestep in range(1,2):    
    

    # Backup transport properties arrays
    etas0 = etas;
    etan0 = etan;
    rho0 = rho;
    # Clear transport properties arrays
    etas = np.zeros([ynum,xnum]);   # Viscosity for shear stress
    etan = np.zeros([ynum,xnum]);   # Viscosity for normal stress
    rho = np.zeros([ynum,xnum]);    # Density
    # Clear wights for basic nodes
    wtnodes=np.zeros([ynum,xnum]);
    # Clear wights for etas
    wtetas=np.zeros([ynum,xnum]);
    # Clear wights for etan
    wtetan=np.zeros([ynum,xnum]);

    # Interpolating parameters from markers to nodes
    #interpolate_mark_nodes()
    
    Parallel(n_jobs=8,verbose=0, backend="threading",pre_dispatch='1.5*n_jobs')(delayed(interpolate_mark_nodes)(n,rho,wtnodes,etas,wtetas,etan,wtetan) 
                                                    for n in range(1,marknum))
    
    
    
    # Computing  Viscosity, density, rock type for nodal points
    for i in range(0,ynum):
        for j in range(0,xnum):
            # Density
            if (wtnodes[i,j]!=0):
                # Compute new value interpolated from markers
                rho[i,j]=rho[i,j]/wtnodes[i,j];
            else:
                # If no new value is interpolated from markers old value is used
                rho[i,j]=rho0[i,j];
            
            # Viscosity etas() (basic nodes)
            if (wtetas[i,j]!=0):
                # Compute new value interpolated from markers
                etas[i,j]=etas[i,j]/wtetas[i,j];
            else:
                # If no new value is interpolated from markers old value is used
                etas[i,j]=etas0[i,j];
            
            # Viscosity etan() (pressure cells)
            if (wtetan[i,j]!=0):
                # Compute new value interpolated from markers
                etan[i,j]=etan[i,j]/wtetan[i,j];
            else:
                # If no new value is interpolated from markers old value is used
                etan[i,j]=etan0[i,j];





    # Matrix of coefficients initialization
    #L=csr_matrix((xnum*ynum*3,xnum*ynum*3),dtype=np.float64);
    L=np.zeros((xnum*ynum*3,xnum*ynum*3),dtype=np.float64);
    # Vector of right part initialization
    R=np.zeros(xnum*ynum*3);

    # Computing Kcont and Kbond coefficients 
    etamin=min(etas.flatten()); # min viscosity in the model
    kcont=2*etamin/(xstp+ystp);
    kbond=4*etamin/(xstp+ystp)**2;

    # Solving x-Stokes, y-Stokes and continuity equations
    
    # x-Stokes: ETA(d2vx/dx2+d2vx/dy2)-dP/dx=0
    # y-Stokes: ETA(d2vy/dx2+d2vy/dy2)-dP/dy=gy*RHO
    # continuity: dvx/dx+dvy/dy=0
    
    # Composing matrix of coefficients L()
    # and vector (column) of right parts R()
    # Boundary conditions: free slip
    # Process all Grid points
    for i in range(0,ynum):
      for j in range(0,xnum):
        # Global index for P, vx, vy in the current node
        
        inp=((j)*ynum+i)*3; # P (j-1)   #это тоже работает 
        invx=inp+1;
        invy=inp+2;

            
        # Continuity equation
        # Ghost pressure unknowns (i=1, j=1) and boundary nodes (4 corners + one cell)
        #if(i==0 or j==0 or (i==1 and j==1) or (i==1 and j==xnum-1) or (i==ynum-1 and j==1)\
        #   or (i==ynum-1 and j==xnum-1) or (i==1 and j==2)):
        if i==0 or j==0 or (i==1 and j==1) or (i==1 and j==xnum-1) or \
            (i==ynum-1 and j==1) or (i==ynum-1 and j==xnum-1) or (i==1 and j==2):
            # Ghost pressure unknowns (i=1, j=1): P(i,j)=0
            if(i==0 or j==0):
                L[inp,inp]          =   1*kbond;            # Coefficient for P(i,j)
                R[inp]            =   0;                  # Right part
            
            # Upper and lower left corners dP/dx=0 => P(i,j)-P(i,j+1)=0
            if (i==1 and j==1) or (i==ynum-1 and j==1):
                L[inp,inp]          =   1*kbond;            # Coefficient for P(i,j) 
                L[inp,inp+ynum*3]   =   -1*kbond;           # Coefficient for P(i,j+1)
                R[inp]            =   0;                  # Right part
            
            # Upper and lower right corners dP/dx=0 => P(i,j)-P(i,j-1)=0
            if((i==1 and j==xnum-1) or (i==ynum-1 and j==xnum-1)):
                L[inp,inp]          =   1*kbond;            # Coefficient for P(i,j) 
                L[inp,inp-ynum*3]   =   -1*kbond;           # Coefficient for P(i,j-1)
                R[inp]            =   0;                  # Right part
            
            # One cell 
            if [i==1 and j==2]:
                L[inp,inp]          =   1*kbond;            # Coefficient for P(i,j)
                R[inp]            =   p0cell;             # Right part
            
            
           
            
        #Internal nodes: dvx/dx+dvy/dy=0
        else:
            #dvx/dx=(vx(i-1,j)-vx(i-1,j-1))/dx
            L[inp,invx-3]           =   kcont/xstp;         # Coefficient for vx(i-1,j) 
            L[inp,invx-3-ynum*3]    =   -kcont/xstp;        # Coefficient for vx(i-1,j-1) 
            #dvy/dy=(vy(i,j-1)-vy(i-1,j-1))/dy
            L[inp,invy-ynum*3]      =   kcont/ystp;         # Coefficient for vy(i,j-1) 
            L[inp,invy-3-ynum*3]    =   -kcont/ystp;        # Coefficient for vy(i-1,j-1) 
            # Right part:0
            R[inp]=0;
            

        # x-Stokes equation
        # Ghost vx unknowns (i=ynum) and boundary nodes (i=1, i=ynum-1, j=1, j=xnum)
        if(i==0 or i==ynum-2 or i==ynum-1 or j==0 or j==xnum-1):
            # Ghost vx unknowns (i=ynum: vx(i,j)=0
            if(i==ynum-1):
                L[invx,invx]        =   1*kbond; # Coefficient for vx(i,j)
                R[invx]           =   0; # Right part
            
            # Left and Right boundaries (j=1, j=xnum) 
            if((j==0 or j==xnum-1) and i<ynum-1):
                # Free slip, No slip: vx(i,j)=0
                L[invx,invx]        =   1*kbond; # Coefficient for vx(i,j)
                R[invx]           =   0; # Right part
            
            # Upper boundary, iner points (i=1, 1<j<xnum)
            if(i==0 and j>0 and j<xnum-1):
                # Free slip dvx/dy=0: vx(i,j)-vx(i+1,j)=0
                L[invx,invx]        =   1*kbond; # Coefficient for vx(i,j)
                L[invx,invx+3]      =   -1*kbond; # Coefficient for vx(i+1,j)
                R[invx]=0; # Right part
    #             # No slip vx=0: vx(i,j)-1/3*vx(i+1,j)=0
    #             L(invx,invx)=1*kbond; # Coefficient for vx(i,j)
    #             L(invx,invx+3)=-1/3*kbond; # Coefficient for vx(i+1,j)
    #             R(invx,1)=0; # Right part
            
            # Lower boundary, iner points (i=ynum-1, 1<j<xnum)
            if(i==ynum-2 and j>0 and j<xnum-1):
                # Free slip dvx/dy=0: vx(i,j)-vx(i-1,j)=0
                L[invx,invx]        =   1*kbond; # Coefficient for vx(i,j)
                L[invx,invx-3]      =   -1*kbond; # Coefficient for vx(i-1,j)
                R[invx]=0; # Right part

            
                
        #Internal nodes: dSxx/dx+dSxy/dy-dP/dx=0
        else:
            #dSxx/dx=2*etan(i+1,j+1)*(vx(i,j+1)-vx(i,j))/dx**2-2*etan(i+1,j)*(vx(i,j)-vx(i,j-1))/dx**2
            L[invx,invx+ynum*3]     =   2*etan[i+1,j+1]/xstp**2;                         # Coefficient for vx(i,j+1)
            L[invx,invx-ynum*3]     =   2*etan[i+1,j]/xstp**2;                           # Coefficient for vx(i,j-1)
            L[invx,invx]            =   -2*etan[i+1,j+1]/xstp**2-2*etan[i+1,j]/xstp**2;   # Coefficient for vx(i,j)

            #dSxy/dy=etas(i+1,j)*((vx(i+1,j)-vx(i,j))/dy**2+(vy(i+1,j)-vy(i+1,j-1))/dx/dy)-
            #         -etas(i,j)*((vx(i,j)-vx(i-1,j))/dy**2+(vy(i,j)-vy(i,j-1))/dx/dy)-
            L[invx,invx+3]          =   etas[i+1,j]/ystp**2;                             # Coefficient for vx(i+1,j)
            L[invx,invx-3]          =   etas[i,j]/ystp**2;                               # Coefficient for vx(i-1,j)
            L[invx,invx]            =   L[invx,invx]-etas[i+1,j]/ystp**2-etas[i,j]/ystp**2; # ADD coefficient for vx(i,j)
            L[invx,invy+3]          =   etas[i+1,j]/xstp/ystp;                          # Coefficient for vy(i+1,j)
            L[invx,invy+3-ynum*3]   =   -etas[i+1,j]/xstp/ystp;                         # Coefficient for vy(i+1,j-1)
            L[invx,invy]            =   -etas[i,j]/xstp/ystp;                           # Coefficient for vy(i,j)
            L[invx,invy-ynum*3]     =   etas[i,j]/xstp/ystp;                            # Coefficient for vy(i,j-1)
            # -dP/dx=(P(i+1,j)-P(i+1,j+1))/dx
            L[invx,inp+3]           =   kcont/xstp;                                     # Coefficient for P(i+1,j)
            L[invx,inp+3+ynum*3]    =   -kcont/xstp;                                    # Coefficient for P(i+1,j+1)
            # Right part:0
            R[invx]               =   0;
        
                
        # y-Stokes equation
        # Ghost vy unknowns (j=xnum) and boundary nodes (i=1, i=ynum, j=1, j=xnum-1)
        if(i==0 or i==ynum-1 or j==0 or j==xnum-2 or j==xnum-1):
            # Ghost vy unknowns (j=xnum: vy(i,j)=0
            if(j==xnum-1):
                L[invy,invy]        =   1*kbond;                                # Coefficient for vy(i,j)
                R[invy]           =   0;
            
            # Upper and lower boundaries (i=1, i=ynum) 
            if((i==0 or i==ynum-1) and j<xnum-1):
                # Free slip, No slip: vy(i,j)=0
                L[invy,invy]        =   1*kbond;                                # Coefficient for vy(i,j)
                R[invy]           =   0;
            
            # Left boundary, iner points (j=1, 1<i<ynum)
            if(j==0 and i>0 and i<ynum-1):
                # Free slip dvy/dx=0: vy(i,j)-vy(i,j+1)=0
                L[invy,invy]        =   1*kbond;                                # Coefficient for vy(i,j)
                L[invy,invy+ynum*3] =   -1*kbond;                               # Coefficient for vy(i,j+1)
                R[invy]           =   0;
    #             # No slip vy=0: vy(i,j)-1/3*vy(i,j+1)=0
    #             L(invy,invy)=1*kbond; # Coefficient for vy(i,j)
    #             L(invy,invy+ynum*3)=-1/3*kbond; # Coefficient for vy(i,j+1)
    #             R(invy,1)=0;
            
            # Right boundary, iner points (j=xnum-1, 1<i<ynum)
            if(j==xnum-2 and i>0 and i<ynum-1):
                #print('applied')
                # Free slip dvy/dx=0: vy(i,j)-vy(i,j-1)=0
                L[invy,invy]        =   1*kbond;                                # Coefficient for vy(i,j)
                L[invy,invy-ynum*3] =  -1*kbond;                                # Coefficient for vy(i,j-1)
                R[invy]           =   0;
            
            
    
        #Internal nodes: dSyy/dy+dSxy/dx-dP/dy=-gy*RHO
        else:
            #dSyy/dy=2*etan(i+1,j+1)*(vy(i+1,j)-vy(i,j))/dy**2-2*etan(i,j+1)*(vy(i,j)-vy(i-1,j))/dy**2
            L[invy,invy+3]          =   2*etan[i+1,j+1]/ystp**2;                 # Coefficient for vy(i+1,j)
            L[invy,invy-3]          =   2*etan[i,j+1]/ystp**2;                   # Coefficient for vy(i-1,j)
            L[invy,invy]            =   -2*etan[i+1,j+1]/ystp**2-2*etan[i,j+1]/ystp**2; # Coefficient for vy(i,j)

            #dSxy/dx=etas(i,j+1)*((vy(i,j+1)-vy(i,j))/dx**2+(vx(i,j+1)-vx(i-1,j+1))/dx/dy)-
            #         -etas(i,j)*((vy(i,j)-vy(i,j-1))/dx**2+(vx(i,j)-vx(i-1,j))/dx/dy)-
            L[invy,invy+ynum*3]     =   etas[i,j+1]/xstp**2;                     # Coefficient for vy(i,j+1)
            L[invy,invy-ynum*3]     =   etas[i,j]/xstp**2;                       # Coefficient for vy(i,j-1)
            L[invy,invy]            =   L[invy,invy]-etas[i,j+1]/xstp**2-etas[i,j]/xstp**2; # ADD coefficient for vy(i,j)
            L[invy,invx+ynum*3]     =   etas[i,j+1]/xstp/ystp;                  # Coefficient for vx(i,j+1)
            L[invy,invx+ynum*3-3]   =   -etas[i,j+1]/xstp/ystp;                 # Coefficient for vx(i-1,j+1)
            L[invy,invx]            =   -etas[i,j]/xstp/ystp;                   # Coefficient for vx(i,j)
            L[invy,invx-3]          =   etas[i,j]/xstp/ystp;                    # Coefficient for vx(i-1,j)

            # -dP/dy=(P(i,j+1)-P(i+1,j+1))/dx
            L[invy,inp+ynum*3]      =   kcont/ystp;                             # Coefficient for P(i,j+1)
            L[invy,inp+3+ynum*3]    =   -kcont/ystp;                            # Coefficient for P(i+1,j+1)
            # Right part: -RHO*gy
            R[invy]               =   -gy*(rho[i,j]+rho[i,j+1])/2;
            
            
            
    #Obtaining vector of solutions S()
    #S=L\R;
    #S=spsolve(L,R);
    S=np.linalg.solve(L,R);
    # Reload solutions to 2D p(), vx(), vy() arrays
    # Dimensions of arrays are reduced compared to the basic grid
    p=np.zeros([ynum,xnum],float);
    vy=np.zeros([ynum,xnum],float);
    vx=np.zeros([ynum,xnum],float);
    # Process all Grid points
    for i in range(0,ynum):
      for j in range(0,xnum):
        # Global index for P, vx, vy in S()
        
        #ind=ravel_multi_index([[i],[j]], (ynum,xnum), order='F')   #cell linear 
        #inp=ind*3;
        #invx=inp+1;             #index vx
        #invy=inp+2;             #index vy, page 113
        
        inp=((j)*ynum+i)*3; # P (j-1)   #это тоже работает 
        invx=inp+1;
        invy=inp+2;
        
        # P
        p[i,j]=S[inp]*kcont;
        # vx
        vx[i,j]=S[invx];
        # vy
        vy[i,j]=S[invy];
  
    # Compute vx,vy for internal nodes
    vx1=np.zeros([ynum,xnum],float);
    vy1=np.zeros([ynum,xnum],float);
    # Process internal Grid points
    for i in range(1,ynum-1):
      for j in range(1,xnum-1):
        # vx
        vx1[i,j]=(vx[i-1,j]+vx[i,j])/2;
         # vy
        vy1[i,j]=(vy[i,j-1]+vy[i,j])/2;

    #Plotting solution
    fig=plt.figure();
    ax1=plt.subplot(121);
    plt.pcolor(x/1000,y/1000,np.log10(etas));
    plt.colorbar();
    
    plt.streamplot(x[1:xnum]/1000,y[1:ynum]/1000,vx1[1:ynum,1:xnum],vy1[1:ynum,1:xnum]);
    #plt.quiver(x[1:xnum]/1000,y[1:ynum]/1000,vx1[1:ynum,1:xnum],vy1[1:ynum,1:xnum]);
    plt.gca().invert_yaxis();
    plt.title('log_{10} viscosity (color, Pa s), velocity (arrows)');
    ax2=plt.subplot(122);
    plt.pcolor(x/1000,y/1000,rho);
    plt.colorbar();
    out='Density (kg/m**3) Step='+str(ntimestep)+' Myr='+str(timesum*1e-6/(365.25*24*3600)); 
    plt.title(out);
    
    plt.streamplot(x[1:xnum]/1000,y[1:ynum]/1000,vx1[1:ynum,1:xnum],vy1[1:ynum,1:xnum]);
    #plt.quiver(x[1:xnum]/1000,y[1:ynum]/1000,vx1[1:ynum,1:xnum],vy1[1:ynum,1:xnum]);
    
    plt.gca().invert_yaxis();
    plt.title('log_{10} viscosity (color, Pa s), velocity (arrows)');
    plt.show();

    
    
    # Check maximal velocity
    vxmax=max(abs(vx.flatten()));
    vymax=max(abs(vy.flatten()));
    # Set initial value for the timestep
    timestep=timemax;
    # Check marker displacement step
    if (vxmax>0):
        if (timestep>markmax*xstp/vxmax):
            timestep=markmax*xstp/vxmax;
        
    if (vymax>0):
        if (timestep>markmax*ystp/vymax):
            timestep=markmax*ystp/vymax;
    
    # Moving Markers by velocity field with first order scheme
    # Create arrays for velocity of markers
    vxm=np.zeros([4,1],float);
    vym=np.zeros([4,1],float);
    
    
    
    # Marker cycle (in a parallel way)
    Parallel(n_jobs=8,verbose=0, backend="threading",pre_dispatch='1.5*n_jobs')(delayed(marker_cycle)(mm1) 
                                                    for mm1 in range(1,marknum))
    
    
    # Advance in time
    timesum=timesum+timestep;

endtime=time.time()-starttime;
print('total time={}'.format(endtime));

