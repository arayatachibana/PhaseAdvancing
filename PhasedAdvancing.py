
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

class PhaseAdvancer(object):
    def __init__(self,N,d,f):
        # number of speakers
        self.N =N
        # separation between speakers (m) (measered between the mid points)
        self.d =d
        # frequency(Hz)
        self.f =f
        # width of the speakers
        self.a =0.04
        # name of the model used e.g.) Gauss
        self.model = ""
        # velocity of sound in (m)
        self.vs = 300
        # wavelength(m)
        self.lamda = self.vs/f
        # wavenumber
        self.k = 2*cmath.pi/self.lamda
        # distance between the source and the screen
        self.L = 10
        # path of saved images
        self.saveLoc = "./NWC/"
    
    # function calculating the time delay for each speaker 
    # n is represents the position of the speakers
    def time_dif(self,n,time_delay):
        
        if time_delay==False:
            return 0
        # if statement used to set the time delay at the central speaker to zero 
        if n==0:
            return 0
        
        if self.model == "Gauss":
#       Gaussian time delay
            return ((1/(2*math.pi)**0.5)* math.exp(-(n*n*self.d*self.d)/(2)))*self.lamda/self.vs
#                 return ((1/(2*math.pi)**0.5)* math.exp(-(n*n*self.d*self.d/(self.lamda * self.lamda))/(2)))*self.lamda/self.vs

        elif self.model =="Quad":
            #Quadratic time delay
            return - 1/(2*cmath.pi*self.f*(((self.N-1)*self.d+self.a)/2)*(((self.N-1)*self.d+self.a)/2))*n*self.d*n*self.d + (1/(2*math.pi*self.f))
        
        else:
            # Patchwork time delay
            return (self.lamda + math.sqrt((4*self.lamda*self.lamda)-(n*self.d*n*self.d)))/(self.vs*2*math.pi)
    
    # path difference originating from the separation of the sources
    def delta(self,n,theta):
        return self.k*n*self.d*math.sin(theta)
    
    # path difference originationg form time delay
    def phi(self,n,time_delay):
        return  self.k * self.time_dif(n,time_delay) * self.vs
    
    # assigning a value to "a"
    def set_a(self, a):
        self.a = a
        return "a is set to {}".format(self.a)
    
    # assigning a value to "L"
    def set_L(self, L):
        self.L = L
        return "L is set to {}".format(self.L)
    
    # fuction plotting the distribution of time delay
    def timedif_plot(self,model):
        self.model = model
        # Create a figure of size 8x6 inches, 80 dots per inch
        plt.figure(figsize=(8, 6), dpi=80)
        # Create a new subplot from a grid of 1x1
        ax = plt.subplot(1, 1, 1)
        ax.set(ylabel="time delay (sec)",xlabel="position of the speakers")
        
        x = np.linspace(-int((self.N-1)/2),int((self.N-1)/2), self.N, endpoint=True)
        plt.xticks(x)
        
        td_List = [self.time_dif(n,10) for  n in range(-int((self.N-1)/2),int((self.N-1)/2)+1)]
        plt.grid()
        plt.plot(x, td_List, color="black",linewidth=1.0, linestyle="-")
        plt.savefig(self.saveLoc+'{}_new_td_N_{}_f_{}_d_{}.png'.format(self.model,self.N,self.f,self.d))
    
    # total amplitude from n point sources
    def Atot(self,time_delay,theta):
        atot=0
        for n in range(-int((self.N-1)/2),int((self.N-1)/2)+1):
            atot+=cmath.exp(complex(0,self.delta(n,theta))+complex(0,self.phi(n,time_delay)))
        return atot
    
    # total Intesity from n point sourses
    def I_points(self,time_delay,theta): 
        abs_val=abs(self.Atot(time_delay, theta))
        return abs_val*abs_val
    
    # Intensity from the finite width slit
    def I_singleslit(self,theta):
        
        numerator =math.sin(0.5*self.k*self.a*math.sin(theta))**2
        denominator = (0.5*self.k*self.a*math.sin(theta))**2
        
        # avoiding zero devision (sin(0)/0)->1
        if(denominator ==0):
            return 1
        else:
            return numerator/denominator
        
    # Total intensity of the effects combined 
    def Itot(self,time_delay,theta):
        return self.I_points(time_delay,theta)*self.I_singleslit(theta)
    
    # Plotting Intensity against parallel displacement on the screen
    def Intensity_plotter_s(self,model):
        self.model = model
        # Create a figure of size 8x6 inches, 80 dots per inch
        plt.figure(figsize=(8, 6), dpi=80)

        # Create a new subplot from a grid of 1x1
        ax = plt.subplot(1, 1, 1)

        theta = np.linspace(-cmath.pi/4, cmath.pi/4, 10000, endpoint=True)
        
        # converting theta to displacement
        xList = [self.L*math.tan(angle) for angle in theta]
        
        # list of intensity with time delay
        ItotList_td = [[angle,self.Itot(True,angle).real] for angle in theta]
        
        # calculating the peak value of intensity
        I_True = self.Itot(True,0)
        
        # calculating variance
        var_td= np.var(ItotList_td,axis=0)[1]/I_True
        
        Itotplot_td=np.transpose(ItotList_td)
        plt.plot(xList, Itotplot_td[1], color="green",linewidth=1.0, linestyle="-",label="with time difference")
        # adding labeling 
        ax.annotate("var1: {:.3f}".format(var_td),xy=(xList[7000],Itotplot_td[1][7000]),xycoords='data',xytext=(xList[7000]+0.1, Itotplot_td[1][7000]+self.Itot(True,0)*0.08),
                    textcoords='data', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),)

        # list of intensity without time delay
        ItotList = [[angle,self.Itot(False,angle).real] for angle in theta]
        
        # calculating the peak value of intensity
        I_False = self.Itot(False,0)
        
        #calculating variance
        var= np.var(ItotList,axis=0)[1]/I_False
        Itotplot = np.transpose(ItotList)
        plt.plot(xList, Itotplot[1], color="red",linewidth=1.0, linestyle="-",label="no time difference")
        # adding labeling 
        ax.annotate("var0: {:.3f}".format(var),xy=(xList[7000],Itotplot[1][7000]),xycoords='data',xytext=(xList[7000]+0.1, Itotplot[1][7000]),
                    textcoords='data', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),)

        # Set x limits
        plt.xlim(-self.L/2, self.L/2)
        plt.ylim(bottom=0)

        # Set x ticks
        plt.xticks(np.linspace(-self.L/2, self.L/2, 9, endpoint=True))

        ax.set(ylabel="relative magnitude of intensity",xlabel="displacement(m)")
        plt.legend(loc=0)

        # saving the plot with a given name in a given path
        plt.savefig(self.saveLoc+'{}_s_N_{}_f_{}_d_{}_L_{}.png'.format(self.model,self.N,self.f,self.d,self.L))
        plt.show()
        
    # plotting resultant intensity against theta
    def Intensity_plotter(self,model):
        self.model = model
        # Create a figure of size 8x6 inches, 80 dots per inch
        plt.figure(figsize=(8, 6), dpi=80)

        # Create a new subplot from a grid of 1x1
        ax = plt.subplot(1, 1, 1)

        theta = np.linspace(-cmath.pi/2, cmath.pi/2, 10000, endpoint=True)
        
        # list of intensity with time delay
        ItotList_td = [[angle,self.Itot(True,angle).real] for angle in theta]
        I_True = self.Itot(True,0)
        
        #calculating variance
        var_td= np.var(ItotList_td,axis=0)[1]/I_True
        Itotplot_td=np.transpose(ItotList_td)
        
        plt.plot(Itotplot_td[0], Itotplot_td[1], color="green",linewidth=1.0, linestyle="-",label="with time difference")
        #adding labelling
        ax.annotate("var1: {:.3f}".format(var_td),xy=(Itotplot_td[0][7000],Itotplot_td[1][7000]),xycoords='data',xytext=(Itotplot_td[0][7000]+0.1, Itotplot_td[1][7000]+self.Itot(True,0)*0.1),
                    textcoords='data', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),)

        # list of intensity without time delay
        ItotList = [[angle,self.Itot(False,angle).real] for angle in theta]
        I_False = self.Itot(False,0)
        #calculating variance
        var= np.var(ItotList,axis=0)[1]/I_False
        Itotplot = np.transpose(ItotList)
        
        plt.plot(Itotplot[0], Itotplot[1], color="red",linewidth=1.0, linestyle="-",label="no time difference")
        #adding labelling
        ax.annotate("var0: {:.3f}".format(var),xy=(Itotplot[0][7000],Itotplot[1][7000]),xycoords='data',xytext=(Itotplot[0][7000]+0.1, Itotplot[1][7000]),
                    textcoords='data', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),)

        # Set x limits
        plt.xlim(-math.pi/2, math.pi/2)
        plt.ylim(bottom=0)

        # Set x ticks
        plt.xticks(np.linspace(-cmath.pi/2, cmath.pi/2, 9, endpoint=True))

        ax.set(ylabel="relative magnitude of intensity",xlabel="theta(rads)")
        plt.legend(loc=0)

        # saving the plot with a given name in a given path
        plt.savefig(self.saveLoc+'{}_N_{}_f_{}_d_{}.png'.format(self.model,self.N,self.f,self.d))
        plt.show()