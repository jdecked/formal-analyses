import matplotlib.pyplot as plt
from frange import frange
from pylab import savefig
from decimal import *
from numpy.random import normal
from numpy.random import randint
from scipy.stats import *
import scipy.stats as ss
from numpy import *
from scipy import *
from pylab import *
import math
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

def pop0(N0, k, t):
    # Input: N0, the initial population.
    # Input: k, the weekly net growth (m - n).
    # Input: t, the total number of weeks from initial measurement.
    
    # Slow method - 2.c.i
    """# Total number of fish initialization
    Nt = N0
    
    # Update Nt
    for i in range(0, t):
        Nt += k
    print round(Nt)"""
    
    # Typical method - 2.c.ii
    if isinstance(k, list):
        for ki in k:
            Nt = N0 + ki * t
            print round(Nt)
    else:
        Nt = N0 + k * t
        print round(Nt)

def pop1(N0, r, t):
    # Input: N0, the initial population.
    # Input: r, the net growth rate per unit time.
    # Input: t, time in units.
    
    # Init
    Nt = N0
    NtLst = []
    plt.figure()
    pop = []
    
    # Updating based on model
    '''if isinstance(r, list):
        for ri in r:
            Nt = ((float(ri) + 1) ** t) * N0
            NtLst.append(round(Nt))
            #print NtLst

        # Population vs growth rate graph
        plt.plot(r, NtLst, 'b-')
        if len(r) >= 1000:
            plt.title("Growth of Fish Population \n r = [%s, %s]" % (min(r), max(r)))
        else:
            plt.title("Growth of Fish Population \n r = %s" % (r))
        plt.xlabel("Growth Rate (r)")
        plt.ylabel("Number of Fish (N" + str(t) + ")")
        plt.ylim(ymin = -1e30)

    else:
        Nt = ((r + 1)**t) * N0
        print round(Nt)'''
    
    if isinstance(r, list):
        for ri in r:
            for i in range(0, t):
                Nt += ri * Nt
                pop.append(Nt)
            
            # Population vs time graph
            plt.plot(range(1, t+1), pop)
            pop = []
    else:
        for i in range(0, t):
            Nt += ri * Nt
            pop.append(Nt)
        plt.plot(range(1, t+1), pop, 'g--', label='pop1')
    
    '''if len(r) >= 20:
        plt.title("Growth of Algae Population \n r = [%s, %s]" % (min(r), max(r)))
    else:'''
    plt.title("Growth of Algae Population \n r = %s" % (r))
    plt.legend()
    plt.xlabel("Time (t)")
    plt.ylabel("Algal Density (mg/L)")
    #plt.ylim(ymin=-0.1e19)
    #plt.xlim(0, t+5)
    #plt.show()

def pop2(N0, r, t, K):
    # Input: N0, the initial population.
    # Input: r, the net growth rate per unit time.
    # Input: t, time in units.
    # Input: K, carrying capacity.
    
    # Init
    NtLst = [] # for plotting later
    pop = [] # for Nt vs t graph
    Nt = N0 # for logistic sequence
    plt.figure()
    
    #k = 1 / float(K)
    
    # for Nt vs t graph
    colors = ['#C91111', '#5BD2C0', '#A78B00', '#FFC1CC', '#9966CC', '#D84E09', '#F6E120', '#788193',
    '#006A93', '#6E7F80', '#FF8000', '#808080', '#514E49', '#867200', '#0070FF', '#F6EB20', '#7E9156',
    '#098FAB', '#E2B631', '#21ABCD', '#51C201', '#83AFDB', '#F4FA9F', '#6EEB6E', '#85BB65', '#1C8E0D',
    '#F863CB', '#FED8B1', '#FFC800', '#32CD32', '#09C5F4', '#B44848', '#A32E12', '#CC99BA', '#00FF7F',
    '#2862B9']
    
    # Update based on logistic sequence
    if isinstance(r, list):
        for ri in r:
            for i in range(0, t):
                Nt += ri * Nt * (1 - Nt) / float(K)
                pop.append(Nt)
                print Nt
            
            # for Nt vs time plot
            plt.plot(range(1, t+1), pop, color = colors[r.index(ri)], label = '%s' % (ri))
            pop = []
            print pop
            
            # for Nt vs r plot
            NtLst.append(round(Nt))
            print NtLst
        
        # Population vs growth rate graph
        '''plt.plot(r, NtLst, 'b-')
        if len(r) >= 20:
            plt.title("Growth of Fish Population \n r = [%s, %s], step = %s, K = %s" % (min(r), max(r), r[1]-r[0], K))
        else:
            plt.title("Growth of Fish Population \n r = %s, K = %s" % (r, K))
        plt.xlabel("Growth Rate (r)")
        plt.ylabel("Number of Fish (N" + str(t) + ")")
        plt.ylim(-50, 650)
        plt.show()'''
    
    else:
        for i in range(0, t):
            Nt += r * Nt * (1 - Nt / float(K))
            pop.append(Nt)
        plt.plot(range(1, t+1), pop, 'r--', label='pop2')
   
    # Population vs time graph 
    '''if len(r) >= 20:
        plt.title("Growth of Fish Population over Time \n r = [%s, %s], step = %s, K = %s" % (min(r), max(r), r[1]-r[0], K))
    else:'''
    plt.title("Growth of Fish Population over Time \n r = %s, K = %s" % (r, K))
    plt.legend()
    plt.xlabel("Time (t)")
    plt.ylabel("Number of Fish (Nt)")
    plt.ylim(0, 11)
    #plt.xlim(0, t+10)
    #plt.show()

def Histogram(population, normed, m):
    plt.figure()
    colors = ['r', 'g', 'b', 'black', 'gray']
    for i in range(len(population)):
        plt.hist(population[i], bins = 5, color = colors[i], normed=False, stacked=True, label=str(m[i]))
    
    plt.title("Distribution of Population \n m = %s" % (m))
    plt.xlabel("Population")
    plt.ylabel("Relative Frequency")
    plt.legend()
    plt.show()

def random_rt(N0, t, s, m, graphKind=None):
    NtLst = []
    pop = []
    
    popLst = []
    Nt = N0
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    
    for mi in m:
        for i in range(0, t):
            rt = normal(mi, s**2)
            Nt += float(rt) * Nt
            pop.append(round(Nt))
        plt.plot(range(1, t+1), pop, 'g--', label='random_rt')
        
        # Nt vs time plot
        '''if graphKind == "time" and m.index(mi) > 0:
            ax2 = ax.twinx()
            ax2.plot(range(1, t+1), pop, linestyle = 'None', marker = '.', color = 'g', label = '%s' % (mi))
        elif m.index(mi) == 0:
            ax.plot(range(1, t+1), pop, linestyle='None', marker='x',color = 'b', label = '%s' % (mi)) '''
        
        NtLst.append(round(Nt))
        print NtLst
        Nt = N0
        popLst.append(pop)
        pop = []
        
    # Population vs mean graph
    if graphKind == "mean":
        plt.plot(m, NtLst)
        
        if len(m) >= 20:
            plt.title("Growth of City Population \n m = [%s, %s], step = %s" % (min(m), max(m), m[1]-m[0]))
        else:
            plt.title("Growth of City Population \n m = %s" % (m))
        plt.xlabel("Mean of rt Distribution (m)")
        plt.ylabel("Population (N" + str(t) + ")")
        #plt.legend()
        #plt.xlim(min(m), max(m)+0.1)
        #plt.ylim(ymin = -0.05*max(NtLst))

    # Population vs time graph
    elif graphKind == "time":
        if len(m) >= 20:
            plt.title("Growth of City Population over Time \n m = [%s, %s], step = %s" % (min(m), max(m), m[1]-m[0]))
        else:
            plt.title("Growth of City Population over Time \n m = %s" % (m))
            plt.legend()
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Population (m = %s)" % m[0])
        ax2.set_ylabel("Population (m = %s)" % m[1])
        #plt.ylim(-10, 10000)
        #plt.xlim(0, t+10)
        plt.legend()
        '''if m[1] == 2.8:
            ax.legend(loc = 2)
            ax2.legend(loc = 3)
        elif m[1] == 0.0:
            ax.legend()
            ax2.legend(loc = 4)'''
    
    # Distribution of population
    #Histogram(popLst, True, m)
    
    #plt.show()    

def TimePlot(t, population, r, label, N0, K):
    plt.figure()
    for i in range(len(population)):
        plt.plot(t, population[i], linestyle='None', marker='.', label=r)
    plt.title("Growth of " + label + " Population \n N0 = %s, K = %s" % (N0, K))
    plt.xlabel("Time (days)")
    plt.ylabel("Algal Density (mg/L)")
    plt.legend()

def city_pop(N0, t, r, K, graphKind):
    # N0 is a list of initial populations for each city!
    
    Nts1 = [] # for plotting later
    Nts2 = []
    
    xts1 = [] # for bifurcation diagram
    xts2 = []
    
    pop1 = [] # for Nt vs t graph
    pop2 = []
    pop1s = []
    pop2s = []
    
    Nt = []
    Nt.append(N0[0]) # for logistic sequence
    Nt.append(N0[1])
    
    limits = [K - Nt[i] for i in range(len(Nt))]
    #plt.figure()
    
    # Do we have a list of rates?
    if isinstance(r, list):
        
        # For each rate in the list of rates
        for ri in r:
            
            # For t time steps
            for i in range(t):
                
                # For each city
                for j in range(len(Nt)):
                    # Calculate new population
                    Nt[j] += ri * Nt[j] * (1 - Nt[j] / float(limits[j]))
                    
                    # Update population limit
                    limits[j] = K - Nt[j]
                
                pop1.append(round(Nt[0]))
                pop2.append(round(Nt[1]))
                                
            pop1s.append(pop1)
            pop2s.append(pop2)
            
            pop1 = []
            pop2 = []
            
            Nts1.append(Nt[0])
            Nts2.append(Nt[1])
            
            xt1 = Decimal(ri) / Decimal(1 + ri) * Decimal(Nt[0]) / Decimal(limits[0])
            xts1.append(xt1)
            
            xt2 = Decimal(ri) / Decimal(1 + ri) * Decimal(Nt[1]) / Decimal(limits[1])
            xts2.append(xt2)
            
            # Reset Nt to N0
            Nt[0] = N0[0]
            Nt[1] = N0[1]
            
        # Population vs growth rate graph - not useful. Exponential shape.
        if graphKind == "rate":
            plt.plot(r, Nts1, label = "City 1")
            plt.plot(r, Nts2, label = '%s' % ("City 2"))
            plt.title("Growth of City Population \n N01 = %s, N02 = %s, K = %s" % (N0[0], N0[1], K))
            plt.xlabel("Growth Rate (r)")
            plt.ylabel("Population (N" + str(t) + ")")
            plt.legend()
            #plt.ylim(-50, 650)
        
        # Population vs time graph
        elif graphKind == "time":
            TimePlot(t, pop1s, r, "City 1", N0[0], K)
            TimePlot(t, pop2s, r, "City 2", N0[1], K)
            #plt.ylim(-10, 260)
            #plt.xlim(0, t+10)
            plt.legend()
        
    # Bifurcation diagram - broken :( :(
    '''plt.figure()
    plt.title("Bifurcation Diagram for City Population \n N01 = %s, N02 = %s" % (N0[0], N0[1]))
    plt.xlabel("Growth rates (r)")
    plt.ylabel("x_t")
    
    plt.plot(r, xts1)'''
    
    # Histogram
    #Histogram(pop1s, True, r)
    
    plt.show()
        
def sim_growth(N01, N02, t, graphKind, alone=True):
    pop1 = []
    pop2 = []
    
    rts = []
    
    N1 = N01
    N2 = N02
    
    fig = plt.figure()
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
       
    # For t time steps
    for i in range(t):
        rt = normal(1, 0.2)
        rts.append(rt)
        
        # Calculate new population
        N1prev = N1
        N2prev = N2
        
        N1 += rt * float(N1) / float(N1 + N2prev) * N1
        N2 += rt * float(N2) / float(N2 + N1prev) * N2
        
        # Population over time
        pop1.append(round(N1))
        pop2.append(round(N2))
    
    if alone:
        
        # Population over time plot
        if graphKind == "time":
            plt.plot(range(1, t+1), pop1, linestyle='None', marker='.',label = "City 1")
            
            plt.title("Growth of City Population over Time \n N01 = %s, N02 = %s" % (N01, N02))
            plt.xlim(xmax = t+5)
            plt.ylim(ymax = max(pop1) + 20)
            plt.legend()
            
            plt.figure()
            plt.plot(range(1, t+1), pop2, label = "City 2")
            
            plt.title("Growth of City Population over Time \n N01 = %s, N02 = %s" % (N01, N02))
            plt.xlim(xmax = t+5)
            plt.ylim(-0.01*max(pop2), 1.2*max(pop2))
            plt.legend()
        
        # Population vs rate plot - which says nothing, but looks like a tornado.
        elif graphKind == "rate":
            plt.plot(pop1, rts, label = "City 1")
            
            plt.title("Growth of City Population over Time \n N01 = %s, N02 = %s" % (N01, N02))
            plt.ylim(ymax = max(rts)+0.5)
            plt.xlim(xmax = max(pop1) + 20)
            plt.legend()
            
            plt.figure()
            plt.plot(pop2, rts, label = "City 2")
            
            plt.title("Growth of City Population over Time \n N01 = %s, N02 = %s" % (N01, N02))
            plt.ylim(ymax = max(rts)+0.5)
            plt.xlim(-0.01*max(pop2), 1.2*max(pop2))
            plt.legend()
        
    else:
        
        # Population over time plot
        if graphKind == "time":
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            
            ax1.plot(range(1, t+1), pop1, linestyle='None', marker='.', color = 'b', label = "City 1")
            ax2.plot(range(1, t+1), pop2, linestyle='None', marker='.',color = 'g', label = "City 2")
            
            ax1.set_xlim(xmax = t+5)
            if N02 == 10000:
                ax1.set_ylim(ymax = 1.02*max(pop1))
            else:
                ax1.set_ylim(ymax = 1.1*max(pop1))
            ax1.legend(loc = 2)
            ax1.set_xlabel("Time (t)")
            ax1.set_ylabel("City 1 Population (N1)")
            
            ax2.set_xlim(xmax = t+5)
            ax2.set_ylim(-0.01*max(pop2), 1.2*max(pop2))
            ax2.legend(loc = 1)
            ax2.set_ylabel("City 2 Population (N2)")
            
            plt.title("Growth of City Population over Time \n N01 = %s, N02 = %s" % (N01, N02))
        
        # Population vs rate plot - which says nothing, but looks like a tornado. Code broken!
        elif graphKind == "rate":
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            
            ax1.plot(pop1, rts, color = 'b', label = "City 1")
            ax2.plot(pop2, rts, color = 'g', label = "City 2")
            
            ax1.set_ylim(ymax = max(rts)+0.5)
            ax1.set_xlim(xmax = max(pop1) + 20)
            ax1.legend(loc = 2)
            ax1.set_ylabel("Growth Rate (rt)")
            ax1.set_xlabel("City 1 Population (N1)")
            
            ax2.set_ylim(ymax = max(rts)+0.5)
            ax2.set_xlim(-0.01*max(pop2), 1.2*max(pop2))
            ax2.legend(loc = 3)
            ax2.set_xlabel("City 2 Population (N2)")

    plt.show()


def LogisticMap(r,x):
	return r * x * (1.0 - x)

def Bifurcation():
    
    # Set the initial condition used across the different parameters
    ic = 0.2
    # Establish the arrays to hold the set of iterates at each parameter value
    # The iterates we'll throw away
    nTransients = 200
    # This sets how much the attractor is filled in
    nIterates = 250
    # This sets how dense the bifurcation diagram will be
    nSteps = 400
    # Sweep the control parameter over the desired range
    rInc = (rhigh-rlow)/float(nSteps)
    for r in arange(rlow,rhigh,rInc):
   	# Set the initial condition to the reference value
   	state = ic
   	# Throw away the transient iterations
   	for i in xrange(nTransients):
  		state = LogisticMap(r,state)
   	# Now store the next batch of iterates
   	rsweep = [ ]   # The parameter value
   	x = [ ]        # The iterates
   	for i in xrange(nIterates):
  		state = LogisticMap(r,state)
  		rsweep.append(r)
  		x.append( state )
   	plot(rsweep, x, 'k,') # Plot the list of (r,x) pairs as pixels
    
    # Use this to save figure as a bitmap png file
    #savefig('LogisticBifn', dpi=600)
    
    # Turn on matplotlib's interactive mode.
    #ion()
    
    # Display plot in window
    show()

def A_naught(A_0, r, T, K, dt):
    A_t = A_0
    t = 0
    colors = ['#C91111', '#5BD2C0', '#A78B00', '#FFC1CC', '#9966CC', '#D84E09', '#F6E120', '#788193',
    '#006A93', '#F4FA9F', '#6EEB6E', '#85BB65', '#1C8E0D',
    '#F863CB', '#FED8B1', '#FFC800', '#32CD32', '#09C5F4', '#B44848', '#A32E12', '#CC99BA', '#00FF7F',
    '#2862B9']
    
    tList = []
    A_tList = []
    for i in range(len(dt)):
        while t <= T:
            A_tList.append(A_t)
            tList.append(t)
            
            dA_dt = r * A_t * (1 - A_t/float(K))
            t_star = t + dt[i]
            A_t = A_t + dA_dt * (t_star - t)
            t = t_star
        #plt.figure()
        '''if i == 0:
            plt.plot(tList, A_tList, color=colors[i], label='dt = 0.1')
        elif i == len(dt) - 1:
            plt.plot(tList, A_tList, color=colors[i], label='dt = 6.0')
        else:'''
        plt.plot(tList, A_tList, color=colors[i], label='linear approx')
        #savefig(str(i) + '.png', bbox_inches='tight')
        A_t = A_0
        t = 0
        tList = []
        A_tList = []
    #plt.ylim(ymax = 11)
    plt.title("Growth of Algae Population \n A0 = %s, K = %s" % (A_0, K))
    plt.xlabel("Time (days)")
    plt.ylabel("Algal Density (mg/L)")
    plt.legend(loc=2)
    plt.show()
    
    return tList, A_tList    

def AofT(A_0, r, T, K):
    At_list = []
    A_t = A_0
    
    for t in T:
        A_t = K / float(1 + (K / float(A_0) -1) * math.exp(-1 * r * t))
        At_list.append(A_t)
    
    df = len(At_list) - 1
    stdev = std(At_list)
    #plt.errorbar(T, At_list, yerr=ss.t.ppf(0.6, df)*stdev, label='actual function')
    
    
    plt.plot(T, At_list, label='actual function')
    plt.legend()
    plt.title("Growth of Algae Population \n A0 = %s, K = %s" % (A_0, K))
    plt.xlabel("Time (days)")
    plt.ylabel("Algal Density (mg/L)")
    plt.show()

def lotka(x10, x20, m12, m21, r1, r2, K1, K2, T):
    # This is total shenanigans and is incorrect/broken, have a fail whale:
    #    W     W      W        
    #    W        W  W     W    
    #                '.  W      
    #      .-""-._     \ \.--|  
    #     /       "-..__) .-'   
    #    |     _         /      
    #    \'-.__,   .__.,'       
    #     `'----'._\--'      
    #    VVVVVVVVVVVVVVVVVVVVV
    
    dx1_dt = x10
    dx2_dt = x20
    list1 = []
    list2 = []
    
    while dx1_dt > 0:
        dx1_dt = r1 * dx1_dt * (1 - (dx1_dt + m12*dx2_dt/float(K1)))
        dx2_dt = r2 * dx2_dt * (1 - (dx2_dt + m21*dx1_dt/float(K2)))
        list1.append(dx1_dt)
        list2.append(dx2_dt)
    
    plt.plot(T, list1, label = "pop 1")
    plt.plot(T, list2, label = "pop 2")
    plt.title("Growth of Algae Population \n x1(0) = %s, x2(0) = %s, r = %s" % (x10, x20, r1))
    plt.xlabel("Time (days)")
    plt.ylabel("Algal Density (mg/L)")
    plt.legend()
    plt.show()
    
    return dx1_dt, dx2_dt

def SIR_approx(c, b, a, dt, S0, I0, R0, L0, T):
    S_t = S0
    I_t = I0
    R_t = R0
    L_t = L0
    t = 0
    
    N = S0 + I0 + R0 + L0
    
    tList = []
    StList = []
    ItList = []
    RtList = []
    LtList = []
    
    tList.append(t)
    StList.append(S_t)
    ItList.append(I_t)
    RtList.append(R_t)
    LtList.append(L_t)
    plt.figure()
    
    for i in range(len(b)):
        while t < T:
            dS_dt = -1 * b[i] / float(N) * S_t * L_t
            dI_dt = c * L_t - a * I_t
            dR_dt = a * I_t
            dL_dt = float(b[i])/float(N) * S_t * I_t - c * L_t
            
            S_t = S_t + dS_dt * dt
            I_t = I_t + dI_dt * dt
            R_t = R_t + dR_dt * dt
            L_t = L_t + dL_dt * dt
                        
            '''S_t = S_t - bSI*dt[i]
            I_t = I_t + (bSI - aI)*dt[i]
            R_t = R_t + aI*dt[i]'''
            
            t_star = t + dt
            t = t_star
            
            tList.append(t)
            StList.append(S_t)
            ItList.append(I_t)
            RtList.append(R_t)
            LtList.append(L_t)
            
        #plt.figure()
        
        # Plotting shenanigans
        #print S_t, I_t, R_t
        plt.plot(tList, StList, 'r-', label = "$S(t)$")
        plt.plot(tList, ItList, 'b-', label = "$I(t)$")
        plt.plot(tList, RtList, 'g-', label = "$R(t)$")
        plt.plot(tList, LtList, 'm-', label = "$L(t)$")
        plt.xlim(0, T + 1)
        plt.title(r"SIR Model, $N = %s,\ \beta = %r,\ \gamma = %r,\ S_0 = %s,\ I_0 = %s,\ R_0 = %s$" % (N, b, a, S0, I0, R0))
        plt.xlabel("Time (days)")
        plt.ylabel("Number of S/I/R Individuals")
        plt.legend(loc=5)
        plt.grid()
        
        #savefig('secondgifCase' + str(i) + '.png', bbox_inches='tight')
        '''maxy = max(StList)
        if max(ItList) > maxy:
            maxy = max(ItList)
        if max(RtList) > maxy:
            maxy = max(RtList)'''
        
        S_t = S0
        R_t = R0
        I_t = I0
        t = 0
        tList = []
        StList = []
        ItList = []
        RtList = []
    #print b/float(a)
    #plt.ylim(-5, 105)
    
    plt.show()
    
def fishlife(C, N, t, h, r):
    bay_pop = [0] * C
    bay_pop = drop(bay_pop, N, C)
    stdev = []
    
    for time in t:
        for i in range(time):
            bay_pop = day(bay_pop, C, h, r)
        stdev.append(np.std(bay_pop))
    
    variance = [x**2 for x in stdev]
    
    #schoolplot(bay_pop, t, N, C)
    plt.figure()
    plt.plot(t, variance)
    plt.title('Variance of Zone Sizes as f(t) \n t = range(500, 2001, 10), N = %s, C = %s' % (N, C))
    plt.xlabel("Time (t)")
    plt.ylabel("Variance ($\sigma^2$)")
    plt.show()

#def strange(r, m, x, K, N):
    

def drop(bay_pop, n, C):
    # bay_pop: list
    # n: integer > 0
    
    for i in range(1, n+1):
        z = randint(0, C)
        bay_pop[z] += 1
    
    return bay_pop

def remove(bay_pop, C):
    re = randint(0, C)
    while bay_pop[re] == 0:
        re = randint(0,C)
    bay_pop[re] -= 1
    
    return bay_pop

def day(bay_pop, C, h, r):
    newbay = [0] * C
    Nt_1 = 0
    
    for i in range(C):
        if bay_pop[i] != 0:
            d = randint(0, C)
            while d == i:
                d = randint(0, C)
            
            if newbay[d] != 0:
                newbay[d] += bay_pop[i]
            else:
                newbay[d] = bay_pop[i]
    
    for pop in bay_pop:
        Nt_1 += pop
        
    newbay = drop(newbay, int(round(h * Nt_1)), C)
    
    for k in range(int(round(r * Nt_1))):
        newbay = remove(newbay, C)
    
    return newbay 

def schoolplot(bay_pop, t, N, C):
    plt.hist(bay_pop, bins=50, normed=True)
    
    plt.title("Size Distribution of Fish Schools \n t = %s, N = %s, C = %s" %  (t, N, C))
    plt.ylabel("Relative Frequency")
    plt.xlabel("School Size")
    plt.show()

def main():
    #Bifurcation()

    fishlife(100, 5000, range(500, 2001, 10), 0.1, 0.1)

    #lotka(0.8, 0.2, 1.5, 1.5, 2.5, 1.5, 4, 4, frange(0.1, 10.0, 0.1))
    
    # Format: SIR_approx(b, a, N, dt, S0, I0, R0, T)
    # Case 1
    #SIR_approx(0.1, 0.4, 100, [0.1], 99, 1, 0, 100)
    
    # Case 2
    #SIR_approx(0.4, 0.1, 100, [0.1], 99, 1, 0, 100)
    
    # Case 3
    #SIR_approx(0.1, 0.4, 100, [0.1], 75, 25, 0, 100)
    
    # Case 4
    #SIR_approx(0.4, 0.1, 100, [0.1], 75, 25, 0, 100)
    
    # Case 5
    #SIR_approx(0.8, 0.1, 100, [0.001], 75, 25, 0, 100)
    
    # I animate things
    #SIR_approx([0.1], 0.4, 0.01, 892500, 75000, 32500, 100)
    #SIR_approx(8, [4], 1, 0.01, 95, 1, 0, 4, 10)
    #SIR_approx(0.1, [0.1], 0.4, 0.01, 925000, 75000, 0, 0, 100)
    
    # 
    
    k = [m-6 for m in range(4, 9)]
    r1 = frange(-0.8, 2.0, 0.1)
    r2 = frange(-0.8, 2.9, 0.001)
    req_r = [-0.8, -0.01, 0.01, 0.2, 0.5, 0.99, 1.01, 2.8, 3.8] # what assignment asked for
    req_r2 = [-0.8, -0.01, 0.01, 0.5, 2.8]
    
    #pop0(100, k, 50)
    #pop1(1, 1, 10)
    #pop2(1, 1, 10, 10)
    
    Anaught = 1
    Kay = 2500
    are = 1
    
    #A_naught(Anaught, are, 10, Kay, [0.1])
    #AofT(Anaught, are, frange(0.1, 10.0, 0.1), Kay)
    
    mList1 = [-0.1, 0.0]
    mList2 = [0.1, 2.8]
    m = [-0.1, 0.0, 0.1, 2.8]
    
    #random_rt(1, 10, 0.2, [1.0])
    #random_rt(100, 500, 0.2, mList2, 'time')
    #random_rt(100, 500, 0.2, [2.81, 2.8])
    #random_rt(100, 500, 0.2, [0.1])
    
    # Four different sets of initial populations
    initPop1 = [100, 100]
    initPop2 = [100, 200]
    initPop3 = [500, 1000]
    initPop4 = [1000, 10000]
    
    # Different Ks depending on initial populations
    K = [500, 1000, 5000, 50000]
    
    # List of rates
    r = [-0.7, -0.1, 0, 0.1, 0.7]
    
    # Calling city_pop with different initial populations
    #city_pop(initPop1, 50, r, K[0], "time") # try rate too
    #city_pop(initPop2, 50, r, K[1], "time")
    #city_pop(initPop3, 50, r, K[2], "time")
    #city_pop(initPop4, 50, r, K[3], "time")
    
    # Calling sim_growth with different initial populations
    #sim_growth(initPop1[0], initPop1[1], 50, "time", False)
    #sim_growth(initPop2[0], initPop2[1], 50, "time", False)
    #sim_growth(initPop3[0], initPop3[1], 50, "time", False)
    #sim_growth(initPop4[0], initPop4[1], 50, "time", False)

if __name__ == "__main__":
    main()