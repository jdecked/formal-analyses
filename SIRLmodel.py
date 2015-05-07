import matplotlib.pyplot as plt

def SIR_approx(a, b, c, dt, S0, I0, R0, L0, T):
    """
    This is a graphical model of a population with four groups:
    people who are not infected (susceptible), people who are infected and
    contagious (infected), people who have been infected but are not yet
    contagious (latent), and people who are unable to contract the infection
    (removed). People can only belong to one group at a time, and everyone has
    to belong to at least one group.
    
    Input:
        a - rate at which people are removed from the infected population
        b - probability that an uninfected person will become infected
        c - rate at which people become contagious
        dt - discrete time variable which mimics continous time
        S0, I0, R0, L0 - initial susceptible, infected, removed, and latent populations
        T - number of time steps to run model for
        
    Output:
        Plot of S_t, I_t, R_t, L_t for t = T
    """
    
    # Setting initial populations to S_t, etc - will need initial numbers later
    S_t = S0
    I_t = I0
    R_t = R0
    L_t = L0
    t = 0
    
    # Calculating total population
    N = S0 + I0 + R0 + L0
    
    # Initializing lists of populations - will need for plotting later
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
    
    # For each value of b - this can be run with multiple at once
    for i in range(len(b)):
        # Have we reached the maximum number of time steps?
        while t < T:
            
            # Calculating SIRL models based on ordinary differential equations
            dS_dt = -1 * b[i] / float(N) * S_t * L_t
            dI_dt = c * L_t - a * I_t
            dR_dt = a * I_t
            dL_dt = float(b[i])/float(N) * S_t * I_t - c * L_t
            
            # Approximating continuous SIR functions with discrete time variables
            S_t += dS_dt * dt
            I_t += dI_dt * dt
            R_t += dR_dt * dt
            L_t += dL_dt * dt
            
            t += dt
            
            tList.append(t)
            StList.append(S_t)
            ItList.append(I_t)
            RtList.append(R_t)
            LtList.append(L_t)
                    
        #print S_t, I_t, R_t, L_t
        
        # Plotting S_t, I_t, R_t, L_t over time
        plt.figure()
        plt.plot(tList, StList, 'r-', label = "$S(t) = %s$" % S0)
        plt.plot(tList, ItList, 'b-', label = "$I(t) = %s$" % I0)
        plt.plot(tList, RtList, 'g-', label = "$R(t) = %s$" % R0)
        plt.plot(tList, LtList, 'm-', label = "$L(t) = %s$" % L0)
        
        # Graph aesthetics
        plt.xlim(0, T + 1)
        plt.title(r"SIRL Model, $N = %s,\ b = %r,\ a = %r$" % (N, b, a))
        plt.xlabel("Time (days)")
        plt.ylabel("Number of S/I/R/L Individuals")
        plt.legend(loc=5)
        plt.grid()
        
        # Uncomment to save each graph - useful for making .gifs
        #savefig('secondgifCase' + str(i) + '.png', bbox_inches='tight')
        
        # Resetting everything to initial values for other values of b
        S_t = S0
        R_t = R0
        I_t = I0
        L_t = L0
        
        t = 0
        tList = []
        StList = []
        ItList = []
        RtList = []
        LtList = []
    
    plt.show()

# Format: SIR_approx(a, b, c, dt, S0, I0, R0, L0, T)
# Here are a few test cases

# Case 1
#SIR_approx(0.4, [0.1], 0.1, 0.01, 98, 1, 0, 1, 100)

# Case 2
#SIR_approx(0.1, [0.4], 0.2, 0.01, 98, 1, 0, 1, 100)

# Case 3
#SIR_approx(0.4, [0.1], 0.1, 0.01, 75, 20, 0, 5, 100)

# Case 4
#SIR_approx(0.1, [0.4], 0.2, 0.01, 75, 20, 0, 5, 100)

# Case 5
SIR_approx(0.1, [0.8], 0.3, 0.01, 75, 20, 0, 5, 100)

# Case 6
#SIR_approx(0.4, [0.1], 0.1, 0.01, 925000, 75000, 0, 0, 100)

# Uncomment this and savefig() above to save each graph to disk to make .gifs
# .gif with varied b
SIR_approx(0.4, [x/10.0 for x in range(1, 11)], 0.3, 0.01, 75, 20, 0, 5, 100)