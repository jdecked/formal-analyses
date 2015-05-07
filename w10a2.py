from w10a1 import npv
import numpy as np
import random
import matplotlib.pyplot as plt

def npv2(rs):
    NPVs = []
    
    for r in rs:
        for i in range(1000):
            k = random.randint(2, 8)
            NPV = 0
            
            for i in range(20):
                if i < k:
                    C = np.random.normal(25, 10)
                    
                    while C <= 0:
                        C = np.random.normal(25, 10)
                            
                    NPV -= npv(C, i + 1, r)
                    
                else:
                    R = np.random.normal(20, 10)
                    
                    #while R <= 0:
                        #R = np.random.normal(20, 10)
                            
                    NPV += npv(R, i + 1, r)
            
            NPVs.append(NPV)
        
        mean = np.mean(NPVs)
        median = np.median(NPVs)
        stdev = np.std(NPVs)
        
        print "VaR 95%: " + str(VaR(0.95, NPVs))
        print "VaR 99%: " + str(VaR(0.99, NPVs))
        
        #print "Mean (r = %s): " % r + str(mean)
        #print "$/sigma$ (r = %s): " % r + str(stdev)
        
        plt.figure()
        plt.hist(NPVs, bins=50, normed=True)
        plt.ylabel("Relative Frequency")
        plt.xlabel("Net Present Value")
        plt.title("Histogram of 1000 Net Present Values \n $C = N(25, 10)$")
        plt.figtext(0.2,0.8,"$r = %s$" % r, fontsize=17)
        plt.savefig("npv2" + str(r), bbox_inches="tight")
    #plt.show()

def npv3(rs):
    NPVs = []
    
    for r in rs:
        for i in range(1000):
            k = random.randint(7, 13)
            NPV = 0
            
            for i in range(20):
                if i < k:
                    C = 60 + abs(np.random.normal(0, 20))
                            
                    NPV -= npv(C, i + 1, r)
                    
                else:
                    R = np.random.normal(70, 30)
                    
                    #while R <= 0:
                        #R = np.random.normal(20, 10)
                            
                    NPV += npv(R, i + 1, r)
            
            NPVs.append(NPV)
        
        mean = np.mean(NPVs)
        #median = np.median(NPVs)
        #stdev = np.std(NPVs)
        
        #print "Mean (r = %s): " % r + str(mean)
        #print "Standard Deviation (r = %s): " % r + str(stdev)
        
        print "VaR 95%: " + str(VaR(0.95, NPVs))
        print "VaR 99%: " + str(VaR(0.99, NPVs))
        
        '''plt.figure()
        plt.hist(NPVs, bins=50, normed=True)
        plt.ylabel("Relative Frequency")
        plt.xlabel("Net Present Value")
        plt.title("Histogram of 1000 Net Present Values \n $C = 25 + |N(0, 10)|$")
        plt.figtext(0.15,0.8,"$r = %s$" % r, fontsize=17)
        plt.savefig("npv3" + str(r), bbox_inches="tight")'''
    #plt.show()

def pi(runs):
    pis = []
    for i in range(runs):
        circle_area = 0
        
        for j in range(runs):
            x = random.random()
            y = random.random()
            
            if (x**2 + y**2 <= 1):
                circle_area += 1
        
        pi_approx = 4 * float(circle_area) / float(runs)
        pis.append(pi_approx)
    print "Pi approximation: " + str(np.mean(pis))
    plt.figure()
    plt.hist(pis, bins=40, normed=True)
    plt.title("Approximation of $\pi$ Using Monte Carlo Method")
    plt.xlabel("Approximation of $\pi$")
    plt.ylabel("Relative Frequency")
    plt.show()
        
def VaR(p, outcomes):
    outcomes.sort()
    index = int(round((1 - p) * len(outcomes)))
    value_at_risk = outcomes[index]
    
    return value_at_risk

#npv2(range(1, 16))
#npv2([10])
#npv3([10])
#npv3(range(1, 16))
pi(1000)