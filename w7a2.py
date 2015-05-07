import math
import matplotlib.pyplot as plt
from frange import frange

def two(prop_x, prop_y, T, dt, mu, index=None):
    t = 0
    F_x = 0
    x_t = prop_x
    y_t = prop_y
    # Calculate fitnesses:
    # f_c = fitness of cleaners, f_n = fitness of non-cleaners
    f_c = prop_x * 10
    f_n = prop_x * 15 + prop_y * 2
    
    # Calculate average population fitness, F(x)
    F_x = (f_c + f_n) / 2.0
    
    x_ts = [x_t]
    y_ts = [y_t]
    ts = [t]
    
    while t <= (T-dt):        
        # Update differential equations
        dx_dt = (x_t * f_c * mu/float(f_n)) - F_x*x_t
        dy_dt = (y_t * f_n * mu/float(f_c)) - F_x*y_t

        # Update values
        x_t = x_t + dx_dt * dt
        y_t = y_t + dy_dt * dt
        t = t + dt
                
        x_ts.append(x_t)
        y_ts.append(y_t)
        ts.append(t)
        
        f_c = (x_t / float(x_t + y_t)) * 10
        f_n = (x_t / float(x_t + y_t)) * 15 + (y_t / float(x_t + y_t)) * 2
        F_x = (f_c + f_n) / 2.0

    plt.figure()
    plt.plot(ts, x_ts, color='blue', label='Cleaners')
    plt.plot(ts, y_ts, color = 'red', label = 'Non-Cleaners')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('People')
    #plt.ylim(-0.1, 1.1)
    plt.title('Proportion of Roles over Time Using Replicator Dynamics')
    #plt.savefig("trying" + str(index) + ".jpg", bbox_inches="tight")
    plt.show()
    
    return x_ts[0], y_ts[0]

def one_a(x_t, y_t, days, index=None):
    
    
    total = 8.0
    x = [x_t]
    y = [y_t]
    
    for i in range(days):
        # Payoff for cleaners
        c_t = (x_t / total) * 10
        
        # Payoff for non-cleaners
        n_t = ((x_t / total) * 15) + ((y_t / total) * 2)
        
        # Average payoff
        m_t = (c_t + n_t) / float(2)
        
        # Max payoff
        if (x_t != 8) and (y_t != 8):
            M_t = 15
        elif x_t == 8:
            M_t = 10
        else:
            M_t = 2
        
        # Update populations
        x_t += x_t * ((c_t - m_t)) / float(M_t)
        #y_t += y_t * ((n_t - m_t)) / float(M_t)
        #x_t = total - round(y_t)
        y_t = total - x_t
        #y_t = min(y_t, 8)
        #x_t = max(x_t, 0)
        
        x.append(round(x_t))
        y.append(round(y_t))
        
        # Update proportions
        #x_t = round((x_t / float(x_t+y_t)) * 8)
        #y_t = round((y_t / float(x_t+y_t)) * 8)
    
    print "x = ", x
    print "y = ", y
    plt.figure()
    plt.plot(range(0, days+1), x, color='blue', label='Cleaners')
    plt.plot(range(0, days+1), y, color='red', label='Not Cleaners')
    plt.title("Change in Roles for %s Steps \n Pop = %s, Initial Cleaners = %s" % (days, total, x[0]))
    plt.xlabel("Time")
    plt.ylabel("People")
    #plt.ylim(-0.5, 8.5)
    plt.grid()
    plt.legend()
    plt.savefig("tragedygraph" + str(index) + ".jpg", bbox_inches="tight")
    plt.show()
    print x[0], y[0]
    
    return round(x[-1]), round(y[-1])

def main():
    clean = range(0, 9)
    nonclean = sorted(clean, reverse=True)

    '''for i in range(len(clean)):
        one_a(clean[i], nonclean[i], 100, i)
    print one_a(8, 0, 10)'''
    
    '''range1 = frange(0, 1, 0.1)
    range2 = sorted(range1, reverse=True)
    for i in range(len(range1)):
        two(range1[i], range2[i], 10, 0.1, i)'''
    two(0.8, 0.2, 20, 0.1, 0.05)

if __name__ == "__main__":
    main()