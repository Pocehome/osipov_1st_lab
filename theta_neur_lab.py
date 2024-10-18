import numpy as np
import matplotlib.pyplot as plt


def RK4_step(f, X, dt):
    k1 = dt * f(X)
    k2 = dt * f(X + k1 / 2)
    k3 = dt * f(X + k2 / 2)
    k4 = dt * f(X + k3)

    X_next = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X_next


def theta_neur_dyn(gamma, n):
    
    def rhs(theta):
        res = gamma - np.cos(theta/n)
        return res

    return rhs


# def normalize_angle(angle):
#     normalized = angle % (2 * np.pi)
#     return normalized


def draw_graph(theta_num, theta_true, t, limits, title, theta_lim=0, T=0): 
    x_lims, t_lims = limits
    plt.ylim(x_lims[0], x_lims[1])
    plt.xlim(t_lims[0], t_lims[1])
        
    plt.plot(t, theta_num, 'darkgreen', label='num. sol.')
    # plt.plot(t, theta_true, 'red', label='true sol.')
    
    if theta_lim:
        plt.axhline(y=theta_lim, color='black', linestyle='--', label='\u03B81')
    if T:
        plt.axvline(x=T, color='black', linestyle='--', label=f'T={T}')
    
    plt.title(f"\u03B3={title[0]}, n={title[1]}")
    plt.xlabel('time', fontsize=10, color='black')
    plt.ylabel('\u03B8', fontsize=10, color='black')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.show()
    
    
def make_graph(gamma, step_count):
    # initial values
    theta0 = 0.
    
    for n in [1, 3]:
        # numerical integration
        rhs = theta_neur_dyn(gamma, n)
        sol = theta0
        
        if n == 3 and 1 < gamma < 1.2:
            step_count *= 2
            
        step_size = 0.01
        arr_sol = np.array([0.] * step_count)
        arr_t = np.array([0.] * step_count)
        
        for k in range(step_count):
            sol = RK4_step(rhs, sol, step_size)
            arr_sol[k] = sol
            arr_t[k] = k * step_size

        theta_true = [0]    # temporary zaglushka
        
        # draw graphs
        norm_arr_sol = np.mod(arr_sol, 2*np.pi)     # [0; 2pi]
        
        # search period
        T = 0
        spikes_count = 0
        for i in range(step_count - 1):
            if (int(norm_arr_sol[i]) == 6 and 
                int(norm_arr_sol[i+1]) == 0):
                spikes_count += 1
            if spikes_count == n:
                T = arr_t[i+1]
                break
        
        if gamma < 1:
            # take neg solution of equation and reduce it to [0, 2pi]:
            theta_lim = 2*np.pi - n * np.arccos(gamma)
            
            # if gamma < 1, we also draw theta limit
            draw_graph(norm_arr_sol, theta_true, arr_t,
                       [(0, 2*np.pi), (0, step_count*step_size)], [gamma, n],
                       theta_lim=theta_lim)
        else:
            #  if gamma > 1, we also draw period
            draw_graph(norm_arr_sol, theta_true, arr_t,
                       [(0, 2*np.pi), (0, step_count*step_size)], [gamma, n],
                       T=T)


if __name__ == '__main__':
    
    make_graph(0.1, 2000)
    make_graph(0.5, 2000)
    make_graph(1.01, 10000)
    make_graph(1.2, 5000)
    make_graph(5., 1000)
    