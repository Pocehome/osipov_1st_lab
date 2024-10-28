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


def theta_true(gamma, n, t):
    if gamma >= 1:
        a = np.tan((t * (gamma**2 - 1)**0.5) / (2*n))
        b = (gamma - 1)**0.5 / (gamma + 1)**0.5
        res = 2*n * np.arctan(a*b)
    else:
        a = ((1 - gamma) / (1 + gamma))**0.5
        # C = abs((a - np.tan(np.pi / n)) / (a + np.tan(np.pi / n)))
        C = 1
        res = 2*n * np.arctan(2*a / (1 + C*np.exp((1 - gamma**2)**0.5 / n * t)) - a)
    return res

# def normalize_angle(angle):
#     normalized = angle % (2 * np.pi)
#     return normalized


def draw_graph(theta_num, theta_true, t, limits, title, theta_lim=0, T=0): 
    x_lims, t_lims = limits
    plt.ylim(x_lims[0], x_lims[1])
    plt.xlim(t_lims[0], t_lims[1])
    
    plt.plot(t, theta_true, 'red', label='true sol.')
    plt.plot(t, theta_num, 'darkgreen', label='num. sol.')
    
    if theta_lim:
        plt.axhline(y=theta_lim, color='orange', linestyle='--', label='\u03B81')
    if T:
        plt.axvline(x=T, color='blue', linestyle='--', label=f'T={T}')
    
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
        
        if n == 3 and 1 < gamma <= 1.2:
            step_count *= 2
            
        step_size = 0.01
        arr_num_sol = np.array([0.] * step_count)
        arr_t = np.array([0.] * step_count)
        
        for k in range(step_count):
            sol = RK4_step(rhs, sol, step_size)
            arr_num_sol[k] = sol
            arr_t[k] = k * step_size
        
        if gamma >= 1:
            arr_true_sol = theta_true(gamma, n, arr_t)
        else:
            arr_true_sol = [0] * step_count
        arr_true_sol = theta_true(gamma, n, arr_t)
        
        # normalize the solution by [0; 2pi]
        norm_arr_num_sol = np.mod(arr_num_sol, 2*np.pi)
        norm_arr_true_sol = np.mod(arr_true_sol, 2*np.pi)
        
        # search period
        T = 0
        spikes_count = 0
        for i in range(step_count - 1):
            if (int(norm_arr_num_sol[i]) == 6 and 
                int(norm_arr_num_sol[i+1]) == 0):
                spikes_count += 1
            if spikes_count == n:
                T = arr_t[i+1]
                break
        
        # draw graphs
        if gamma < 1:
            # take neg solution of equation and reduce it to [0, 2pi]:
            theta_lim = 2*np.pi - n * np.arccos(gamma)
            
            # if gamma < 1, we also draw theta limit
            draw_graph(norm_arr_num_sol, norm_arr_true_sol, arr_t,
                       [(0, 2*np.pi), (0, step_count*step_size)], [gamma, n],
                       theta_lim=theta_lim)
        else:
            # if gamma > 1, we also draw period
            draw_graph(norm_arr_num_sol, norm_arr_true_sol, arr_t,
                       [(0, 2*np.pi), (0, step_count*step_size)], [gamma, n],
                       T=T)
            

def draw_phase_portrait(gamma, n=1):
    plt.title(f"\u03B3={gamma}")
    plt.ylim(-1.25, 1.25)
    plt.xlim(-1.25, 1.25)
    circle = plt.Circle((0, 0), 1, color='black', linewidth=1, fill=False)
    ax=plt.gca()
    ax.add_patch(circle)
    # plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.text(-0.03, 1.1, '0')
    plt.text(-0.1, -1.2, '\u03C0 (-\u03C0)')
        
    if 0 <= gamma < 1:
        state1 = n * np.arccos(gamma)
        state2 = -n * np.arccos(gamma)

        plt.scatter(np.cos(state1 + np.pi/2), np.sin(state1 + np.pi/2), 
                    s=300, c='black', marker='x', linewidth=4)
        
        plt.scatter(np.cos(state2 + np.pi/2), np.sin(state2 + np.pi/2), 
                    s=300, c='black', marker='o', 
                    edgecolor='black', linewidth=4)
        
        plt.annotate('', xy=(0.1, 1),
                 xytext=(0, 1),
                 arrowprops=dict(headwidth=13, color='black'))
        
    elif gamma == 1:
        state1 = 0
        
        plt.scatter(np.cos(state1 + np.pi/2), np.sin(state1 + np.pi/2), 
                    s=300, c='black', marker='x', linewidth=4)
        
    plt.annotate('', xy=(0.1, -1),
             xytext=(0, -1),
             arrowprops=dict(headwidth=13, color='black'))
        
    plt.gca().set_aspect('equal')
    plt.axis('off')
    
    plt.show()


if __name__ == '__main__':
    
    make_graph(0.1, 2000)
    make_graph(0.5, 2000)
    make_graph(1.01, 10000)
    make_graph(1.2, 2500)
    make_graph(5., 500)
    
    draw_phase_portrait(0.)
    draw_phase_portrait(0.7)
    draw_phase_portrait(1.)
    draw_phase_portrait(1.5)
    