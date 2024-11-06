from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np
from wf_dynamics import ode_wrapper_for_system, sat

def test_sat():
    assert sat(0.9, -1, 1) == 0.9
    assert sat(-1.2, -1, 1) == -1
    assert sat(1.2, -1, 1) == 1
    print('Saturation function test successful')
    
test_sat()

def test_system():
    # System
    dsdt = ode_wrapper_for_system
    # Initial conditions
    y0 = np.zeros(14)
    # Speed
    y0[3] = 250 * 1852 / 3600 # 250 kts
    y0[12] = 210 * 1852 / 3600
    # Altitude
    y0[2] = 22000 * 0.3048
    y0[13] = 11000 * 0.3048 # descend to 11000 ft
    print('Target GS: ', y0[12])
    print('Target ALT: ', y0[13])
    # Heading change
    y0[5] = 0
    y0[11] = 270 * np.pi / 180 # turn to 90 degrees
    sol = solve_ivp(dsdt, t_span=[0,600], y0=y0)
    
    # Plot all the fundamental states
    plt.figure(figsize=(10,9))
    plt.title('Simulated Aircraft States')
    
    plt.subplot(3,3,1)
    plt.plot(sol.t, sol.y[0,:]) # x
    plt.title('x')
    
    plt.subplot(3,3,2)
    plt.plot(sol.t, sol.y[1,:]) # y
    plt.title('y')
    
    plt.subplot(3,3,3)
    plt.plot(sol.t, sol.y[2,:]) # h
    plt.title('Alt')
    
    plt.subplot(3,3,4)
    plt.plot(sol.t, sol.y[3,:]) # V
    plt.title('GS')
    
    plt.subplot(3,3,5)
    plt.plot(sol.t, sol.y[4,:]) # gamma 
    plt.title('FPA')
    
    plt.subplot(3,3,6)
    plt.plot(sol.t, sol.y[5,:]) # psi
    plt.title('HDG')
    
    plt.subplot(3,3,7)
    plt.plot(sol.t, sol.y[6,:]) # phi
    plt.title('Roll')
    
    plt.tight_layout()
    
    plt.show()

test_system()