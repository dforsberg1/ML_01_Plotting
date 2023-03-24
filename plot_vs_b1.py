import numpy as np
import matplotlib.pyplot as plt

# Some additional functions

# Create function to compute the b1 gradient at a specific point on the cost function
def compute_gradient_at_point(x, y, b0, b1):
    m = x.shape[0]  # Number of training examples
    dj_db0 = 0
    dj_db1 = 0

    for i in range(m):  
        f_b = b0 + (b1 * x[i])
        dj_db0_i = f_b - y[i] 
        dj_db1_i = (f_b - y[i]) * x[i] 
        dj_db0 += dj_db0_i
        dj_db1 += dj_db1_i 
    dj_db0 = dj_db0 / m
    dj_db1 = dj_db1 / m 
    
    if abs(dj_db1) < 1e-4:
        return 0.0
    else:
        return dj_db1
    
# Define the partial derivative w.r.t. b1
def partial_derivative_b1(x, y, b0, b1):
    n = len(x)
    error = (b0 + (b1 * x)) - y
    return np.sum(error * x) / n

# Define tangent line
# y = m*(x - x1) + y1
def tangent_line(x, x1, y1, slope):
    return slope*(x - x1) + y1

# This function checks whether a line segment connecting two points lies within specified bounds.
def inbounds(a,b,xlim,ylim):
    xlow,xhigh = xlim
    ylow,yhigh = ylim
    ax, ay = a
    bx, by = b
    if (ax > xlow and ax < xhigh) and (bx > xlow and bx < xhigh) \
        and (ay > ylow and ay < yhigh) and (by > ylow and by < yhigh):
        return True
    return False

def plot_vs_b1(x_train_0, y_train_0, b0_final_0, b1_final_0, compute_cost_0):
    # Round final parameter estimates to nearest integer (for graphing purposes)
    b0_final = round(b0_final_0,0)
    b1_final = round(b1_final_0,0)
    # Define range and step size for b1
    b1_values = np.arange(b1_final - 200, b1_final + 200, 1)
    
    # Calculate costs # this is y = f(x) and is a parabola
    costs = [compute_cost_0(x_train_0, y_train_0, b0_final, b1) for b1 in b1_values]
    offset = b1_final - (min(b1_values) + b1_final)/2
    
    # Identify specific points on cost function for plotting
    b1_points = np.array([b1_final - offset, b1_final, b1_final + offset])
    cost_points = [compute_cost_0(x_train_0, y_train_0, b0_final, b1) for b1 in b1_points]
    
    # Calculate the gradients at specific points
    b1_gradients = np.round([compute_gradient_at_point(x_train_0, y_train_0, b0_final, b1) for b1 in b1_points])
    
    """
    Create a graph that shows the cost function vs. b1 when b0 = b0_in
    """

    # Plot the cost vs. b1
    plt.ticklabel_format(style='plain', axis='y')
    plt.plot(b1_values, costs)

    # Plot and label gradients at specific points
    plt.scatter(b1_points, cost_points, color = 'darkred', zorder = 3)
    plt.text(b1_points[0] - 12.5, cost_points[0] - 1000, r'$\frac{\partial J}{\partial \beta_1}$' + ' = ' + f"{round(b1_gradients[0])}", ha='right')
    plt.text(b1_points[1], cost_points[1] + 3e3, r'$\frac{\partial J}{\partial \beta_1}$' + ' = ' + f"{round(b1_gradients[1])}", ha='center')
    plt.text(b1_points[2] + 12.5, cost_points[2] - 1000, r'$\frac{\partial J}{\partial \beta_1}$' + ' = ' + f"{round(b1_gradients[2])}", ha='left')

    # Iterate through each point in b1_points
    for b1 in b1_points:
        b1range = np.linspace(b1-25, b1+25, 10)
        y1 = compute_cost_0(x_train_0, y_train_0, b0_final, b1)  # choose point to plot tangent line
        slope1 = partial_derivative_b1(x_train_0, y_train_0, b0_final, b1)  # establish slope parameter
        
        plt.plot(b1range, tangent_line(b1range, b1, y1, slope1), color='darkred', linestyle='dashed', linewidth=2)

    # Create labels for x-axis, y-axis, and main title
    plt.xlabel(r"$\beta_1$ with $\beta_0$ = " + f"{b0_final:.0f}")
    plt.ylabel('Cost')
    plt.title(r'Cost vs. $\beta_1$')

    # Show plot
    plt.show()
