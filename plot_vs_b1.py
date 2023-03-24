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

def plot_cost_vs_b1(x_train_0, y_train_0, b0_final_0, b1_final_0, compute_cost_0):
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

def quiver_plot(x_train_0, y_train_0, b0_final_0, b1_final_0, compute_gradient_0):
    
    """
    Quiver Plot
    """
    
    # Adjust size of plot
    plt.rcParams['figure.dpi'] = 150
    
    # Round final parameter estimates to nearest integer (for graphing purposes)
    b0_final = round(b0_final_0,0)
    b1_final = round(b1_final_0,0)
    
    # Set up grid of values for b_0 and b_1
    b_0, b_1 = np.meshgrid(np.linspace(b0_final_0 - 300, b0_final_0 + 300, 10), np.linspace(b1_final_0 - 300, b1_final_0 + 300, 10))
    
    # Compute gradients at each point on the grid
    grad_b1 = np.zeros_like(b_0)
    grad_b0 = np.zeros_like(b_1)
    for i in range(b_1.shape[0]):
        for j in range(b_0.shape[1]):
            grad_b1[i][j], grad_b0[i][j] = compute_gradient_0(x_train_0, y_train_0, b_1[i][j], b_0[i][j])
    
    # Set color array based on magnitude of gradients
    n=-2
    color_array = np.sqrt(((grad_b0-n)/2)**2 + ((grad_b1-n)/2)**2)
    
    # Create a quiver plot of the gradients
    plt.quiver(b_0, b_1, grad_b0, grad_b1, color_array, units='width')
    plt.xlabel(r'$ \beta_0$')
    plt.ylabel(r'$ \beta_1$')
    plt.title(r'Quiver Plot of Gradients Over Values of $\beta_0$ and $\beta_1$')
    
    # Display the plot
    plt.show()

def cost_vs_iteration(x_train_0, y_train_0, b0_final_0, b1_final_0, compute_cost_0, J_hist_0):
    
    # Round final parameter estimates to nearest integer (for graphing purposes)
    b0_final = round(b0_final_0,0)
    b1_final = round(b1_final_0,0)
    # Define range and step size for b1
    b1_values = np.arange(b1_final - 200, b1_final + 200, 1)
    
    # Calculate costs # this is y = f(x) and is a parabola
    costs = [compute_cost_0(x_train_0, y_train_0, b0_final, b1) for b1 in b1_values]

    """
    Plot cost vs. iteration step
    """

    # Adjust size of plot
    plt.rcParams['figure.dpi'] = 100

    # Plot cost versus iteration
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,5))
    ax1.plot(J_hist_0[:101]) 
    ax2.plot(J_hist_0[:1001])
    ax1.set_title("Cost vs. Iteration (start)");  ax2.set_title("Cost vs. Iteration (end)")
    ax1.set_ylabel('Cost');  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('Iteration Step');  ax2.set_xlabel('Iteration Step') 

    # Set the format of the tick labels on each y-axis
    ax1.yaxis.set_major_formatter('{:.0f}'.format)
    ax2.yaxis.set_major_formatter('{:.0f}'.format)

    # Display the plot
    plt.show()

def contour_plot(x_train_0, y_train_0, b0_final_0, b1_final_0, compute_cost_0, p_hist_0):

    """
    2D Contour plot of cost(b0,b1) over a range of values for b0 and b1
    """
    
    # Adjust size of plot
    plt.rcParams['figure.dpi'] = 80
    
    # Round final parameter estimates to nearest integer (for graphing purposes)
    b0_final = round(b0_final_0,0)
    b1_final = round(b1_final_0,0)

    # Define a function to format the label values
    def label_formatter(val):
        """
      Function to format the label values.

      Parameters:
      val (float): The value of the label.

      Returns:
      str: The formatted string value of the label.
      """
        return str(int(round(np.exp(val),0)))

    # Caclulate total cost given optimized parameters
    cost_final = compute_cost_0(x_train_0, y_train_0, b0_final, b1_final)

    # Define the b0 and b1 range
    b0_range = np.linspace(b0_final - 300, b0_final + 300, 100)
    b1_range = np.linspace(b1_final - 300, b1_final + 300, 100)

    # Define the contour levels
    levels = [np.log(cost) for cost in [100, 1000, 5000, 12000, 25000, 50000, 100000]]

    # Create a 2D meshgrid of the b0 and b1 values
    b0, b1 = np.meshgrid(b0_range, b1_range)

    # Compute the cost for each combination of b0 and b1
    cost_vals = np.zeros_like(b0)
    for i in range(b0.shape[0]):
        for j in range(b1.shape[0]):
            cost_vals[i,j] = compute_cost_0(x_train_0, y_train_0, b0[i,j], b1[i,j])

    # Create a contour plot of the cost function
    fig, ax = plt.subplots(1,1, figsize=(12, 8))

    # Use the dictionary dlc to map color names to their hexadecimal codes
    dlc = dict(dlblue='#0096ff', dlorange='#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
    dlcolors = [dlc['dlblue'], dlc['dlorange'], dlc['dldarkred'], dlc['dlmagenta'], dlc['dlpurple']]

    # Create contour lines with specified colors and levels
    contour = ax.contour(b0, b1, np.log(cost_vals), levels=levels, linewidths=2, alpha=0.7, colors=dlcolors)

    # Label contour lines with formatted values
    ax.clabel(contour, inline=True, fontsize=7, fmt=label_formatter)

    # Set the labels for the x and y axes
    ax.set_xlabel(r'$ \beta_0$', fontsize = 15)
    ax.set_ylabel(r'$ \beta_1$', fontsize = 15)

    # Set the title & subtitle for the plot
    fig.suptitle(r'Contour Plot of Cost vs. ($\beta_0$, $\beta_1$)', fontsize = 27)
    ax.set_title(r'J($\beta_0$='+f'{b0_final:.0f}, '+r'$\beta_1$='+f'{b1_final:.0f}) = ' + f'{cost_final:.0f}', fontsize=20)            

    # Plot the purple dotted lines pointing to minimum cost
    ax.plot([ax.get_xlim()[0], b0_final], [b1_final, b1_final], lw=2, color='purple', ls='dotted')
    ax.plot([b0_final, b0_final], [ax.get_ylim()[0], b1_final], lw=2, color='purple', ls='dotted')
    ax.scatter(x=[b0_final], y=[b1_final], c='purple', zorder = 3, s = 10) 

    # Plot the path of gradient descent showing gradient size every 10 steps
    hist = p_hist_0
    step = 10
    resolution = 5

    # Initialize empty list to store arrow coordinates
    arrow_coords = []

    # Initialize variable called "base"
    b0_min, b0_max = min(b0_range), max(b0_range)
    b1_min, b1_max = min(b1_range), max(b1_range)

    for i in range(len(hist)):
        if ((hist[i][0] > b0_min) and (hist[i][0] < b0_max)) and ((hist[i][1]> b1_min) and (hist[i][1]<b1_max)):
            base_ind = i
            break
    base = hist[base_ind]

    # for loop to plot gradients of gradient descent as red arrows
    for point in hist[0::step]:
        # Normalize the gradient to get a unit vector
        edist = np.sqrt((base[0] - point[0])**2 + (base[1] - point[1])**2)
        if(edist > resolution or point==hist[-1]):
            if inbounds(point, base, ax.get_xlim(),ax.get_ylim()):
                plt.annotate('', xy=point, xytext=base, xycoords='data',
                         arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 3},
                         va='center', ha='center', zorder = 4)
                # Append arrow coordinates to list
                arrow_coords.append((base[0], base[1], point[0], point[1]))
                base=point

    # Convert list of arrow coordinates to pandas DataFrame
    # df_arrow_coords = pd.DataFrame(arrow_coords, columns=['x1', 'y1', 'x2', 'y2'])

    # Print the DataFrame
    # print(df_arrow_coords)

    # Display the plot
    plt.show()

def contour_plot_zoom(x_train_0, y_train_0, b0_final_0, b1_final_0, compute_cost_0, p_hist_0):

    """
    2D Contour plot again but zoomed in
    """

    # Adjust size of plot
    plt.rcParams['figure.dpi'] = 80

    # Round final parameter estimates to nearest integer (for graphing purposes)
    b0_final = round(b0_final_0,0)
    b1_final = round(b1_final_0,0)
    
    # Define a function to format the label values
    def label_formatter(val):
        """
      Function to format the label values.

      Parameters:
      val (float): The value of the label.

      Returns:
      str: The formatted string value of the label.
      """
        return str(int(round(np.exp(val),0)))
    
    # Caclulate total cost given optimized parameters
    cost_final = compute_cost_0(x_train_0, y_train_0, b0_final, b1_final)

    # Define the b0 and b1 range
    b0_range = np.linspace(b0_final - 20, b0_final + 20, 100)
    b1_range = np.linspace(b1_final - 20, b1_final + 20, 100)

    # Define the contour levels
    levels = [np.log(cost) for cost in [10, 40, 100, 250, 500, 1000]]

    # Create a 2D meshgrid of the b0 and b1 values
    b0, b1 = np.meshgrid(b0_range, b1_range)

    # Compute the cost for each combination of b0 and b1
    cost_vals = np.zeros_like(b0)
    for i in range(b0.shape[0]):
        for j in range(b1.shape[0]):
            cost_vals[i,j] = compute_cost_0(x_train_0, y_train_0, b0[i,j], b1[i,j])

    # Create a contour plot of the cost function
    fig, ax = plt.subplots(1,1, figsize=(12, 8))

    # Use the dictionary dlc to map color names to their hexadecimal codes
    dlc = dict(dlblue='#0096ff', dlorange='#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
    dlcolors = [dlc['dlblue'], dlc['dlorange'], dlc['dldarkred'], dlc['dlmagenta'], dlc['dlpurple']]

    # Create contour lines with specified colors and levels
    contour = ax.contour(b0, b1, np.log(cost_vals), levels=levels, linewidths=2, alpha=0.7, colors=dlcolors)

    # Label contour lines with formatted values
    ax.clabel(contour, inline=True, fontsize=7, fmt=label_formatter)

    # Set the labels for the x and y axes
    ax.set_xlabel(r'$ \beta_0$', fontsize = 15)
    ax.set_ylabel(r'$ \beta_1$', fontsize = 15)

    # Set the title & subtitle for the plot
    fig.suptitle(r'Contour Plot of Cost vs. ($\beta_0$, $\beta_1$)', fontsize = 27)
    ax.set_title(r'J($\beta_0$='+f'{b0_final:.0f}, '+r'$\beta_1$='+f'{b1_final:.0f}) = ' + f'{cost_final:.0f}', fontsize=20)            

    # Plot the purple dotted lines pointing to minimum cost
    ax.plot([ax.get_xlim()[0], b0_final], [b1_final, b1_final], lw=2, color='purple', ls='dotted')
    ax.plot([b0_final, b0_final], [ax.get_ylim()[0], b1_final], lw=2, color='purple', ls='dotted')
    ax.scatter(x=[b0_final], y=[b1_final], c='purple', zorder = 3, s = 10) 

    # Plot the path of gradient descent showing gradient size every 20 steps
    hist = p_hist_0
    step = 20
    resolution = 0.5

    # Initialize empty list to store arrow coordinates
    arrow_coords = []

    # Initialize variable called "base"
    b0_min, b0_max = min(b0_range), max(b0_range)
    b1_min, b1_max = min(b1_range), max(b1_range)
    for i in range(len(hist)):
        if ((hist[i][0] > b0_min) and (hist[i][0] < b0_max)) and ((hist[i][1]> b1_min) and (hist[i][1]<b1_max)):
            base_ind = i
            break
    base = hist[base_ind]

    # for loop to plot gradients of gradient descent as red arrows
    for point in hist[0::step]:
        # Normalize the gradient to get a unit vector
        edist = np.sqrt((base[0] - point[0])**2 + (base[1] - point[1])**2)
        if(edist > resolution or point==hist[-1]):
            if inbounds(point, base, ax.get_xlim(),ax.get_ylim()):
                plt.annotate('', xy=point, xytext=base, xycoords='data',
                         arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 3},
                         va='center', ha='center', zorder = 4)
                # Append arrow coordinates to list
                arrow_coords.append((base[0], base[1], point[0], point[1]))
                base=point

    # Convert list of arrow coordinates to pandas DataFrame
    # df_arrow_coords = pd.DataFrame(arrow_coords, columns=['x1', 'y1', 'x2', 'y2'])

    # Print the DataFrame
    # print(df_arrow_coords)

    # Display the plot
    plt.show()

