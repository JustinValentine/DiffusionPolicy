import matplotlib.pyplot as plt
import numpy as np

def softmax(values):
    exp_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
    return exp_values / np.sum(exp_values)

def plot_drawing(data, output_file="drawing_plot.png"):

    x_prev, y_prev = None, None  
    plt.figure(figsize=(6, 6))
    plt.axis([0, 255, 0, 255])
    plt.gca().invert_yaxis() 

    for point in data:
        x, y, on_paper, termination = point

        softmax_values = softmax([on_paper, termination])
        print(on_paper)
        # on_paper_prob = softmax_values[0]
        # termination_prob = softmax_values[1]

        if on_paper == 1: 
            if x_prev is not None and y_prev is not None:
                plt.plot([x_prev, x], [y_prev, y], color="black")  # Draw a line

        if termination < 0.0:
            break

        x_prev, y_prev = x, y


    plt.title("Generated Doodle")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # Save the figure to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close() 



data = [[59, 66, 0, 0], [67, 96, 1, 0], [80, 135, 1, 0], [93, 150, 1, 0], [109, 158, 1, 0], [124, 157, 1, 0], [146, 137, 1, 0], [155, 118, 1, 0], [158, 90, 1, 0], [158, 71, 1, 0], [157, 61, 1, 1]]

plot_drawing(data, output_file="drawing_output.png")