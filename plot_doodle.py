import os
import csv
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np

def read_data(file_path):
    data_dict = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            name = row[0].strip()
            data_str = row[1].strip()
            data_list = ast.literal_eval(data_str)
            data_dict[name] = data_list
    return data_dict

def ensure_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def plot_drawing(data, doodle_name, output_folder="images", num=1):
    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_drawing_{num}.png")

    x_prev, y_prev = None, None  
    fig = plt.figure(figsize=(6, 6))
    plt.axis([0, 255, 0, 255])
    plt.gca().invert_yaxis() 

    for point in data:
        x, y, on_paper = point

        if on_paper > 0.0:  # If on_paper > 0.0, draw line
            if x_prev is not None and y_prev is not None:
                plt.plot([x_prev, x], [y_prev, y], color="black", linewidth=3)  # Draw a line

        x_prev, y_prev = x, y

    # plt.title(doodle_name)
    plt.axis('off')     # Remove axes, grid lines, ticks

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Image saved to {output_file}")

def plot_colored_drawing(data, doodle_name, output_folder="images", num=1):
    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_colored_{num}.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.invert_yaxis()  # Invert y-axis
    ax.axis('off')     # Remove axes, grid lines, ticks

    num_lines = sum(1 for i in range(1, len(data)) if data[i-1][2] > 0.0 and data[i][2] > 0.0)

    cmap = plt.cm.get_cmap("inferno")

    color_index = 0
    x_prev, y_prev = None, None
    for i, point in enumerate(data):
        x, y, on_paper = point
        if x_prev is not None and y_prev is not None and on_paper > 0.0:
            color = cmap(color_index / num_lines)
            ax.plot([x_prev, x], [y_prev, y], color=color, linewidth=4)
            color_index += 1
        x_prev, y_prev = x, y

    # plt.title(doodle_name)

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Image saved to {output_file}")

def plot_drawing_gif(data, doodle_name, output_folder="images", num=1):
    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_drawing_{num}.gif")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.invert_yaxis()  # Invert y-axis
    ax.axis('off')     # Remove axes

    frames = []
    x_prev, y_prev = None, None
    for point in data:
        x, y, on_paper = point
        if x_prev is not None and y_prev is not None and on_paper > 0.0:
            frames.append(((x_prev, y_prev), (x, y), on_paper > 0.0))
        x_prev, y_prev = x, y

    def update(i):
        if i < len(frames):  # Draw frames of the lines being added
            frame = frames[i]
            x_prev, y_prev = frame[0]
            x, y = frame[1]
            line, = ax.plot([x_prev, x], [y_prev, y], color="black", linewidth=3)
            return line,
        else:  # Hold the final drawing
            return []

    hold_frames = 1 * 10 
    total_frames = len(frames) + hold_frames

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True, repeat=False)

    ani.save(output_file, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"GIF saved to {output_file}")

data_dict = read_data('/home/odin/DiffusionPolicy/cnn/data_files/test_data.csv')

# Plot each doodle
output_folder = "images"
for doodle_name, data in data_dict.items():
    plot_drawing(data, doodle_name, output_folder=output_folder, num=5)
    plot_colored_drawing(data, doodle_name, output_folder=output_folder, num=5)
    plot_drawing_gif(data, doodle_name, output_folder=output_folder, num=5)
