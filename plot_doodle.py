import os
import csv
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import sys

# Increase the CSV field size limit to avoid _csv.Error: field larger than field limit
csv.field_size_limit(sys.maxsize)

def read_data(file_path):
    data_dict = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header row if there is one
        for row in reader:
            if len(row) < 2:
                continue  # Skip rows that don't have at least two fields
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

    x_prev, y_prev = None, None
    lines = []

    def init():
        return []

    def update(frame):
        point = data[frame]
        nonlocal x_prev, y_prev
        x, y, on_paper = point
        if x_prev is not None and y_prev is not None and on_paper > 0.0:
            line, = ax.plot([x_prev, x], [y_prev, y], color="black", linewidth=3)
            lines.append(line)
        x_prev, y_prev = x, y
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True, repeat=False)

    ani.save(output_file, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"GIF saved to {output_file}")

def plot_denoising_steps_gif(traj_data, doodle_name, output_folder="images", num=1):
    """
    Plot the denoising steps and save as a GIF.
    traj_data is a list of steps, where each step is a list of points.
    """
    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_denoising_{num}.gif")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.invert_yaxis()  # Invert y-axis
    ax.axis('off')     # Remove axes

    # Prepare data for each frame
    frames_data = []
    for step_data in traj_data:
        frames_data.append(step_data)

    # Number of frames
    num_frames = len(frames_data)

    def init():
        return []

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.invert_yaxis()  # Invert y-axis
        ax.axis('off')     # Remove axes

        data = frames_data[frame_idx]
        x_prev, y_prev = None, None
        for point in data:
            x, y, on_paper = point
            if x_prev is not None and y_prev is not None and on_paper > 0.0:
                ax.plot([x_prev, x], [y_prev, y], color="black", linewidth=3)
            x_prev, y_prev = x, y
        return []

    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, repeat=False)

    ani.save(output_file, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Denoising GIF saved to {output_file}")

def plot_denoising_steps_colored_gif(traj_data, doodle_name, output_folder="images", num=1):
    """
    Plot the denoising steps with colored lines and save as a GIF.
    traj_data is a list of steps, where each step is a list of points.
    """
    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_denoising_colored_{num}.gif")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.invert_yaxis()  # Invert y-axis
    ax.axis('off')     # Remove axes

    cmap = plt.cm.get_cmap("viridis")

    # Prepare data for each frame
    frames_data = traj_data + [traj_data[-1]] * (3 * 25)  # Hold the last frame for 3 seconds (3 seconds * 25 fps)

    # Number of frames
    num_frames = len(frames_data)

    def init():
        return []

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.invert_yaxis()  # Invert y-axis
        ax.axis('off')     # Remove axes

        data = frames_data[frame_idx]
        x_prev, y_prev = None, None

        num_lines = sum(1 for i in range(1, len(data)) if data[i-1][2] > 0.0 and data[i][2] > 0.0)
        color_index = 0

        for i, point in enumerate(data):
            x, y, on_paper = point
            x = x * (55) + 255 / 2
            y = y * (55) + 255 / 2
            if x_prev is not None and y_prev is not None and on_paper > 0.0:
                color = cmap(color_index / max(num_lines, 1))
                ax.plot([x_prev, x], [y_prev, y], color=color, linewidth=4)
                color_index += 1
            x_prev, y_prev = x, y

        return []

    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, repeat=False)

    ani.save(output_file, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f"Denoising Colored GIF saved to {output_file}")


if __name__ == "__main__":
    action_data_dict = read_data('/home/odin/DiffusionPolicy/cnn/data_files/action_data.csv')

    output_folder = "images"
    for doodle_name, data in action_data_dict.items():
        pass
        # plot_drawing(data, doodle_name, output_folder=output_folder, num=5)
        # plot_colored_drawing(data, doodle_name, output_folder=output_folder, num=5)
        # plot_drawing_gif(data, doodle_name, output_folder=output_folder, num=5)

    traj_data_dict = read_data('/home/odin/DiffusionPolicy/cnn/data_files/traj_data.csv')

    for doodle_name, traj_data in traj_data_dict.items():
        # plot_denoising_steps_gif(traj_data, doodle_name, output_folder=output_folder, num=5)
        plot_denoising_steps_colored_gif(traj_data, doodle_name, output_folder=output_folder, num=5)
