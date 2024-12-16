import os
import csv
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
import json
from PIL import Image, UnidentifiedImageError
from matplotlib import cm

from matplotlib.colors import LinearSegmentedColormap


# Increase the CSV field size limit to avoid _csv.Error: field larger than field limit
csv.field_size_limit(sys.maxsize)

def read_data(file_path):
    data_dict = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header row if there is one
        i = 0
        for row in reader:
            if len(row) < 2:
                continue  # Skip rows that don't have at least two fields
            name = row[0].strip()
            data_str = row[1].strip()
            data_list = ast.literal_eval(data_str)
            data_dict[f"{name}_{i}"] = data_list
            i += 1
    return data_dict

def ensure_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def plot_drawing(data, doodle_name, output_folder="big_grid_images", num=1):
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


def plot_merged_drawing(output_folder):
    ensure_folder(output_folder)

    samples = False

    if samples:
        df = pd.read_csv('./cnn/data_files/data.csv')
    else:
        df = pd.read_csv('./cnn/data_files/generated_data_uncond.csv')
        
    c = df.iloc[:,0].unique().tolist()

    for c_index in tqdm(range(len(c)), desc="class"):
        class_name = c[c_index]
        sequences = df[df.iloc[:, 0].isin([class_name])].iloc[:1000, 1].tolist()

        if samples:
            output_file = os.path.join(output_folder, f"{class_name}_sample_merged_drawing.png")
        else:
            output_file = os.path.join(output_folder, f"{class_name}_generation_merged_drawing_uncond.png")

        x_prev, y_prev = None, None  
        fig = plt.figure(figsize=(6, 6))
        plt.axis([0, 255, 0, 255])
        plt.gca().invert_yaxis() 

        for index in tqdm(range(len(sequences))):
            s = sequences[index]
            sequence = ast.literal_eval(s)
            for point in sequence:
                if samples:
                    x, y, on_paper, terminate = point
                else:
                    x, y, on_paper = point

                if on_paper > 0.0:  # If on_paper > 0.0, draw line
                    if x_prev is not None and y_prev is not None:
                        plt.plot([x_prev, x], [y_prev, y], color="black", linewidth=0.2, alpha=0.5)  # Draw a line

                x_prev, y_prev = x, y

        # plt.title(doodle_name)
        plt.axis('off')     # Remove axes, grid lines, ticks

        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # print(f"Image saved to {output_file}")


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

def plot_denoising_steps_colored_gif(traj_data, doodle_name, output_folder="images", num=1, index=1):
    """
    Plot the denoising steps with colored lines and save as a GIF.
    traj_data is a list of steps, where each step is a list of points.
    """
    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_denoising_colored_{index}.gif")

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
            x = x * (60) + 255 / 2
            y = y * (60) + 255 / 2
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


def plot_denoising_steps_colored_static(traj_data, doodle_name, output_folder="images", num=1, index=1):
    """
    Plot all denoising steps with colored lines overlaid with low opacity.
    Final step is shown with full opacity.
    traj_data is a list of steps, where each step is a list of points.
    """
    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_denoising_colored_{index}.png")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.invert_yaxis()  # Invert y-axis
    ax.axis('off')  # Remove axes
    
    cmap = plt.cm.get_cmap("viridis")
    
    # Plot each step
    for step_idx, step_data in enumerate(traj_data):
        x_prev, y_prev = None, None
        num_lines = sum(1 for i in range(1, len(step_data)) if step_data[i-1][2] > 0.0 and step_data[i][2] > 0.0)
        color_index = 0
        
        for i, point in enumerate(step_data):
            x, y, on_paper = point
            x = x * (60) + 255 / 2
            y = y * (60) + 255 / 2
            
            if x_prev is not None and y_prev is not None and on_paper > 0.0:
                color = list(cmap(color_index / max(num_lines, 1)))
                # Set full opacity for final step, 0.1 for all others
                color[3] = 1.0 if step_idx == len(traj_data) - 1 else 0.015
                line_width = 4 if step_idx == len(traj_data) - 1 else 6
                ax.plot([x_prev, x], [y_prev, y], color=color, linewidth=line_width)
                color_index += 1
            x_prev, y_prev = x, y
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Denoising Colored static plot saved to {output_file}")


def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def plot_final_trajectories_with_noise_and_final(traj_data, doodle_name, output_folder="images", index=1):
    """
    Plot a static image showing:
    - The initial noise as grey dots in the background
    - Every 10th denoising step using viridis colormap at 10% opacity
    - The trajectories of vertices using a dark-to-light grey colormap (not reaching pure white)
    - The final denoised image lines on top using the reference logic
    """
    
    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_final_trajectories_with_noise_final.png")

    # Extract initial and final steps
    initial_data = traj_data[0]
    final_data = traj_data[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.invert_yaxis()
    ax.axis('off')

    def transform_coords(px, py):
        return px * 60 + 255 / 2, py * 60 + 255 / 2



    # Create custom colormap that doesn't reach pure white
    colors = [(0.2, 0.2, 0.2), (0.8, 0.8, 0.8)]  # From dark grey to light grey
    trajectory_cmap = LinearSegmentedColormap.from_list("custom_grey", colors)

    # Plot every 10th denoising step using viridis
    viridis = cm.get_cmap("viridis")
    step_indices = range(0, len(traj_data), 1)  # Every 10th step
    
    for step_idx in step_indices:
        step_data = traj_data[step_idx]
        x_prev, y_prev = None, None
        color = viridis(step_idx / len(traj_data))  # Color based on progress through denoising
        
        for i, point in enumerate(step_data):
            x, y, on_paper = point
            x, y = transform_coords(x, y)
            
            if x_prev is not None and y_prev is not None and on_paper > 0.0:
                ax.plot([x_prev, x], [y_prev, y], color=color, linewidth=4, alpha=0.1) # me !
            
            x_prev, y_prev = x, y

    # Plot the initial noise as grey dots
    init_xs = []
    init_ys = []
    for px, py, pon in initial_data:
        x_t, y_t = transform_coords(px, py)
        init_xs.append(x_t)
        init_ys.append(y_t)
    ax.scatter(init_xs, init_ys, s=10, color='grey', alpha=0.5, label='Initial Noise')

    # Identify final line segments
    final_segments = []
    current_segment = []
    for i, point in enumerate(final_data):
        _, _, on_paper = point
        if on_paper > 0.0:
            current_segment.append(i)
        else:
            if current_segment:
                final_segments.append(current_segment)
                current_segment = []
    if current_segment:
        final_segments.append(current_segment)

    # Plot trajectories with custom grey colormap
    for segment_indices in final_segments:
        for idx in segment_indices:
            vertex_positions = []
            for step_data in traj_data:
                px, py, pon = step_data[idx]
                x_t, y_t = transform_coords(px, py)
                vertex_positions.append((x_t, y_t))

            num_positions = len(vertex_positions)
            for i in range(1, num_positions):
                color = trajectory_cmap(i / (num_positions - 1))
                x_prev, y_prev = vertex_positions[i - 1]
                x_curr, y_curr = vertex_positions[i]
                ax.plot([x_prev, x_curr], [y_prev, y_curr], color=color, linewidth=3, alpha=1.0)

    # Draw the final denoised image lines
    data = final_data
    x_prev, y_prev = None, None
    cmap = cm.get_cmap("viridis")
    
    num_lines = sum(1 for i in range(1, len(data)) if data[i-1][2] > 0.0 and data[i][2] > 0.0)
    color_index = 0

    for i, point in enumerate(data):
        x, y, on_paper = point
        x, y = transform_coords(x, y)
        if x_prev is not None and y_prev is not None and on_paper > 0.0:
            color = cmap(color_index / max(num_lines, 1))
            ax.plot([x_prev, x], [y_prev, y], color=color, linewidth=4)
            color_index += 1
        x_prev, y_prev = x, y

    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Final trajectory image with noise and final lines saved to {output_file}")

    

def create_grids(image_dir, class_index_path, output_dir, frame_duration=100):
    # Load class index
    with open(class_index_path, 'r') as f:
        class_index = json.load(f)
    
    classes = class_index.keys()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for cls in classes:
        # Gather relevant files for the class
        gif_files = []
        png_files = []
        for filename in os.listdir(image_dir):
            # Only include files that match the format {class}_{index}_drawing.{ext}
            if filename.startswith(cls) and '_drawing' in filename:
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1].isdigit():  # Check format and valid index
                    if filename.endswith('.gif'):
                        gif_files.append(filename)
                    elif filename.endswith('.png'):
                        png_files.append(filename)
        
        # Sort the files by the index (assumes the format {class}_{index}_drawing.ext)
        gif_files.sort(key=lambda x: int(x.split('_')[1]))
        png_files.sort(key=lambda x: int(x.split('_')[1]))
        
        # Limit to the first 25 files
        gif_files = gif_files[:25]
        png_files = png_files[:25]
        
        # Create 5x5 grid for gifs
        if gif_files:
            gif_frames = []
            max_frames = 0
            images_per_gif = []

            for f in gif_files:
                try:
                    gif = Image.open(os.path.join(image_dir, f))
                    frames = []
                    while True:
                        frames.append(gif.copy())
                        try:
                            gif.seek(gif.tell() + 1)
                        except EOFError:
                            break
                    images_per_gif.append(frames)
                    max_frames = max(max_frames, len(frames))
                except (UnidentifiedImageError, FileNotFoundError):
                    print(f"Skipping invalid or corrupted GIF file: {f}")

            if images_per_gif:
                # Extend frames for GIFs with fewer frames to match the max
                for frames in images_per_gif:
                    while len(frames) < max_frames:
                        frames.append(frames[-1])

                # Subsample frames to reduce animation duration if needed
                subsample_rate = max(1, max_frames // 50)  # Cap at 50 frames
                gif_frames = []

                for frame_idx in range(0, max_frames, subsample_rate):
                    grid_images = [images[frame_idx] for images in images_per_gif]
                    grid_frame = create_image_grid(grid_images, (5, 5))
                    gif_frames.append(grid_frame)

                # Add the last frame with half the frame duration
                grid_images = [images[-1] for images in images_per_gif]
                last_frame = create_image_grid(grid_images, (5, 5))
                for i in range(25):
                    gif_frames.append(last_frame)

                # Save the animated grid GIF
                gif_frames[0].save(
                    os.path.join(output_dir, f"{cls}_grid.gif"),
                    save_all=True,
                    append_images=gif_frames[1:],
                    duration=[frame_duration] * (len(gif_frames) - 1) + [frame_duration // 2],
                    loop=0
                )
        
        # Create 5x5 grid for pngs
        if png_files:
            png_images = []
            for f in png_files:
                try:
                    png_images.append(Image.open(os.path.join(image_dir, f)))
                except (UnidentifiedImageError, FileNotFoundError):
                    print(f"Skipping invalid or corrupted PNG file: {f}")
            
            if png_images:
                png_grid = create_image_grid(png_images, (5, 5))
                png_grid.save(os.path.join(output_dir, f"{cls}_grid.png"))

def create_image_grid(images, grid_size):
    """Create a grid of images."""
    if not images:
        raise ValueError("No valid images provided for the grid.")
    
    width, height = images[0].size
    grid_width = width * grid_size[0]
    grid_height = height * grid_size[1]
    grid_image = Image.new('RGBA', (grid_width, grid_height))
    
    for idx, image in enumerate(images):
        x = (idx % grid_size[0]) * width
        y = (idx // grid_size[0]) * height
        grid_image.paste(image, (x, y))
    
    return grid_image

def create_40x22_grid(image_dir, class_index_path, output_dir, grid_size=(40, 22), cell_size=(30, 30)):
    # Load class index
    with open(class_index_path, 'r') as f:
        class_index = json.load(f)
    
    classes = class_index.keys()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for cls in classes:
        # Gather relevant PNG files for the class
        png_files = []
        for filename in os.listdir(image_dir):
            if filename.startswith(f"{cls}_drawing") and filename.endswith('.png'):
                png_files.append(filename)
        
        # Skip if no files are found
        if not png_files:
            print(f"No PNG files found for class '{cls}'. Skipping...")
            continue
        
        # Sort the files by the index
        png_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        # Limit the number of images to fit the grid
        total_cells = grid_size[0] * grid_size[1]
        if len(png_files) < total_cells:
            png_files = (png_files * (total_cells // len(png_files) + 1))[:total_cells]
        else:
            png_files = png_files[:total_cells]
        
        # Create 40x22 grid
        grid_width, grid_height = grid_size[0] * cell_size[0], grid_size[1] * cell_size[1]
        grid_image = Image.new('RGBA', (grid_width, grid_height))
        
        for idx, filename in enumerate(png_files):
            try:
                img = Image.open(os.path.join(image_dir, filename)).resize(cell_size)
                x = (idx % grid_size[0]) * cell_size[0]
                y = (idx // grid_size[0]) * cell_size[1]
                grid_image.paste(img, (x, y))
            except (UnidentifiedImageError, FileNotFoundError):
                print(f"Skipping invalid or corrupted PNG file: {filename}")
        
        # Save the grid image
        output_path = os.path.join(output_dir, f"{cls}_grid_40x22.png")
        grid_image.save(output_path)
        print(f"Saved: {output_path}")


def plot_denoising_steps_colored_image(traj_data, doodle_name, output_folder="images", num_steps=6, index=1):
    """
    Plot the denoising steps with colored lines and save as a single image.
    traj_data is a list of steps, where each step is a list of points.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ensure_folder(output_folder)
    output_file = os.path.join(output_folder, f"{doodle_name}_denoising_colored_{index}.png")

    fig, axes = plt.subplots(1, num_steps, figsize=(18, 6), dpi=100, constrained_layout=True)
    cmap = plt.cm.get_cmap("viridis")

    # Ensure the first and last steps are included, with evenly spaced steps in between
    step_indices = [0] + [int(i * (len(traj_data) - 1) / (num_steps - 1)) for i in range(1, num_steps)]

    for ax, step_idx in zip(axes, step_indices):
        data = traj_data[step_idx]
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_aspect('equal')  # Ensure the scaling is correct
        ax.invert_yaxis()  # Invert y-axis
        ax.axis('off')     # Remove axes

        x_prev, y_prev = None, None
        num_lines = sum(1 for i in range(1, len(data)) if data[i-1][2] > 0.0 and data[i][2] > 0.0)
        color_index = 0

        for i, point in enumerate(data):
            x, y, on_paper = point
            x = x * (60) + 255 / 2
            y = y * (60) + 255 / 2
            if x_prev is not None and y_prev is not None and on_paper > 0.0:
                color = cmap(color_index / max(num_lines, 1))
                ax.plot([x_prev, x], [y_prev, y], color=color, linewidth=4)
                color_index += 1
            x_prev, y_prev = x, y

        # Add step label
        if step_idx == 98:
            step_idx = 99
        ax.set_title(f"Step {step_idx}", fontsize=10)

    # Save the final image
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Denoising Colored Image saved to {output_file}")



if __name__ == "__main__":

    output_folder = "images"

    # Plot all images of a class overlayed on top of each other images 
    # plot_merged_drawing(output_folder)

    # Plot basic plots for all gnerated actions
    # action_data_dict = read_data('./cnn/data_files/generated_data_uncond.csv')

    # for doodle_name, data in action_data_dict.items():
    #     if doodle_name.split("_")[0] in ['duck']:
    #         plot_drawing(data, doodle_name.split("_")[0], output_folder=output_folder, num=doodle_name.split("_")[1])
    #         # plot_colored_drawing(data, doodle_name, output_folder=output_folder, num=5)
    #         plot_drawing_gif(data, doodle_name, output_folder=output_folder, num=5)
    #     else:
    #         continue

    # Plot denoising steps as gif
    traj_data_dict = read_data('/home/odin/DiffusionPolicy/cnn/data_files/traj_data_good_fig.csv')

    for doodle_name, traj_data in traj_data_dict.items():        
        # plot_denoising_steps_colored_gif(traj_data, doodle_name, output_folder=output_folder, num=6, index=1)
        # if (doodle_name.split('_')[0] == 'circle' or doodle_name.split('_')[0] == 'triangle' or doodle_name.split('_')[0] == 'star'):
            # plot_final_trajectories_with_noise_and_final(traj_data, doodle_name, output_folder="images", index=1)
            # plot_denoising_steps_colored_static(traj_data, doodle_name, output_folder="images", num=1, index=1)
        plot_denoising_steps_colored_image(traj_data, doodle_name, output_folder="images", num_steps=6, index=1)
    
    # Plot gird of classes
    # create_grids('./images', './data/doodle/20_hot_class_index.json', './output_grids')
    # create_40x22_grid('./images', './data/doodle/20_hot_class_index.json', './output_grids')
