import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import matplotlib.pyplot as plt
import io
from PIL import Image

from diffusion_policy.policy.base_policy import BasePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_runner import BaseRunner


class DoodleRunner(BaseRunner):
    def __init__(self, 
            output_dir,
            n_classes=10
        ):
        super().__init__(output_dir)

        self.n_classes = n_classes


    def run(self, policy: BasePolicy):

        obs_dict = {
            "obs": {"class_quat": torch.arange(self.n_classes)}
        }

        policy.eval()

        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(1).to(policy.device))

        with torch.no_grad():
            gen_doodle = policy.predict_action(obs_dict)

        # Convert the tensor to a list of lists
        action_tensor = gen_doodle['action']  # Extract the action tensor
        action_tensor = action_tensor.cpu().numpy()  # Move to CPU and convert to NumPy (if on CUDA)

        # Flatten the first dimension (batch size or sequence size) if needed
        # action_array = action_tensor.reshape(-1, 4)  # Shape to (N, 4) where N is the number of steps
        log_data = {}
        i = 0
        for action in action_tensor:
            data = action.tolist()
            fig = self.plot_drawing(data)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            log_data['image/'+str(i)] = wandb.Image(Image.open(buf))
            i+=1
            plt.close()

        return log_data


    def plot_drawing(self, data, output_file="drawing_plot.png"):
        x_prev, y_prev = None, None  
        fig = plt.figure(figsize=(6, 6))
        plt.axis([0, 255, 0, 255])
        plt.gca().invert_yaxis() 

        for point in data:
            x, y, on_paper = point
      
            if on_paper > 0.0:  # If termination probability > 0.5, stop drawing
                if x_prev is not None and y_prev is not None:
                    plt.plot([x_prev, x], [y_prev, y], color="black")  # Draw a line

            x_prev, y_prev = x, y

        # plt.title("Generated Doodle")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(False)
        plt.axis('off')
       
        return fig
    
    def close(self):
        pass
    