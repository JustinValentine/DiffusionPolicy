import pandas
import matplotlib.pyplot as plt
import numpy as np
from eval import generate
import os
from cnn.frechet_metric import get_score as fid

class PWExperiment():
    def __init__(self, p, w, models):
        self.data_file = f'./data/doodle/20_hot_data_train.csv'
        self.image_gen_path = "./pw_experiment_generations/generations.csv"
        self.pw_data_path = "./experiment_metrics/pw_metrics.txt"
        self.metric_file = open(self.pw_data_path, "w")
        self.p_values = p
        self.w_values = w
        self.model_files = models


    def generate_images(self, model_file, w):
        generate(model_file, '/tmp', '', generation_file=self.image_gen_path, num_samples=100, w=w) #Num samples is samples per class for generation


    def delete_images(self):
        os.remove(self.image_gen_path)


    def get_fid_score(self):
        return fid(self.data_file, self.image_gen_path, dataset_scope='dataset', feature_sizes=[2048])


    def get_iscore(self):
        pass


    def run_experiment(self):
        metrics = ''
        for p in self.p_values: # Go through each p value
            for w in self.w_values: # Go through each w value
                print(f'P: {p}, W: {w}')
                self.generate_images(self.model_files[p], w)
                output = self.get_fid_score()
                #Get Is score
                metrics += f'P: {p}, W: {w}, FID: {output[0]}, IS: {0}\n'
                self.delete_images()

        with open(self.pw_data_path, "w") as f:
            f.write(metrics)
            f.close()


if __name__ == "__main__":
    p_values = [0.01, 0.1, 0.25, 0.5]
    w_values = [0, 1, 2, 3, 4, 5]

    model_files = {
        0.01: '/home/odin/DiffusionPolicy/data/outputs/2024.12.14/17.50.54_flow_matching_doodle/checkpoints/epoch_2000.ckpt', # P = 0.01
        0.1: '/home/odin/DiffusionPolicy/data/outputs/2024.12.13/01.23.36_flow_matching_doodle/checkpoints/epoch_2500.ckpt', # P = 0.1
        0.25: '/home/odin/DiffusionPolicy/data/outputs/2024.12.13/13.41.08_flow_matching_doodle/checkpoints/epoch_2500.ckpt', # P = 0.25
        0.5: '/home/odin/DiffusionPolicy/data/outputs/2024.12.14/14.49.36_flow_matching_doodle/checkpoints/epoch_2500.ckpt', # P = 0.25
    }

    exp = PWExperiment(p_values, w_values, model_files)

    exp.run_experiment()
