import torch
from ignite.metrics import FrechetInceptionDistance

from cnn_utils import sequencesToDrawings
import pandas as pd

def main():

	df_sample = pd.read_csv('./data_files/data.csv')
	df_generated = pd.read_csv('./data_files/generatyed_data.csv')
	class_name = "smiley face"

	sample_imgs = sequencesToDrawings(df_sample[df_sample.iloc[:, 0].isin([class_name])].iloc[:, 1].tolist())
	generated_imgs = sequencesToDrawings(df_generated[df_generated.iloc[:, 0].isin([class_name])].iloc[:, 1].tolist())

	real_images_tensor = torch.tensor(real_images, dtype=torch.float32).permute(0, 3, 1, 2) # Shape: [N, C, H, W]
	generated_images_tensor = torch.tensor(generated_images, dtype=torch.float32).permute(0, 3, 1, 2)

	fid_metric = FrechetInceptionDistance(device="cuda" if torch.cuda.is_available() else "cpu")
	fid_metric.update((real_images_tensor, generated_images_tensor))
	fid_score = fid_metric.compute()

	print(f"FID Score: {fid_score}")

if __name__ == "__main__":
	main()