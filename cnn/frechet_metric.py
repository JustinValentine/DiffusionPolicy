import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from cnn.cnn_utils import sequencesToDrawings
import pandas as pd
from io import StringIO
from csv import writer
from tqdm import tqdm


def calculate_fid(sample, generation, feature_sizes, class_name = False):
	
	# print("Creating Tensors")
	# Convert grayscale images to RGB by stacking the single channel three times
	sample_imgs_rgb = np.stack((sample,)*3, axis=-1)  # Shape: [N, H, W, 3]
	generated_imgs_rgb = np.stack((generation,)*3, axis=-1)  # Shape: [N, H, W, 3]

	# Convert to tensors and permute dimensions to [N, C, H, W]
	real_images_tensor = torch.tensor(sample_imgs_rgb, dtype=torch.uint8).permute(0, 3, 1, 2)
	generated_images_tensor = torch.tensor(generated_imgs_rgb, dtype=torch.uint8).permute(0, 3, 1, 2)

	# print("Creating Datasets")
	real_dataset = TensorDataset(real_images_tensor)
	generated_dataset = TensorDataset(generated_images_tensor)

	batch_size = 64
	real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
	generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	feature_scores = []
	for f_index in tqdm(range(len(feature_sizes)), desc="feature"):
		fid = FrechetInceptionDistance(feature=feature_sizes[f_index]).to(device)

		# print("Calculating FID Score")
		for real_batch in real_loader:
			fid.update(real_batch[0].to(device), real=True)
		for generated_batch in generated_loader:
			fid.update(generated_batch[0].to(device), real=False)

		fid_score = fid.compute()
		feature_scores.append(f"{fid_score}")

	if class_name:
		return [class_name] + [fs for fs in feature_scores]
	else:
		return [fs for fs in feature_scores]


def get_score(data_file, generated_file, output_file=None, dataset_scope='class', feature_sizes: list=[64, 192, 768, 2048]):
	#dataset_scope sets the comparison method, either compares fid for classes or for entire datasets
	print("Creating CSV")
	eval = StringIO()
	csv_writer = writer(eval)

	print("Loading data")
	df_sample = pd.read_csv(data_file)
	df_generated = pd.read_csv(generated_file)

	scores = [] # Used if output_file = None
	if dataset_scope == 'class':
		class_list = df_sample.iloc[:, 0].unique().tolist()
		for c_index in tqdm(range(len(class_list)), desc="Class"):
			class_name = class_list[c_index]
			# print("Generating Drawings")
			sample_imgs = np.array(sequencesToDrawings(df_sample[df_sample.iloc[:, 0].isin([class_name])].iloc[:, 1].tolist()))
			generated_imgs = np.array(sequencesToDrawings(df_generated[df_generated.iloc[:, 0].isin([class_name])].iloc[:, 1].tolist(), generated=True))

			output = calculate_fid(sample_imgs, generated_imgs, feature_sizes, class_name=class_name)
			if output_file:
				csv_writer.writerow(output)
			else:
				scores.append(output)

	elif dataset_scope == 'dataset':
		sample_imgs = np.array(sequencesToDrawings(df_sample.iloc[:, 1].tolist()))
		generated_imgs = np.array(sequencesToDrawings(df_generated.iloc[:, 1].tolist(), generated=True))

		output = calculate_fid(sample_imgs, generated_imgs, feature_sizes)
		if output_file:
			csv_writer.writerow(output)
		else:
			scores.append(output)

	if output_file == None:
		return scores
	else:
		eval.seek(0)  # Reset StringIO pointer to the beginning
		df = pd.read_csv(eval, header=None)
		df.columns = ['class'] + [str(feature_size) for feature_size in feature_sizes]
		df.to_csv(output_file, index=False)


def main():
	# File name quick access
	extra = "_uncond"
	data_file = f'./data_files/data{extra}.csv'
	generated_file = f'./data_files/generated_data{extra}.csv'
	output_file = f'./data_files/fid_metrics{extra}.csv'

	get_score(data_file, generated_file, output_file=output_file, dataset_scope='class')

if __name__ == "__main__":
	main()