import pandas as pd
import kagglehub
import shutil
from tqdm import tqdm
import json
import ast
import os
import re
from io import StringIO
from csv import writer
import random

class DataTools():
	def __init__(self):
		self.difficulty = "easy"
		self.class_name = "20_hot"

	def download_data(self):
		"""
		Only run this function if there is no file called doodle_data.csv in the data folder
		"""
		# Download the cleaned kaggle dataset for google quickdraw
		path = kagglehub.dataset_download("ashishjangra27/doodle-dataset")
		source = f'{path}/master_doodle_dataframe.csv'

		self.clean_data(source)

		match = re.search(r'/kagglehub/', source)
		truncated_path = source[: match.end()]

		shutil.rmtree(truncated_path)

	def clean_data(self, source):
		import pandas as pd

		# Load the CSV file
		df = pd.read_csv(source)
		df = df.drop(columns=['countrycode', 'key_id', 'recognized', 'image_path'])
		df = df[['word', 'drawing']] # Reorder the columns

		# Save the dataframe
		df.to_csv('./doodle_data.csv', index=False)

	def format_data(self):
		"""
		each point will be [x, y, pen_touching] where pen_touching is true if we want to go to that point with the pen on the paper, false of if we want to go to that point without contact
		"""
		print("Creating dataframe")
		df = pd.read_csv('./doodle_data.csv')

		train = StringIO()
		csv = writer(train)

		classes = df["word"].unique().tolist()
		if self.class_name:
			if self.class_name == "smiley_face":
				class_group = ["smiley face"]
			elif self.class_name == "circle_group":
				class_group = ["smiley face", "cat", "apple"]
			elif self.class_name == "20_hot":
				class_group = ["circle", "triangle", "anvil", "moon", "t-shirt", "star", "sun", "rainbow", "apple", "hand", "fish", "carrot", "bowtie", "bush", "chair", "envelope", "pizza", "mountain", "key", "butterfly", "smiley face", "bee", "bird", "diamond", "eyeglasses"]
		else:
			if self.difficulty == "easy":
				class_group = random.sample(classes, 10)
			elif self.difficulty == "medium":
				class_group = random.sample(classes, 25)
			elif self.difficulty == "hard":
				class_group = random.sample(classes, 100)

		if self.difficulty != "full":
			filtered_trajectories = df[df["word"].isin(class_group)]
			print(filtered_trajectories["word"].unique())

			print("Converting to nested list")
			data = filtered_trajectories.values.tolist()
		else:
			data = df.values.tolist()

		print("Writing each image to new format")
		for dp in tqdm(range(len(data)), desc="Datapoint"):
			
			strokes = ast.literal_eval(data[dp][1])
			new_format = []

			for i in range(len(strokes)):
				stroke = strokes[i]
				x = stroke[0]
				y = stroke[1]

				new_format.append([x[0], y[0], 0, 0])
				for j in range(1, len(x)):
					new_format.append([x[j], y[j], 1, 0])

			new_format[-1][-1] = 1
			csv.writerow([data[dp][0], str(new_format)])

		print("Writing data to file")
		train.seek(0) # we need to get back to the start of the BytesIO
		df = pd.read_csv(train)

		if self.class_name:
			prefix = self.class_name
		else:
			prefix = self.difficulty

		# Collect all class names to create a one-hot encoding
		classes = df.iloc[:,0].unique().tolist()

		classes = sorted(classes)
		class_to_index = {cls: idx for idx, cls in enumerate(classes)}

		with open(f'./outputs/{prefix}_class_index.json', 'w') as f:
		    json.dump(class_to_index, f)

		# Save the dataframe
		df.to_csv(f'./outputs/{prefix}_data_train.csv', index=False)


def main():
	dt = DataTools()

	while True:
		x = input("Options:\n\t(0) Quit\n\t(1) Download Data\n\t(2) Reformat Data\n: ")
		while x not in ["0", "1", "2", "3"]:
			x = input("Options:\n\t(1) Download Data\n\t(2) Reformat Data\n: ")
		if x == "0":
			break
		else:
			if x == "1":
				dt.download_data()
			elif x == "2":
				dt.format_data()


	
if __name__ == "__main__":
	main()