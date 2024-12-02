#Local Imports
from cnn_utils import sequencesToDrawings, onehotClasses
from model import CNNModel

#External Libraries
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import json
import click

#Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CNNTrainer():
	'''
	This class is for training and evaluating the CNNModel in model.py
	'''
	def __init__(self):
		self.classes = None
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = CNNModel().to(self.device)

	def train(self):

		#Process all the data
		df = pd.read_csv('./data_files/data.csv')
		self.classes = df.iloc[:, 0].unique().tolist()
		validateClassData = []
		validateImageData = []
		trainClassData = []
		trainImageData = []
		for item in self.classes:
			images = sequencesToDrawings(df[df.iloc[:, 0].isin([item])].iloc[:, 1].tolist())
			encodings = onehotClasses(df[df.iloc[:, 0].isin([item])].iloc[:, 0].tolist())
			validateClassData += encodings[:450]
			validateImageData += images[:450]
			trainClassData += encodings[450:]
			trainImageData += images[450:]

		validateClassData = np.array(validateClassData)
		validateImageData = np.array(validateImageData)
		trainClassData = np.array(trainClassData)
		trainImageData = np.array(trainImageData)
		
		# Convert data to PyTorch tensors
		train_images_tensor = torch.tensor(trainImageData, dtype=torch.float32).unsqueeze(1)
		train_labels_tensor = torch.tensor(np.argmax(trainClassData, axis=1), dtype=torch.long)
		validate_images_tensor = torch.tensor(validateImageData, dtype=torch.float32).unsqueeze(1)
		validate_labels_tensor = torch.tensor(np.argmax(validateClassData, axis=1), dtype=torch.long)

		train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
		validate_dataset = TensorDataset(validate_images_tensor, validate_labels_tensor)

		train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
		validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)

		# Define the model, criterion, and optimizer
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.model.parameters(), lr=0.001)

		# Training loop
		epochs = 70
		tLoss = []
		vLoss = []
		vAccuracy = []
		for epoch in tqdm(range(epochs), desc="Epoch:"):
			self.model.train()
			running_loss = 0.0
			for images, labels in train_loader:
				images, labels = images.to(self.device), labels.to(self.device)

				outputs = self.model(images)  # No softmax during training
				loss = criterion(outputs, labels)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				running_loss += loss.item()

			# Validation
			self.model.eval()
			val_loss = 0.0
			correct = 0
			total = 0
			with torch.no_grad():
				for images, labels in validate_loader:
					images, labels = images.to(self.device), labels.to(self.device)

					outputs = self.model(images, applySoftmax=True)  # Apply softmax for probabilities
					loss = criterion(outputs, labels)
					val_loss += loss.item()

					_, predicted = torch.max(outputs, 1)
					correct += (predicted == labels).sum().item()
					total += labels.size(0)

			tLoss.append(running_loss / len(train_loader))
			vLoss.append(val_loss / len(validate_loader))
			vAccuracy.append(100 * correct / total)

			# Checkpoint the model
			if epoch % 5 == 0:
				self.write_metrics(tLoss, vLoss, vAccuracy, epoch)

				torch.save(self.model.state_dict(), './model-state.pt')
			
		# Training Finished
		torch.save(self.model.state_dict(), 'model-state.pt')

	def write_metrics(self, tLoss, vLoss, vAccuracy, epoch):
		# Number of epochs (this will be the length of any of the lists)
		epochs = range(1, len(tLoss) + 1)

		# Directory where the images will be saved
		directory = "training_metrics"

		# Create a figure and axes
		plt.figure(figsize=(12, 6))

		# Plot Training Loss
		plt.subplot(1, 3, 1)
		plt.plot(epochs, tLoss, label="Training Loss", color='b')
		plt.title("Training Loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.grid(True)
		plt.legend()

		# Plot Validation Loss
		plt.subplot(1, 3, 2)
		plt.plot(epochs, vLoss, label="Validation Loss", color='r')
		plt.title("Validation Loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.grid(True)
		plt.legend()

		# Plot Validation Accuracy
		plt.subplot(1, 3, 3)
		plt.plot(epochs, vAccuracy, label="Validation Accuracy", color='g')
		plt.title("Validation Accuracy")
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy (%)")
		plt.grid(True)
		plt.legend()

		# Adjust layout to prevent overlap
		plt.tight_layout()

		# Save the plot as a PNG file in the training_metrics directory
		plt.savefig(f"./training_metrics/epoch_{epoch}_training_metrics_plot.png")
		plt.close()

	def test(self):
		print("Loading model")
		self.model.load_state_dict(torch.load('model-state.pt', weights_only=True))
		self.model.eval()
		
		print("Loading data")
		df = pd.read_csv('./data_files/test_data.csv')
		c = df.iloc[:,0].tolist()
		s = df.iloc[:,1].tolist()
		drawings = np.array(sequencesToDrawings(s, generated=True))

		output_vectors = []

		with open('./data_files/data_index.json', 'r') as f:
			indexes = json.load(f)

		# Process all drawings
		print("Processing drawings")
		wrong = 0
		total = len(drawings)
		keys = list(indexes.keys())
		output = ""
		for i, drawing in enumerate(drawings):
			# Convert the drawing into a tensor
			image_tensor = torch.tensor(drawing, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			image_tensor = image_tensor.squeeze(2).to(self.device)  # Remove unnecessary dimensions

			# Get the model's output
			with torch.no_grad():
				output_vector = self.model(image_tensor, applySoftmax=True).cpu().numpy().flatten()
			
			output_vectors.append(output_vector)

			actual = c[i]
			predicted = keys[output_vector.argmax()]
			if actual != predicted:
				wrong += 1
				output += f"Actual: {actual}"+" "*(20-len(c[i]))+f"Predicted: {predicted}\n"

			# Only plot the first image
			if i == 0:
				# Display the first image with the output vector in the legend
				fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size to leave room for the text
				ax.imshow(drawing, cmap='gray')
				output_str = ''.join([f'{v}: {output_vector[i]:.2f}\n' for i, v in enumerate(indexes.keys())])

				# Position the text to the right of the image
				plt.text(1.05, 0.5, f'{output_str}', transform=ax.transAxes, fontsize=12, verticalalignment='center')
				plt.title("Model Prediction Example")

				# Adjust layout to prevent clipping
				plt.subplots_adjust(right=0.75)  # Adjust this to leave enough space for the text

				plt.show()

		accuracy = ((total-wrong)/total )* 100
		top = f"Accuracy: {accuracy}%\n"
		top += f"Classes tested: {len(df.iloc[:,0].unique())}\n"
		top += output

		with open("./data_files/cnn_output.txt", "w") as f:
			f.write(top)
			f.close()
		# print(f"Accuracy: {accuracy}%")
		# print(f"Classes tested: {len(df.iloc[:,0].unique())}")

@click.command()
@click.option('-m', '--mode', required=True)
def main(mode):
	trainer = CNNTrainer()
	if mode == "train":
		trainer.train()
	elif mode == "eval":
		trainer.test()

if __name__ == "__main__":
	main()