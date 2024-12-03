from tqdm import tqdm
import numpy as np
import cv2
import ast
import math
import json

def sn(num):
	'''
	Scaling the number
	'''
	return math.floor((num/255)*128)

def sequenceToDrawing(sequence, generated, size=(128, 128)):
	'''
	Converts the drawing sequence coordinates into an NP array representing the drawings
	'''
	img = np.zeros(size, dtype=np.uint8)

	for index in range(len(sequence)):
		if sequence[index][2] == 1 or (generated and sequence[index][2] > 0.0):
			x1, y1 = sequence[index-1][0], sequence[index-1][1]
			x2, y2 = sequence[index][0], sequence[index][1]
			img = cv2.line(img, (sn(x1), sn(y1)), (sn(x2), sn(y2)), 255, 2)
	return img/255


def sequencesToDrawings(data, generated=False):
	'''
	Converting the data list into a numpy array of images
	'''
	images = []
	for drawing in tqdm(data, desc="Sequences to Drawings") if not generated else data:
		sequence = ast.literal_eval(drawing)
		img = sequenceToDrawing(sequence, generated, size=(128, 128))
		images.append(img)
	return images


def onehotClasses(data, index_file):
	'''
	Get One Hot Encoding of Classes
	'''
	with open(index_file, 'r') as f:
		indexes = json.load(f)
	encodings = [[1 if name == index else 0 for index in indexes] for name in tqdm(data, desc="Classes To One Hots")]
	return encodings