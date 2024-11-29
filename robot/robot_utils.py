

def scaleCoords(x, y, bounds: tuple):
	"""
	bounds is a tuple in the format min_x, min_y, max_x, max_y
	
	returns x and y
	"""
	new_x = (x/255) * (bounds[2] - bounds[0]) + bounds[0]
	new_y = (y/255) * (bounds[3] - bounds[1]) + bounds[1]

	return new_x, new_y


def main():
	pass


if __main__ == "__main__":
	main()