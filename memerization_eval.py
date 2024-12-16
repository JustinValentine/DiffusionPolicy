import csv
import ast
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from tqdm import tqdm


def load_sketches_from_csv(file_path):
    """Loads sketches from a CSV file."""
    sketches = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                label = row[0]
                # Handle quoted or unquoted sketch data
                sketch_data = row[1].strip('"')
                sketch = ast.literal_eval(sketch_data)
                sketches.append((label, np.array(sketch)))
            except Exception as e:
                print(f"Error parsing row: {row} - {e}")
    return sketches


def pad_or_truncate(sketch, target_columns):
    """
    Ensures the sketch has the correct number of columns by padding or truncating.
    """
    sketch = np.array(sketch)
    if sketch.shape[1] > target_columns:  # Truncate extra columns
        return sketch[:, :target_columns]
    elif sketch.shape[1] < target_columns:  # Pad with zeros
        padding = np.zeros((sketch.shape[0], target_columns - sketch.shape[1]))
        return np.hstack((sketch, padding))
    return sketch


def hausdorff_distance(sketch1, sketch2):
    """
    Computes the Hausdorff distance between two sketches.
    Each sketch is a numpy array of points.
    """
    return max(
        directed_hausdorff(sketch1, sketch2)[0],
        directed_hausdorff(sketch2, sketch1)[0]
    )


def analyze_memorization(training_sketches, generated_sketches, threshold=0.01, target_columns=4):
    """
    Analyzes memorization by computing similarity between training and generated sketches.
    Returns a summary of matches based on a given threshold.
    """
    matches = []
    total_iterations = len(generated_sketches) * len(training_sketches)
    with tqdm(total=total_iterations, desc="Analyzing Memorization", unit="comparison") as pbar:
        for gen_label, gen_sketch in generated_sketches:
            gen_sketch = pad_or_truncate(gen_sketch, target_columns)
            for train_label, train_sketch in training_sketches:
                train_sketch = pad_or_truncate(train_sketch, target_columns)
                if gen_label == train_label:  # Compare sketches of the same class
                    distance = hausdorff_distance(gen_sketch, train_sketch)
                    if distance < threshold:
                        matches.append((gen_label, distance))
                pbar.update(1)
    return matches


if __name__ == "__main__":
    # File paths
    training_file = "/home/odin/DiffusionPolicy/data/doodle/30_class_data_train.csv"
    generated_file = "/home/odin/DiffusionPolicy/cnn/data_files/generated_data_uncond.csv"

    # Load the training and generated sketches
    training_sketches = load_sketches_from_csv(training_file)
    generated_sketches = load_sketches_from_csv(generated_file)

    # Analyze memorization
    threshold = 10.0  # Adjust threshold for Hausdorff distance
    target_columns = 4  # Adjust to the expected number of columns in your sketches
    matches = analyze_memorization(training_sketches, generated_sketches, threshold, target_columns)

    # Output results
    print(f"Found {len(matches)} matches with distance < {threshold}:")
    for label, distance in matches:
        print(f"Class: {label}, Distance: {distance:.2f}")