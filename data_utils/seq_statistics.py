import csv
import ast
import matplotlib.pyplot as plt
import numpy as np
import sys

# Increase the CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)

def compute_statistics(sequence_lengths):
    """
    Compute mean, median, standard deviation of sequence lengths.
    """
    mean_length = np.mean(sequence_lengths)
    median_length = np.median(sequence_lengths)
    std_length = np.std(sequence_lengths)
    return mean_length, median_length, std_length

def plot_histogram(sequence_lengths, stats):
    """
    Plot a histogram of sequence lengths and annotate with statistics.
    """
    mean_length, median_length, std_length = stats

    # Create the histogram
    plt.figure(figsize=(12, 6))
    bins = max(10, len(set(sequence_lengths)) // 10)  # Dynamically determine bin count
    plt.hist(sequence_lengths, bins=bins, color='skyblue', alpha=0.7, edgecolor='black')

    # Add vertical lines for mean and median
    plt.axvline(mean_length, color='red', linestyle='--', label=f"Mean: {mean_length:.2f}")
    plt.axvline(median_length, color='green', linestyle='--', label=f"Median: {median_length:.2f}")

    # Title and labels
    plt.title("Trajectory Length Histogram")
    plt.xlabel("Trajectory Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Show or save the plot
    plt.tight_layout()
    plt.show()

def main():
    csv_file = "output.csv"  # Path to your CSV file
    sequence_lengths = []   # To store lengths of each sequence

    # Read the CSV file
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # Second column contains the sequence as a string
            stroke_str = row[1]

            # Convert the string representation to a Python list
            data_points = ast.literal_eval(stroke_str)

            # Append the length of this sequence
            sequence_lengths.append(len(data_points))

    # Compute statistics
    stats = compute_statistics(sequence_lengths)
    print(f"Mean Length: {stats[0]:.2f}")
    print(f"Median Length: {stats[1]:.2f}")
    print(f"Standard Deviation: {stats[2]:.2f}")

    # Plot the histogram
    plot_histogram(sequence_lengths, stats)

if __name__ == "__main__":
    main()