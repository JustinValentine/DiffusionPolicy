
import json
import matplotlib.pyplot as plt
from pathlib import Path
import click

# Function to parse logs from a file
def parse_logs(file_path):
    data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            log_entry = json.loads(line.strip())
            global_step = log_entry["global_step"]
            for key, value in log_entry.items():
                if key != "global_step":  # Skip global_step for value collection
                    if key not in data_dict:
                        data_dict[key] = {"global_step": [], "value": []}

                    data_dict[key]["global_step"].append(global_step)
                    data_dict[key]["value"].append(value)


    return data_dict

@click.command()
@click.option('-f', '--file_path', required=True)
def main(file_path):
    file_path = Path(file_path)
    data_dict = parse_logs(file_path)

    # Plotting
    
    # Iterate through the keys to plot each one
    for key, value in data_dict.items():
        if "max_reward" in key:
            continue
        plt.figure(figsize=(10, 5))
        plt.plot(value["global_step"], value["value"], marker='o', linestyle='-', label=key)

        plt.title('Log Data vs. Global Step')
        plt.xlabel('Global Step')
        plt.ylabel('Values')
        plt.grid()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
