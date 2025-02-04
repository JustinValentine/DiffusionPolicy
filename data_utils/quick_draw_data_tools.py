import pandas as pd
import json
import csv


class DataTools():
    def __init__(self):
        self.classes = [
             "moon",
             "airplane",
             "fish",
             "umbrella",
             "train",
             "spider",
             "shoe",
             "apple",
             "lion",
             "bus",
        ]

    def format_data(self):
        for c in self.classes:
            with open(f"full_simplified_{c}.ndjson", 'r') as ndjson_f:
                # Initialize a CSV writer
                with open(f"full_simplified_{c}.csv", 'w', newline='') as csv_f:
                    writer = csv.DictWriter(csv_f, fieldnames=[])
                    
                    writer.fieldnames = ["word", "drawing"]
                    writer.writeheader()
                    # Read each line from the NDJSON file
                    for line in ndjson_f:
                        json_data = json.loads(line.strip())  # Parse JSON from the line
                    

                        # Write the data into the CSV file
                        writer.writerow(json_data)
	



def main():
    dt = DataTools()
    dt.format_data()
    


	
if __name__ == "__main__":
	main()