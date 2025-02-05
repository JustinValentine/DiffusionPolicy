import pandas as pd
import json
import csv
from io import StringIO
from csv import writer
import tqdm
import ast
import os

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
            print(c)
            train = StringIO()
            csv = writer(train)
            
            with open(f"./data/full_simplified_{c}.ndjson", 'r') as ndjson_f:
                # Initialize a CSV writer
                train = StringIO()
                csv = writer(train)
                csv.writerow(["word", "drawing"])

                # Read each line from the NDJSON file
                for line in ndjson_f:
                    strokes = json.loads(line)["drawing"]
            
                    new_format = []

                    for i in range(len(strokes)):
                        stroke = strokes[i]
                        x = stroke[0]
                        y = stroke[1]

                        new_format.append([x[0], y[0], 0, 0])
                        for j in range(1, len(x)):
                            new_format.append([x[j], y[j], 1, 0])

                    new_format[-1][-1] = 1

                    # Write the data into the CSV file
                    csv.writerow([c, str(new_format)])

                ndjson_f.close()

            train.seek(0) # we need to get back to the start of the BytesIO
            df = pd.read_csv(train)

            df = df.sample(75000)

            df.to_csv(f'./data/full_simplified_{c}.csv', index=False)

        dfs = []
        for c in self.classes:
            df = pd.read_csv(f'./data/full_simplified_{c}.csv')
            dfs.append(df)
            os.remove(f'./data/full_simplified_{c}.csv')

        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined.to_csv('./data/combined_data.csv', index=False)





def main():
    dt = DataTools()
    dt.format_data()
    


	
if __name__ == "__main__":
	main()