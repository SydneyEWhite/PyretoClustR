# Outputs a CSV file with the correlation matrix for the selected col_correlation_matrix
import pandas as pd
import configparser
from pathlib import Path

# Load configurations
config_path = Path('..') / 'input' / 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)

input_file = config['Data']['input_file']
col_correlation_matrix = config['Data']['col_correlation_matrix'].split(', ')

# Load data from the CSV file
input_path = Path('..') / 'input' / input_file
data = pd.read_csv(input_path)

# Subset the data to only include the columns of interest
data_subset = data[col_correlation_matrix]

# Calculate correlation coefficients
correlation_matrix = data_subset.corr()

# Save the correlation matrix to a CSV file
# Define output directory
output_path = Path('..') / 'output' 
output_path.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist
out_file_path = output_path / "correlation_matrix.csv"
correlation_matrix.to_csv(out_file_path, index=True)
