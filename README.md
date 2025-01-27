# K-Means & K-Medoid Clustering on Pareto Optimal Solutions
### Last Updated: January 26th 2025
### Contributors: Sydney White, Michael Strauch, Felix Witing, Cordula Wittekind, Martin Volk
### [Paper Submitted]

## Overview
 - This framework performs k-means and k-medoids clustering on a set of Pareto optimal solutions derived from a multi-objective optimization algorithm. 
 - (Optional) A correlation matrix of the input variables is returned (the goal is to help users reduce the input variable count).
 - Before clustering occurs, the data is cast onto principal component axes and extreme solutions are handled (if desired). 
 - The code iterates through different possible inputs for the number of clusters, the number of principal components, and the variables that define 'extreme solutions'.
 - After these iterations, the best solution, as defined by silhouette score, is visualized in several ways: 
     - Representative solutions are plotted on up to 4 dimensions
     - Distributions of values within the clusters are plotted in a violin plot
     - (optional) Maps with the frequency of a trait can be plotted, if locational data (*.shp file) is provided

## Prerequisites
- Python >= 3.6
- R 4.4.X
- Required Python packages (installation instructions below):
  - pandas
  - scikit-learn
  - scikit-learn-extra
  - matplotlib
  - numpy
  - seaborn
  - scipy
  - configparser
  - pathlib

## Directory Structure
project_root/
<br>│
<br>├── README.md
<br>├── requirements.txt
<br>│
<br>├── input/
<br>│ ├── config.ini
<br>│ ├── [your_input_data].csv
<br>│ └── [shape_files]
<br>│
<br>├── python_files/
<br>│ ├── correlation_matrix.py
<br>│ ├── kmeans.py
<br>│ └── kmedoid.py
<br>│
<br>├── r_files/
<br>│ └── plot_frequency_maps.R
<br>│
<br>└── output/ (populated once code is run)
<br> &ensp;├── correlation_matrix.csv (if run)
<br> &ensp;├── kmeans_data_w_clusters_representativesolutions.csv (created with kmeans.py when Extreme Solutions are not handled) 
<br> &ensp;├── kmeans_data_w_clusters_representativesolutions_outliers.csv (created with kmeans.py when Extreme Solutions are handled)
<br> &ensp;├── kmedoid_data_w_clusters_representativesolutions.csv (created with kmedoid.py when Extreme Solutions are not handled)
<br> &ensp;├── kmedoid_data_w_clusters_representativesolutions_outliers.csv (created with kmedoid.py when Extreme Solutions are handled)
<br> &ensp;└── freq_map_cluster_X.png (if run)

## Setup
1. **Upload CSV File**: Place your CSV file containing Pareto optimal solutions data within the `input` folder. The CSV file should contain any columns you are interested in clustering or visualizing with values in float or integer format. Each row of the CSV file should represent a Pareto optimal solution. Additionally if you plan to utilize spatial mapping visualizations, provide a shapefile with all spatial units that were taken into account in the optimization (see below). In such a case, the CSV file should have additional columns formatted as 'UNIT_###' where ### represents the spatial unit's identification number. The data in these columns should be numerical values representing the optimization options (usually land use options) you're interested in plotting. Values for different options must be sequential, starting with 1.

2. **Update Config File**: Edit the `config.ini` file in the `input` folder to define all the necessary variables for the input code. More information available in `config.ini`.

3. **Upload Shape File**: Provide a shapefile with all spatial units that were taken into account in the optimization. The shapefile must contain a column named 'UNIT' containing the spatial unit's identification number, matching the ids (###) defined in the CSV input_file.

## Running the Code
1. Download dependencies: 
 - Python >3.6 (https://www.python.org/downloads/)
 - Required python packages:
   - If `pip` is not already installed, run `pip install pip` in the command prompt
   - Navigate to the project folder within the command prompt, run `pip install -r requirements.txt`
   - If you have issues downloading scikit-learn see https://scikit-learn.org/0.16/install.html for dependencies (Windows users require a C/C++ compiler (ex. Microsoft Visual C++ 14) --> instructions for downloading are available in the link).
 - Download R (https://cran.r-project.org/)
 - Note: It might be necessary to run the above using an administrator account.

2. Update `config.ini` within the `input` folder to define all variables
 
3. Correlation Matrix (Optional): To help define the variables you want to keep in the framework, you can first run correlation_matrix.py on your data. This will help identify highly correlated variables which can be considered for removal. Update the `config.ini` file if the input variables have changed.

4. There are two options for running the Python scripts; launching the pre-compiled Python executable (a) or using standard Python commands (b).
   
    4.a) Run kmeans.exe
   - navigate to the python_files folder
   - click on kmeans.exe
   - this will create a temporary Python environment, locate and prepare all necessary dependencies, start the Python interpreter and execute the script
   
    4.b) Run kmeans.py
   - open the command prompt
   - navigate to the python_files folder
   - run: `python kmeans.py`

   
 - This script will read the input CSV and configuration file, perform PCA, handle extreme solutions if specified, and apply K-means clustering. The best solution as defined by silhouette score will be graphed, and the results, including cluster assignments and representative solutions, will be saved in the output folder.
 - Output
   - The output files will be saved in the output folder and include:
   - kmeans_data_w_clusters_representativesolutions.csv (when outliers are not removed)
   - kmeans_data_w_clusters_representativesolutions_outliers.csv (when outliers are removed)
   - These files contain the original data with additional columns indicating the cluster assignments and representative solutions.
   - (if run) freq_map_cluster_X.png (frequency map images for each cluster)
     
5. There are two options for running the Python scripts; launching the pre-compiled Python executable (a) or using standard Python commands (b).
   
    5.a) Run kmedoid.exe
   - navigate to the python_files folder
   - click on kmedoid.exe
   
    5.b) Run kmedoid.py
   - open the command prompt
   - navigate to the python_files folder
   - run: `python kmedoid.py`

   
 - This script will read the input CSV and configuration file, perform PCA, handle extreme solutions if specified, and apply K-medoid clustering. The best solution as defined by silhouette score will be graphed, and the results, including cluster assignments and representative solutions, will be saved in the output folder.
  - Output
    - The output files will be saved in the output folder and include:
    - kmedoid_data_w_clusters_representativesolutions.csv (when extreme solutions are not handled)
    - kmedoid_data_w_clusters_representativesolutions_outliers.csv (when extreme solutions are handled)
    - These files contain the original data with additional columns indicating the cluster assignments and representative solutions.
    - (if run) freq_map_cluster_X.png (frequency map images for each cluster)

6. If unsatisfied with results, edit the `config.ini` file to rerun over a wider range of possible inputs

## Notes
 - If the user doesn't want to iterate over different values for principal components or extreme solution handeling, they can set the min/max values in the config file to the same value. This causes the code to only run for said singular number.
 - Ensure the directory structure is maintained as outlined above.
 - Modify the `config.ini` file as needed to reflect your specific data and requirements.
 - The script contains a plethora of comments explaining the code, if ever necessary, the user can edit the code for their specific needs.
 - It is possible for a cluster to be empty either during initialization of kmeans/kmedoids or in the final cluster. If this occurs during initialization, a warning will appear, but it's possible as the initialization continues, datapoints will subsequently be assigned to that once empty cluster. If the cluster remains empty after initialization, it simply won't show up in the results
 - For any issues or questions, please refer to the script comments or reach out to the project maintainers.
