import pandas as pd
import numpy as np
import configparser
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import subprocess
#import time
#starttime = time.time()

def remove_outliers(data, deviation, count):
    # Omit extreme soltuions, data points where "count" or more variables within a datapoint are at least "deviation" standard deviations away from their respective means, from clustering (these later become their own representative solutions)
    z_scores = (data - data.mean()) / data.std()
    outliers = (np.abs(z_scores) > deviation)
    outliers_count = outliers.sum(axis=1)
    data_no_outliers = data[outliers_count < count]
    num_outliers = np.sum(outliers_count >= count)
    #print(data[outliers_count >= count]) # Prints the data points considered extreme solutions
    return data_no_outliers, num_outliers

def kmeans_clustering(pca_data, num_clusters):
    # Create clusters using the kmeans algorithm from the sklearn package
    # Return the cluster assignments and the coordinates of the cluster centers
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=100, random_state=58) # initialization methods can be k-means++ or random 
    kmeans.fit(pca_data)
    labels = kmeans.predict(pca_data)
    return labels, kmeans.cluster_centers_

def find_closest_pareto_solution(centroid, input_data):
    # Intake the center coordinates of a cluster and returns the closest Pareto optimal solution
    distances = cdist([centroid], input_data)
    closest_pos_idx = np.argmin(distances) # Returns the index within the distances array 
    closest_idx = input_data.index[closest_pos_idx] # Returns the index from the original data -- this step is required since the index within the distance array might not match the index of the input data if outliers are removed
    return closest_idx, input_data.loc[[closest_idx]]

def run_kmeans_multiple_times(pca_data, min_clusters, max_clusters, input_data, pca, fixed_clusters_boolean, fixed_clusters): #here input data has the outliers removed if outliers are removed from the dataset
    # Iterate through the kmeans algorithm to determine which number of clusters performs the best according to the silhouette score
    # Return the set of representative solutions with the highest silhouette score
    best_score = -1
    best_labels = None
    best_centroids = None
    if fixed_clusters_boolean == True:      # If the user has opted for a fixed number of clusters, the code will only run for fixed_clusters
        best_labels, best_centroids = kmeans_clustering(pca_data, fixed_clusters)
        #best_score = silhouette_score(pca_data, best_labels)
        best_score = silhouette_score(input_data, best_labels)
        dbi = davies_bouldin_score(input_data, best_labels)
        print(f"Clusters: {fixed_clusters}, Input Data Silhouette Score: {best_score:.4f}, Davies Bouldin Score: {dbi:.4f}")        
    elif fixed_clusters_boolean == False:   # If the user has NOT opted for a fixed number of clusters, the code will itereate through min_clusters to max_clusters inclusive
        for num_clusters in range(min_clusters, max_clusters + 1): # The "+1" makes the interation inclusive
            labels, centroids = kmeans_clustering(pca_data, num_clusters)
            #inputdata_sil_score = silhouette_score(pca_data, labels)
            inputdata_sil_score = silhouette_score(input_data, labels)
            dbi = davies_bouldin_score(input_data, labels)
            print(f"Clusters: {num_clusters}, Input Data Silhouette Score: {inputdata_sil_score:.4f}, Davies Bouldin Score: {dbi:.4f}")
            if inputdata_sil_score > best_score:
                best_score = inputdata_sil_score
                best_labels = labels
                best_centroids = centroids
    
    # Initialize a list of representative solutions
    representative_solutions = []
    representative_solutions_index = []

    # Unstandardize the centroids
    # Then cast the data (which is currently in terms of the principal components) back to the original axes
    input_axes_centroids = pca.inverse_transform(best_centroids)

    # Intake the un-transformed centroids of the clusters and finds the closest representative solution
    for centroid in input_axes_centroids:
        solution_index, closest_pareto_solution = find_closest_pareto_solution(centroid, input_data)
        representative_solutions.append(closest_pareto_solution)
        representative_solutions_index.append(solution_index)
    
    return best_labels, representative_solutions_index, representative_solutions, best_score

def non_integer_range(start, stop, step, precision=10):
    # This function allows the code to iterate with a step value that's not an integer
    current = start
    while current <= stop:
        yield current
        current += step
        current = round(current, precision) # necessary to prevent error buildup

def main():
    #### Load Configuration ####
    config_path = Path('..') / 'input' / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    input_file = config['Data']['input_file']
    columns = config['Data']['columns'].split(', ')
    min_clusters = config.getint('Clustering', 'min_clusters')
    max_clusters = config.getint('Clustering', 'max_clusters')
    fixed_clusters_boolean = config.getboolean('Clustering', 'fixed_clusters_boolean')
    fixed_clusters = config.getint('Clustering', 'fixed_clusters')
    handle_outliers_boolean = config.getboolean('Extreme_solutions', 'handle_outliers_boolean')
    deviations_min = config.getfloat('Extreme_solutions', 'deviations_min')
    deviations_max = config.getfloat('Extreme_solutions', 'deviations_max')
    deviations_step = config.getfloat('Extreme_solutions', 'deviations_step')
    count_min = config.getint('Extreme_solutions', 'count_min')
    count_max = config.getint('Extreme_solutions', 'count_max')
    outlier_to_cluster_ratio = config.getfloat('Extreme_solutions', 'outlier_to_cluster_ratio')
    min_components = config.getint('PCA', 'min_components')
    max_components = config.getint('PCA', 'max_components')
    num_variables_to_plot = config.getint('Plots','num_variables_to_plot')
    var_1 = config['Plots']['var_1']
    var_1_label = config['Plots']['var_1_label']
    var_2 = config['Plots']['var_2']
    var_2_label = config['Plots']['var_2_label']
    var_3 = config['Plots']['var_3']
    var_3_label = config['Plots']['var_3_label']
    var_4 = config['Plots']['var_4']
    var_4_label = config['Plots']['var_4_label']
    size_min = config.getfloat('Plots', 'size_min')
    size_max = config.getfloat('Plots', 'size_max')
    plot_frequency_maps = config.getboolean('Frequency_Plots','plot_frequency_maps')
    rscript_package_path = config['Frequency_Plots']['rscript_package_path']

    # Load input data
    input_path = Path('..') / 'input' / input_file
    raw_data = pd.read_csv(input_path)

    # Subset the data to only include the desired columns
    input_data = raw_data[columns].copy()

    # Define output directory
    output_path = Path('..') / 'output' 
    output_path.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

    final_score = -1
    final_labels = None
    final_rep_solutions = None
    final_rep_solutions_index = None
    final_input_data_no_outliers = None
    final_raw_data_no_oultiers = None
    final_num_outliers = None
    final_pca = None

    # If handle_outliers_boolean = True then extreme solutions will be defined based on the number of variables within the point (count) that are a certain number of standard deviations from their mean (deviations)
    # These solutions will then be ommitted from clustering and defined as their own representative solutions
    if handle_outliers_boolean == True:
        # Iterate through the range of principal component values
        for num_components in range(min_components, max_components + 1):
            print(f"\nNumber of Principal Components: {num_components}")

            # Define PCA
            pca = PCA(n_components=num_components)

            # Cache to store previously computed results
            cache = {}
            
            # Iterate through the range of deviation and count values used for defining extreme solutions
            for d in non_integer_range(deviations_min, deviations_max, deviations_step):
                for c in range(count_min, count_max+1):

                    # Temporarily remove extreme solutions before clustering
                    input_data_no_outliers, num_outliers = remove_outliers(input_data, d, c)

                    # It's possible that difference values for count and deviations result in the same outliers
                    # In order to prevent the code from repeating clustering for the same PCA and outliers, we cache the results

                    # Generate a unique key based on the index of the cleaned data (sorted for consistency)
                    data_hash_key = hash(tuple(sorted(input_data_no_outliers.index)))
                    #print(tuple(sorted(input_data_no_outliers.index)))
                    #print(data_hash_key)

                    # Check if this cleaned dataset has already been computed
                    if data_hash_key in cache:
                        print(f"\nnumber of extreme solutions: {num_outliers}, deviations: {d}, count: {c}")
                        num_clusters_cache, silhouette_score_cache, d_cache, c_cache = cache[data_hash_key]
                        print(f"this input led to the same outliers produced as for deviations = {d_cache} and count = {c_cache} thus the clustering results are the same.")
                        print(f"Best Cluster Count: {num_clusters_cache}, Best Input Data Silhouette Score: {silhouette_score_cache:.4f}")
                    else:
                        # Temporarily remove outliers from the original data
                        raw_data_no_outliers = raw_data[raw_data.index.isin(input_data_no_outliers.index)].copy()

                        # Cast the input data onto the defined number of principal components
                        principal_components = pca.fit_transform(input_data_no_outliers)
                        #print(pca.explained_variance_ratio_)
                        #print(pca.components_)


                        # Create a DataFrame to hold the principal component data (aka the input data cast onto the x principal component axes)
                        pca_data = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(num_components)])

                        # Perform k-means clustering and select the best result based on the highest silhouette score
                        print(f"\nnumber of extreme solutions: {num_outliers}, deviations: {d}, count: {c}")
                        labels, representative_solutions_index, representative_solutions, silhouette_score = run_kmeans_multiple_times(pca_data, min_clusters, max_clusters, input_data_no_outliers, pca, fixed_clusters_boolean, fixed_clusters)
                        
                        # Store results in cache using the hash key
                        cache[data_hash_key] = (max(labels)+1, silhouette_score, d, c)

                    # Define the final solution as the solution with the highest silhouette score; the number of extreme solutions must also be less than the number of clusters multiplied by the outlier_to_cluster_ratio
                    if ((silhouette_score > final_score) and (num_outliers < (len(representative_solutions_index) * outlier_to_cluster_ratio))):
                        final_score = silhouette_score
                        final_labels = labels
                        final_rep_solutions = representative_solutions
                        final_rep_solutions_index = representative_solutions_index
                        final_input_data_no_outliers = input_data_no_outliers
                        final_raw_data_no_outliers = raw_data_no_outliers
                        final_num_outliers = num_outliers
                        final_components = num_components
                        #pca_data.to_csv(output_path / "pca_data.csv", index=False) # Creates a CSV file with the pca data in case that's of interest

        # Error message if no solution fits the criteria: # of extreme solution < ( # of clusters * outlier_to_cluster_ratio )
        if final_rep_solutions == None:
            print("No acceptable representative solution set was found")
            print("This occurs when the number of extreme solutions is greater than the total number of rows multiplied by the outlier_to_cluster_ratio")
            print("In essence the goal is to prevent too many extreme solutions, so to resolve this issue either") 
            print("\t - Change the extreme solution criteria -- increase `deviations_max` or`count_max`")
            print("\t - Change the acceptable number of extreme solutions by increasing the outlier_to_cluster_ratio")
            exit()

        print(f"\nBest Silhouette Score: {final_score}, Number of Clusters: {len(final_rep_solutions_index)}, Number of Extreme Solutions: {final_num_outliers}, Num of PCA: {final_components}")
        
        # Add cluster labels to the dataframe
        final_raw_data_no_outliers['Cluster'] = final_labels

        # Add a column to the dataframe that indicates a representative solution
        final_raw_data_no_outliers.loc[final_raw_data_no_outliers.index.isin(final_rep_solutions_index), 'Representative_Solution'] = final_raw_data_no_outliers.loc[final_raw_data_no_outliers.index.isin(final_rep_solutions_index), 'Cluster'] #1

        # Create a dataframe of the outliers with the original columns
        outliers = raw_data[~raw_data.index.isin(final_input_data_no_outliers.index)].copy()

        # Create a dataframe of the entire original dataset with a column labeled Cluster which represents the cluster number of the data point and a column labeled Representative_Solution which is populated by the cluster for which it is a representative solution
        all_data = pd.concat([final_raw_data_no_outliers, outliers.assign(Cluster ='outlier',Representative_Solution='outlier')])

        # Export final dataframe to a CSV
        out_file_path = output_path / "kmeans_data_w_clusters_representativesolutions_outliers.csv"
        all_data.to_csv(out_file_path, index=False)

    # If extreme solutions are not to be handled
    else:        
        # Iterate through the range of principal component values
        for num_components in range(min_components, max_components + 1):
            print(f"\nNumber of Principal Components: {num_components}")

            # Define PCA
            pca = PCA(n_components=num_components)

            # Cast the input data onto the defined number of principal components
            principal_components = pca.fit_transform(input_data)

            # Create a DataFrame to hold the principal component data (aka the input data cast onto the x principal component axes)
            pca_data = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(num_components)])
            
            # Perform k-means and select the best result based on silhouette score
            labels, representative_solutions_index, representative_solutions, silhouette_score = run_kmeans_multiple_times(pca_data, min_clusters, max_clusters, input_data, pca, fixed_clusters_boolean, fixed_clusters)

            if (silhouette_score > final_score):
                final_score = silhouette_score
                final_labels = labels
                final_rep_solutions = representative_solutions
                final_rep_solutions_index = representative_solutions_index
                final_components = num_components
                #pca_data.to_csv(output_path / "pca_data.csv", index=False) # Creates a CSV file with the pca data incase that's of interest

        print(f"\nBest Silhouette Score: {final_score}, Number of Clusters: {len(final_rep_solutions_index)}, Num of PCA: {final_components}")

        # Add Cluster labels to the dataframe
        raw_data['Cluster'] = final_labels
        
        # Add a column to the dataframe that indicates a representative solution
        raw_data.loc[raw_data.index.isin(final_rep_solutions_index), 'Representative_Solution'] = raw_data.loc[raw_data.index.isin(final_rep_solutions_index), 'Cluster'] 
        
        # Export dataframe to a CSV
        out_file_path = output_path / "kmeans_data_w_clusters_representativesolutions.csv"
        raw_data.to_csv(out_file_path, index=False)

        all_data = raw_data


    ## Qualitative Clustering ##
    # To understand the data better qualitatively, return the number of points within a cluster that fall into specific percentile ranges
    # Create a DataFrame
    if num_variables_to_plot == 2:
        qualitative_clustering_columns = [var_1, var_2]
    elif num_variables_to_plot == 3:
        qualitative_clustering_columns = [var_1, var_2, var_3]
    elif num_variables_to_plot == 4:
        qualitative_clustering_columns = [var_1, var_2, var_3, var_4]
    else: 
        raise ValueError("num_variables_to_plot must be between 2 and 4")

    qualitative_clustering_columns_list = qualitative_clustering_columns.copy()
    qualitative_clustering_columns_list.append('Cluster')
    df = all_data[qualitative_clustering_columns_list].copy()

    # Function to create columns based on percentile ranges (inclusive of upper boundary)
    def create_binary_columns(series, lower_percentile, upper_percentile):
        lower_threshold = series.quantile(lower_percentile / 100)
        upper_threshold = series.quantile(upper_percentile / 100)
        if lower_percentile == 0:
            return((series <= upper_threshold).astype(int))
        else:
            return ((series > lower_threshold) & (series <= upper_threshold)).astype(int)

    # Define percentile ranges
    percentile_ranges = [(0, 33), (33, 66), (66, 100)]

    for col in qualitative_clustering_columns:
        for lower, upper in percentile_ranges:
            percentile_col_name = f"{col}_{lower}_{upper}"
            df[percentile_col_name] = create_binary_columns(df[col], lower, upper)

    # Drop the original columns
    df.drop(columns=qualitative_clustering_columns, inplace=True)

    # Group by 'cluster' and sum the percentile range columns
    df_grouped = df.groupby('Cluster').sum().reset_index()

    # Transpose the dataframe
    df_transposed = df_grouped.transpose()

    # Set the first row as the header
    new_header = df_transposed.iloc[0]
    df_transposed = df_transposed[1:]
    df_transposed.columns = new_header

    # Display the transposed DataFrame
    # print("Transposed DataFrame:")
    # print(df_transposed)

    # Define the table color as a list of the same color for each row/column
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(df_transposed)))
    header_color = (0.77236448,0.84513649,0.91186467,1.00)
    row_colors = [header_color] * len(df_transposed)  # One color for each row
    col_colors = [header_color] * len(df_transposed.columns)  # One color for each column

    # Plot the table
    fig1, ax1 = plt.subplots()
    ax1.axis('off')
    table = ax1.table(cellText=df_transposed.values
                    , colLabels=df_transposed.columns
                    , cellLoc='right'
                    , rowLabels=df_transposed.index
                    , rowColours=row_colors
                    , rowLoc='left'
                    , colColours=col_colors
                    , colLoc='right'
                    , loc='center')
    ax1.set_title('Percentile Distribution of Solutions within Clusters')

    plt.tight_layout()
    #print(time.time()-starttime)


    # Show the table
    plt.show()

    
    ## Plot the Representative Solutions ##
    sns.set_theme()

    # Create a figure with GridSpec
    fig2 = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])  # Allocate space for the plot and the legend

    # Create the main plot
    ax2 = fig2.add_subplot(gs[0])
    plt.title("Representative Solutions")

    # Plot Final Representative Solutions
    rep_data = all_data[all_data['Representative_Solution'].notnull()].copy()
    rep_data.rename(columns={var_3: var_3_label,var_4:var_4_label}, inplace=True)
    
    # Plot data on a 2-D axis using color and size of points as additional inputs when necessary
    if num_variables_to_plot == 4:
        sns.scatterplot(x=var_1,y=var_2, hue=var_3_label, size=var_4_label, sizes=(size_min,size_max), data=rep_data, palette = 'viridis', ax=ax2)
        ax2.get_legend().remove() # Remove the legend from the main plot    
    elif num_variables_to_plot == 3: 
        sns.scatterplot(x=var_1,y=var_2, hue=var_3_label, data=rep_data, palette = 'viridis', ax=ax2)
        ax2.get_legend().remove() # Remove the legend from the main plot
    elif num_variables_to_plot == 2: 
        sns.scatterplot(x=var_1,y=var_2, data=rep_data, ax=ax2) 
    else: 
        raise ValueError("num_variables_to_plot must be between 2 and 4")

    # Annotate points with the values from 'Representative_Solution'
    for line in range(0, rep_data.shape[0]):
        value = rep_data['Representative_Solution'].iloc[line]
        if isinstance(value, (int, float)):
            annotation = int(value)  # Convert to integer if it's a number
        else:
            annotation = value  # Keep as text if it's text
        ax2.text(rep_data[var_1].iloc[line],
             rep_data[var_2].iloc[line],
             annotation,
             horizontalalignment='left',
             size='medium',
             color='black',
             weight='semibold')
    
    # Set axis labels
    ax2.set_xlabel(var_1_label)
    ax2.set_ylabel(var_2_label)

    # Set axis limits if desired
    #ax2.set_xlim(1.7, 3.7)
    #ax2.set_ylim(-2, 34)

    # ax.set_xticks([1.75,2.25,2.75,3.25])
    # ax.set_yticks([0,10,20,30])

    # Creaate the legend in the allocated space
    ax2_legend = fig2.add_subplot(gs[1])
    ax2_legend.axis('off')  # Hide the axes for the legend subplot

    # Draw the legend
    if num_variables_to_plot == 4:
        handles, labels = ax2.get_legend_handles_labels()
        ax2_legend.legend(handles, labels, loc='center left')
    elif num_variables_to_plot == 3:
        handles, labels = ax2.get_legend_handles_labels()
        ax2_legend.legend(handles, labels, title =var_3_label, loc='center left')
    elif num_variables_to_plot == 2:
        pass

    # Show the plot
    plt.show()


    ## Violin Plots ##
    ## To learn more about the data points present in each cluster, find the distribution of the points in each clusters

    # Function to sort cluster names, handling non-integer values
    numerical_order = all_data['Cluster'].unique()
    if 'outlier' in numerical_order.tolist():
        numerical_order = numerical_order[numerical_order != 'outlier']
        # Determine the order of clusters numerically
        numerical_order = sorted(numerical_order, key=lambda x: int(x))
        numerical_order.append('outlier')
    else: 
        numerical_order = sorted(numerical_order, key=lambda x: int(x))

    # Set up the figure and axes for the subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Create a violin plot for each variable, grouped by the Cluster column

    # Violin plot for var_1
    sns.violinplot(x=all_data['Cluster'], y=var_1, data=all_data, ax=axes[0, 0], order=numerical_order)
    axes[0, 0].set_title(f'Distribution of {var_1} by Cluster')
    axes[0, 0].set_ylabel(var_1_label)

    # Violin plot for var_2
    sns.violinplot(x=all_data['Cluster'], y=var_2, data=all_data, ax=axes[0, 1], order=numerical_order)
    axes[0, 1].set_title(f'Distribution of {var_2} by Cluster')
    axes[0, 1].set_ylabel(var_2_label)

    if num_variables_to_plot >= 3:
        # Violin plot for var_3
        sns.violinplot(x=all_data['Cluster'], y=var_3, data=all_data, ax=axes[1, 0], order=numerical_order)
        axes[1, 0].set_title(f'Distribution of {var_3} by Cluster')
        axes[1, 0].set_ylabel(var_3_label)
    else:
        axes[1, 0].axis('off')
    
    if num_variables_to_plot == 4:
        # Violin plot for var_4
        sns.violinplot(x=all_data['Cluster'], y=var_4, data=all_data, ax=axes[1, 1], order=numerical_order)
        axes[1, 1].set_title(f'Distribution of {var_4} by Cluster')
        axes[1, 1].set_ylabel(var_4_label)
    else:
        axes[1, 1].axis('off')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    ## Frequency Plots ##
    # Run Frequency Plot from R Code
    if plot_frequency_maps == True:
        # Path to the executable R script
        r_script_path = Path('..') / 'r_files' / 'plot_frequency_maps.R'

        # Execute the R script using subprocess
        subprocess.call([rscript_package_path, r_script_path])    

    #### Additional Plots for Manual Editing ####
    # # Create 2D pairplot comparing the several variables (more readable for smaller variable count)
    # sns.pairplot(rep_data, vars=[list of column names])
    # plt.show()

    # # Plot a 3D scatter plot with color as the fourth dimension
    # fig = plt.figure()
    # ax3 = fig.add_subplot(projection='3d')
    # sp = ax3.scatter(rep_data[column1], rep_data[column2], rep_data[column3], c=rep_data[column4])
    # legend = ax3.legend(*sp.legend_elements(), title='Title', loc=2, bbox_to_anchor=(1.1,1))
    # ax3.add_artist(legend)
    # plt.show()


if __name__ == "__main__":
    main()

