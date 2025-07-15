# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_column_sources(column_list, source_table):
    """
    Given a list of variable (column) names and a metadata table, identify
    the original source of each variable.

    Parameters:
    - column_list: list of column names to check
    - source_table: pandas DataFrame containing at least 'attribute' and 'source' columns

    Returns:
    - unique_sources: a set of unique data sources among the selected columns
    - source_map: a dictionary mapping each column to its corresponding source
    """
    source_map = {}

    for col in column_list:
        # Search for the column in the source table
        row = source_table[source_table['attribute'] == col]
        if not row.empty:
            source_map[col] = row['source'].values[0]
        else:
            source_map[col] = 'Unknown'  # If not found, mark as unknown

    unique_sources = set(source_map.values())

    return unique_sources, source_map

def check_missing_row_overlap(df, column_group, group_name):
    """
    Analyzes how missing values overlap across a group of related columns.

    It helps determine whether missing values occur in the same rows (e.g.,
    all missing at once) or are scattered independently.

    Parameters:
    - df (DataFrame): Input dataset.
    - column_group (list): List of column names to check for missing overlap.
    - group_name (str): Label used in printed output.

    Returns:
    - Dictionary with:
        'union': rows with at least one missing value,
        'intersection': rows where all columns are missing,
        'difference': rows partially missing (not full overlap).
    """

    # Get the set of indices (rows) with missing values for each column in the group
    missing_index_sets = {
        col: set(df[df[col].isnull()].index)
        for col in column_group
    }

    # Union: all rows that have at least one missing value in the group
    union_index = set.union(*missing_index_sets.values())

    # Intersection: rows where all columns in the group have missing values
    intersection_index = set.intersection(*missing_index_sets.values())

    # Print summary information
    print(f"\n{group_name} Missing Row Check:")
    print(f"Total unique rows with at least one missing value: {len(union_index)}")
    print(f"Total rows missing in ALL columns: {len(intersection_index)}")

    # Compare the union and intersection
    if union_index == intersection_index:
        print("All missing values occur on the same rows.")
    else:
        print("Missing values do NOT occur on exactly the same rows.")
        diff_rows = union_index - intersection_index
        print(f"Number of differing rows (partial missing): {len(diff_rows)}")
        print(f"Examples: {list(diff_rows)[:10]}")

    # Return result as dictionary for optional further use
    return {
        "union": union_index,
        "intersection": intersection_index,
        "difference": union_index - intersection_index,
    }

def check_missing_and_plot(df, df_name):
    """
    Reports and visualizes missing data percentages per column.

    Parameters:
    - df (DataFrame): Input dataset.
    - df_name (str): Name to include in printed headers and plots.

    Returns:
    - DataFrame summarizing missing values and their percentages.
    """

    print(f"\n--- Missing Values in {df_name} ---")
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    dtypes = df.dtypes

    result = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': percent.values,
        'Dtype': dtypes.values
    })

    # Filter only columns with missing values
    result = result[result['Missing Count'] > 0]
    result.insert(0, 'DataFrame', df_name)

    if not result.empty:
        # Plot bar chart with labels
        plt.figure(figsize=(12, max(6, len(result) * 0.3)))
        ax = sns.barplot(
            y='Column', x='Missing %', data=result,
            color='darkorange', edgecolor='black'
        )
        plt.title(f"Missing Percentage per Column ({df_name})")
        plt.xlabel("Missing Percentage (%)")
        plt.ylabel("Column Name")
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        # Add value labels to each bar
        for i in range(len(result)):
            percent_val = result['Missing %'].values[i]
            ax.text(
                percent_val + 0.01,
                i,
                f"{percent_val:.5f}%",
                va='center',
                ha='left',
                fontsize=9,
                color='black'
            )

        plt.tight_layout()
        plt.show()
    else:
        print("No missing values found.")

    return result.reset_index(drop=True)

def check_missing(df, df_name):
    """
    Reports missing data counts and percentages per column (no plotting).

    Parameters:
    - df (DataFrame): Input dataset.
    - df_name (str): Name for printed summary.

    Returns:
    - DataFrame summarizing missing values and their percentages.
    """

    print(f"\n--- Missing Values in {df_name} ---")
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    result = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': percent.values
    })
    result = result[result['Missing Count'] > 0]
    result.insert(0, 'DataFrame', df_name)
    print(result if not result.empty else "No missing values found.")
    return result.reset_index(drop=True)

def plot_missing_spatial(df, cols_to_check, lon_col='lon', lat_col='lat'):
    """
    Visualizes the spatial location of missing values for selected columns
    on a scatter plot using latitude and longitude.

    Parameters:
    - df: pandas DataFrame
    - cols_to_check: list of column names to check for missing values
    - lon_col: name of the longitude column (default: 'lon')
    - lat_col: name of the latitude column (default: 'lat')
    """

    for col in cols_to_check:
        missing_rows = df[df[col].isnull()]

        plt.figure(figsize=(8, 6))
        
        # Non-missing data in grey
        plt.scatter(df[lon_col], df[lat_col], 
                    color='lightgrey', alpha=0.3, label='Non-missing')

        # Missing data in red
        plt.scatter(missing_rows[lon_col], missing_rows[lat_col], 
                    color='red', alpha=0.7, label='Missing')

        plt.title(f"Missing Value Spatial Distribution: {col}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        plt.show()

def move_column_after(df, column_to_move, after_column):
    """
    Move a specified column to immediately follow another column in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        column_to_move (str): The column name to be moved.
        after_column (str): The column name after which to insert the moved column.
        
    Returns:
        pd.DataFrame: A DataFrame with the column reordered.
    """
    cols = df.columns.tolist()
    new_order = []
    
    for col in cols:
        if col == column_to_move:
            continue  # skip it for now
        new_order.append(col)
        if col == after_column:
            new_order.append(column_to_move)
    
    return df[new_order]