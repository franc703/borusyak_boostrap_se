import pandas as pd
import numpy as np

def boot_sample(your_data, unit_id, time, random_seed=None):
    """
    Draws units with replacement from all units in your_data.
    Returns a new bootstrap panel dataset with unique IDs for these units.

    Parameters:
    - your_data (DataFrame): Your panel data with specified columns.
    - unit_id (str): Column name for unit IDs.
    - time (str): Column name for time.
    - random_seed (int, optional): Seed for random number generation.
    
    Returns:
    - DataFrame: New data with bootstrapped samples and unique unit IDs.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if unit_id not in your_data.columns or time not in your_data.columns:
        raise ValueError("Specified unit_id or time column not found in the data.")
    
    # Get the unique units
    IDs = pd.DataFrame({'ID': np.unique(your_data[unit_id])})
    
    # The number of unique units
    N = IDs.shape[0]
    
    # Sample with replacement
    index = np.random.randint(0, N, size=N)
    bs_ID = IDs.iloc[index]
    
    # Add a column with the bootstrap sample number
    bs_ID['bs'] = np.arange(N)
    
    # Full bootstrap sample
    bs_data = your_data.merge(bs_ID, how='inner', left_on=unit_id, right_on='ID')
    
    # Make 'bs' the new unit_id
    bs_data[unit_id] = bs_data['bs']
    
    # Drop unnecessary columns and sort by new unit_id and time
    bs_data = bs_data.drop(['bs', 'ID'], axis=1).sort_values([unit_id, time])
    
    return bs_data