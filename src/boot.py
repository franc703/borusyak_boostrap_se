import pandas as pd
import numpy as np
import statsmodels.api as sm

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


def compute_treatment_effects(data, outcome, treatment, unit_id, time_id, covariates=[]):
    # Prepare the variables for fixed-effects regression
    features = [outcome, treatment] + covariates
    
    # Dummy encode first and then split into treated and untreated
    full_data = pd.get_dummies(data, columns=[unit_id, time_id], drop_first=True)
    
    # Convert Boolean to Integer
    for col in full_data.select_dtypes(include=['bool']).columns:
        full_data[col] = full_data[col].astype(int)
    
    # Identify and skip truly textual columns
    skip_cols = []
    for col in full_data.select_dtypes(include=['object']).columns:
        if col not in features:
            try:
                full_data[col] = pd.to_numeric(full_data[col])
            except ValueError:
                print(f"Skipping column {col} as it appears to be truly textual.")
                skip_cols.append(col)
    
    full_data = full_data.drop(skip_cols, axis=1)
    
    # Split into treated and untreated groups
    data_untreated = full_data[full_data[treatment] == 0]
    data_treated = full_data[full_data[treatment] == 1]
    
    # Check for untreated units
    if len(data_untreated) == 0:
        raise ValueError("There are no untreated units for comparison.")
    
    # Create feature matrix and target for untreated units
    X = data_untreated.drop(features, axis=1)
    y = data_untreated[outcome]

    # Debug statements to understand data types and values
    # print("Debug Info: Data types of X columns:")
    # print(X.dtypes)
    # print("Debug Info: Data types of y:")
    # print(y.dtypes)
    
    # Run the two-way fixed effects regression on untreated units
    model = sm.OLS(y, sm.add_constant(X))
    results = model.fit()
    
    # Prepare the feature matrix for treated units
    X_treated = data_treated.drop(features, axis=1)
    
    # Add any missing columns to X_treated and fill with zeros
    for col in X.columns:
        if col not in X_treated.columns:
            X_treated[col] = 0
            
    X_treated = X_treated[X.columns]  # Reorder columns to match X
    
    # Compute predicted outcome for treated units
    data.loc[data[treatment] == 1, 'y_0'] = results.predict(sm.add_constant(X_treated))
    
    # Compute treatment effects
    data['t_effects'] = data[outcome] - data['y_0']
    
    return data


def aggregate_treatment_effects(data, treatment_column, estimand='overall', groupby_column=None, time_column=None):
    """
    Aggregate treatment effects into a specific estimand.
    
    Parameters:
        data (DataFrame): The data containing the treatment effects.
        treatment_column (str): The column name for treatment effects.
        estimand (str): The type of estimand to calculate ('overall', 'cohort', 'event').
        groupby_column (str): The column to group by for cohort-specific estimands.
        time_column (str): The time column for event-specific estimands.
    
    Returns:
        float or Series: The aggregated treatment effect estimand.
    """
    
    if estimand == 'overall':
        weights = 1 / data[treatment_column].count()
        weighted_treatment_effect = np.sum(weights * data[treatment_column])
        return weighted_treatment_effect
    
    elif estimand == 'cohort':
        if groupby_column is None or time_column is None:
            raise ValueError('Both groupby and time columns must be provided for cohort-specific estimands.')
        
        # Calculate elapsed time
        data['elapsed_time'] = data[time_column] - data[groupby_column]
        
        # Initialize an empty Series to hold the final weighted average treatment effects for each cohort
        weighted_avg_treatment_effects = pd.Series(dtype=float)
        
        # Loop through each cohort to calculate the weighted average treatment effect
        for cohort, cohort_data in data.groupby(groupby_column):
            group_data = cohort_data.groupby('elapsed_time')
            
            # Calculate average treatment effects for each elapsed time
            avg_treatment_effect = group_data[treatment_column].mean()
            
            # Count the number of units in each elapsed time
            counts_in_elapsed = group_data[treatment_column].count()
            
            # Calculate the weights
            weights = counts_in_elapsed / counts_in_elapsed.sum()
            
            # Compute the weighted average treatment effect for this cohort
            weighted_avg_treatment_effect = (avg_treatment_effect * weights).sum()
            
            # Append this to the final Series
            weighted_avg_treatment_effects[cohort] = weighted_avg_treatment_effect
        
        return weighted_avg_treatment_effects
    
    elif estimand == 'event':
        if groupby_column is None or time_column is None:
            raise ValueError('Both groupby and time columns must be provided for event-specific estimands.')
        
        # Calculate elapsed time
        data['elapsed_time'] = data[time_column] - data[groupby_column]
        
        # Group by cohort and elapsed time
        group_data = data.groupby([groupby_column, 'elapsed_time'])
        
        # Calculate average treatment effects for each cohort and elapsed time
        avg_treatment_effect = group_data[treatment_column].mean()
        
        # Count the number of units in each cohort and elapsed time combination
        counts_in_cohort_elapsed = group_data[treatment_column].count()
        
        # Count the number of units in each elapsed time across all cohorts
        total_counts_in_elapsed = data.groupby('elapsed_time')[treatment_column].count()
        
        # Calculate the weights
        weights = counts_in_cohort_elapsed / total_counts_in_elapsed.reindex(counts_in_cohort_elapsed.index, level=1)
        
        # Compute the weighted sum of treatment effects for each elapsed time
        weighted_avg_treatment_effect = (avg_treatment_effect * weights).groupby('elapsed_time').sum()
        
        return weighted_avg_treatment_effect
    
    else:
        raise ValueError('Invalid estimand type. Choose among "overall", "cohort", or "event".')