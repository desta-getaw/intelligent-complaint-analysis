# File: src/preprocess.py

import pandas as pd
import numpy as np
import re # For regular expressions
import os # For creating directories and path manipulation

def run_preprocessing():
    """
    Loads the full CFPB complaint dataset, filters it, cleans the consumer
    complaint narratives, and saves the cleaned and filtered dataset.
    """
    print("--- Starting Data Preprocessing (src/preprocess.py) ---")

    # --- 0. Set up Paths ---
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(project_root, os.pardir))

    input_csv_path = os.path.join(project_root, 'data', 'raw', 'complaints.csv')
    output_data_dir = os.path.join(project_root, 'data', 'filtered')

    try:
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
            print(f"Created directory: {output_data_dir}")
    except OSError as e:
        print(f"Error creating output directory {output_data_dir}: {e}")
        print("Please check directory permissions.")
        return

    output_file_path = os.path.join(output_data_dir, 'filtered_complaints.csv')

    # --- 1. Load the full CFPB complaint dataset ---
    df = None
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Dataset loaded successfully from '{input_csv_path}'!")
        print(f"Initial total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    except FileNotFoundError:
        print(f"Error: '{input_csv_path}' not found.")
        print("Please ensure the CSV file is located at 'data/raw/complaints.csv' relative to your project root.")
        # Dummy data for demonstration if file not found
        print("\nCreating a dummy DataFrame for preprocessing demonstration (for testing this script)...")
        data = {
            'Date received': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12']),
            'Product': [
                'Credit card', 'Personal loan', 'Mortgage', 'Credit card', 'Student loan',
                'Buy Now, Pay Later (BNPL)', 'Savings account', 'Money transfers',
                'Credit reporting, credit repair services, or other personal consumer reports',
                'Credit card', 'Savings account', 'Personal loan', 'Bank account or service'
            ],
            'Sub-product': [
                'General purpose credit card', np.nan, 'Other mortgage', np.nan, 'Federal student loan',
                np.nan, np.nan, np.nan, np.nan, 'Store credit card', np.nan, np.nan, 'Checking account'
            ],
            'Consumer complaint narrative': [
                "My credit card interest rate was increased without proper notification. This is affecting my ability to pay off the balance.",
                "I was approved for a personal loan but the funds were never disbursed.",
                "I applied for a mortgage refinance and the process has been extremely slow and uncommunicative.",
                "This is a short credit card complaint about a billing error. I am writing to file a complaint.",
                "My student loan servicer has misapplied payments leading to higher interest charges.",
                "I am having trouble with a BNPL payment plan. The auto-debit failed for no reason and now I'm being charged late fees.",
                np.nan, # Will be filtered out by empty narrative
                "My international money transfer was delayed by over two weeks with no explanation.",
                "Credit report error.",
                "Unauthorized charges appeared on my credit card statement. The following is a narrative provided by the consumer: I am writing to express my concern about these charges.",
                "I noticed unauthorized activity on my savings account. To whom it may concern.",
                "I need help with my personal loan. I would like to file a complaint.",
                "I have an issue with my bank account."
            ],
            'Company': [
                'Bank of America', 'LendingClub', 'Wells Fargo', 'Capital One', 'Sallie Mae',
                'Affirm', 'Chase Bank', 'Western Union', 'Experian', 'Discover',
                'Ally Bank', 'Prosper', 'PNC Bank'
            ],
            'State': ['CA', 'NY', 'TX', 'FL', 'IL', 'GA', 'WA', 'AZ', 'CA', 'IL', 'OH', 'CO', 'PA'],
            'ZIP code': [
                '90210', '10001', '75001', '33101', '60601', '30301', '98101', '85001',
                '90001', '60602', '43201', '80202', '15201'
            ],
            'Tags': [np.nan, 'Servicemember', np.nan, np.nan, 'Older American', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'Consumer consent provided?': [
                'Consent provided', 'Consent provided', 'Consent provided', 'Consent provided', 'Consent provided',
                'Consent provided', 'Consent not provided', 'Consent provided', 'Consent provided', 'Consent provided',
                'Consent provided', 'Consent provided', 'Consent provided'
            ],
            'Submitted via': ['Web', 'Web', 'Web', 'Web', 'Web', 'Web', 'Web', 'Web', 'Web', 'Web', 'Web', 'Web', 'Web'],
            'Date sent to company': pd.to_datetime([
                '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09',
                '2023-01-10', '2023-01-11', '2023-01-12'
            ]),
            'Company response to consumer': [
                'Closed with explanation', 'Closed with explanation', 'In progress', 'Closed with explanation', 'Closed with explanation',
                'Closed with explanation', 'Closed with explanation', 'Closed with explanation', 'Closed with explanation', 'Closed with explanation',
                'Closed with explanation', 'Closed with explanation', 'Closed with explanation'
            ],
            'Timely response?': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
            'Consumer disputed?': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'],
            'Complaint ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        df = pd.DataFrame(data)
        print("Dummy DataFrame created for preprocessing demonstration.")
    except pd.errors.EmptyDataError:
        print(f"Error: '{input_csv_path}' is empty. No data to process.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading '{input_csv_path}': {e}")
        return

    if df is None or df.empty:
        print("No data loaded or DataFrame is empty. Exiting preprocessing.")
        return

    if 'Consumer complaint narrative' not in df.columns:
        print("\nError: 'Consumer complaint narrative' column not found in the dataset. Please check column names.")
        return

    print(f"\nInitial dataset shape for preprocessing: {df.shape}")

    # --- 2. Filter the dataset to meet the project's requirements ---
    print("\n--- Filtering Data ---")

    target_products = [
        'Credit card',
        'Personal loan',
        'Buy Now, Pay Later (BNPL)',
        'Savings account',
        'Money transfers'
    ]

    df_filtered = df[df['Product'].isin(target_products)].copy()
    print(f"Shape after filtering for target products: {df_filtered.shape}")

    df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
    print(f"Shape after removing records with empty narratives: {df_filtered.shape}")

    # --- 3. Clean the text narratives to improve embedding quality ---
    print("\n--- Cleaning Text Narratives ---")

    def clean_text(text):
        """
        Cleans the input text by performing:
        1. Lowercasing
        2. Removing common boilerplate text specific to CFPB narratives
        3. Removing special characters (keeping alphanumeric and basic whitespace)
        4. Normalizing whitespace (reducing multiple spaces to single, stripping leading/trailing)
        """
        text = str(text)

        text = text.lower()

        boilerplate_patterns = [
            r"consumer complaint narrative",
            r"the following is a narrative that was provided by the consumer to the cfpb",
            r"i am writing to file a complaint",
            r"i am writing to express my concern",
            r"this letter is to inform you",
            r"to whom it may concern",
            r"this complaint is regarding",
            r"i would like to report",
            r"i would like to file a complaint",
            r"i would like to complain",
            r"i am writing you this letter to inform you",
            r"i am writing this letter to you for the reason of"
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    df_filtered['Cleaned_Narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)

    initial_rows_after_filter = df_filtered.shape[0]
    df_filtered['Cleaned_Narrative_Length_Chars'] = df_filtered['Cleaned_Narrative'].apply(lambda x: len(x.strip()))
    df_cleaned_final = df_filtered[df_filtered['Cleaned_Narrative_Length_Chars'] > 0].copy()

    print(f"Dataset shape after text cleaning and removing narratives that became empty: {df_cleaned_final.shape}")
    print(f"Number of narratives removed due to becoming empty after cleaning: {initial_rows_after_filter - df_cleaned_final.shape[0]}")

    # Drop the temporary length column
    df_cleaned_final = df_cleaned_final.drop(columns=['Cleaned_Narrative_Length_Chars'])

    # --- IMPORTANT: Select and save ONLY the required columns for next step ---
    # This ensures embed_and_index.py gets exactly what it expects.
    columns_to_save = ['Complaint ID', 'Product', 'Cleaned_Narrative']
    # Filter df_cleaned_final to only these columns, ensuring they exist
    # Check for column existence before selecting to avoid KeyError if preprocess logic changes
    df_final_for_save = df_cleaned_final[[col for col in columns_to_save if col in df_cleaned_final.columns]].copy()

    # If any required column is missing here, print a warning
    if not all(col in df_final_for_save.columns for col in columns_to_save):
        missing_on_save = [col for col in columns_to_save if col not in df_final_for_save.columns]
        print(f"Warning: The following required columns are missing before saving: {missing_on_save}")
        print(f"Columns present in DataFrame before saving: {df_final_for_save.columns.tolist()}")

    # --- 4. Save the cleaned and filtered dataset ---
    df_final_for_save.to_csv(output_file_path, index=False)
    print(f"\nCleaned and filtered dataset saved to: {output_file_path}")

    print("\n--- Data Preprocessing Complete ---")

if __name__ == "__main__":
    run_preprocessing()