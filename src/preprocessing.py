import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Text cleaning function
# -----------------------
def clean_text(text: str) -> str:
    """Lowercase, remove special chars, normalize spaces."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# -----------------------
# Product mapping function
# -----------------------
def map_to_target_product(product: str, narrative: str = "") -> str or None:
    """Map raw product & narrative to standardized product names."""
    product = str(product).lower()
    narrative = str(narrative).lower()

    if "credit card" in product or "prepaid card" in product:
        return "Credit card"
    elif any(term in product for term in [
        "personal loan", "consumer loan", "payday loan",
        "title loan", "advance loan", "vehicle loan"
    ]):
        if "buy now pay later" in narrative or "bnpl" in narrative:
            return "Buy Now, Pay Later"
        return "Personal loan"
    elif "checking or savings" in product or "bank account" in product or "savings" in product:
        return "Savings account"
    elif "money transfer" in product or "virtual currency" in product or "money service" in product:
        return "Money transfer"
    return None

# -----------------------
# Main preprocessing function
# -----------------------
def run_preprocessing(
    input_path="../data/raw/complaints.csv",
    output_path="../data/filtered/filtered_complaints.csv",
    plot_dir="plots",
    chunksize=100_000
) -> pd.DataFrame:
    print("📦 Starting chunked preprocessing...")

    filtered_chunks = []
    total_raw, total_kept = 0, 0
    total_with_narrative, total_without_narrative = 0, 0

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize, low_memory=False)):
        print(f"🔹 Processing chunk {i+1}...")

        chunk.columns = chunk.columns.str.strip()
        total_raw += len(chunk)

        # Keep needed columns and rename
        chunk = chunk[['Complaint ID', 'Product', 'Consumer complaint narrative']].copy()
        chunk.columns = ['complaint_id', 'product', 'narrative']

        total_with_narrative += chunk['narrative'].notna().sum()
        total_without_narrative += chunk['narrative'].isna().sum()

        # Filter rows with narrative
        chunk = chunk.dropna(subset=['narrative'])
        chunk['narrative'] = chunk['narrative'].astype(str)

        # Map product
        chunk['product'] = chunk.apply(
            lambda row: map_to_target_product(row['product'], row['narrative']),
            axis=1
        )
        chunk = chunk[chunk['product'].notna()]

        # Clean text & compute word count
        chunk['cleaned_narrative'] = chunk['narrative'].apply(clean_text)
        chunk['narrative_len'] = chunk['cleaned_narrative'].apply(lambda x: len(x.split()))

        # Filter by length
        chunk = chunk[
            (chunk['narrative'].str.strip().str.len() > 30) &
            (chunk['narrative_len'] > 10) &
            (chunk['narrative_len'] < 1000)
        ]

        total_kept += len(chunk)
        filtered_chunks.append(
            chunk[['complaint_id', 'product', 'cleaned_narrative', 'narrative_len']]
        )

    # Combine chunks
    df_final = pd.concat(filtered_chunks, ignore_index=True)

    # ✅ EDA summary
    print("\n✅ EDA Summary")
    print(f"Total complaints processed: {total_raw:,}")
    print(f"Complaints with narratives: {total_with_narrative:,}")
    print(f"Complaints without narratives: {total_without_narrative:,}")
    print(f"Complaints retained after filtering: {total_kept:,}")
    print("\nProduct distribution (filtered):")
    print(df_final['product'].value_counts())

    # Create dirs & save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ Saved filtered dataset to: {output_path}")

    # -----------------------
    # Plots
    # -----------------------
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: Narrative length distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df_final['narrative_len'], bins=50, kde=True)
    plt.title("Distribution of Narrative Length (Words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.tight_layout()
    plot1_path = os.path.join(plot_dir, "narrative_length_distribution.png")
    plt.savefig(plot1_path)
    print(f"📊 Saved: {plot1_path}")

    # Plot 2: Product distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df_final['product'], order=df_final['product'].value_counts().index, palette="Set2")
    plt.title("Complaint Count per Product")
    plt.xlabel("Count")
    plt.ylabel("Product")
    plt.tight_layout()
    plot2_path = os.path.join(plot_dir, "product_distribution.png")
    plt.savefig(plot2_path)
    print(f"📊 Saved: {plot2_path}")

    return df_final

# -----------------------
# Run when script executed directly
# -----------------------
if __name__ == "__main__":
    run_preprocessing()
