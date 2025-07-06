# 📝 Complaint Data Cleaning & EDA (Modular Script)

# 📦 Imports & configuration
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn')

# ----------------------------
# 🧹 Text cleaning module
# ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    boilerplate = [
        r'(i\s?(am|have)\s?(writing|filing|submitting).*complaint)',
        r'(this\s?(is|concerns).*complaint)',
        r'(dear\s.*(sir|madam|representative))',
        r'(company\s?(name)?\s?:.*\w+)',
        r'(account\s?(number|#)?\s?:.*[\w-]+)',
        r'(date\s?(of)?\s?(incident|complaint).*[\w/]+)',
        r'(reference\s?(number|id).*\w+)',
        r'(phone\s?(number)?.*[\d-]+)',
        r'xx+',
        r'(\bsincerely\b|\bregards\b)'
    ]
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?$%]', '', text)
    for pattern in boilerplate:
        text = re.sub(pattern, '', text)
    text = re.sub(r'\b([a-z]{2,})\b', lambda m: m.group(1).lower(), text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# 🎯 Product filtering module
# ----------------------------
TARGET_PRODUCTS = {
    'Credit card': ['credit card', 'prepaid card'],
    'Personal loan': ['personal loan', 'consumer loan'],
    'Buy Now, Pay Later': ['buy now pay later', 'bnpl'],
    'Savings account': ['savings account', 'bank account'],
    'Money transfer': ['money transfer', 'wire transfer']
}

def filter_products(product: str) -> str or None:
    product = str(product).lower()
    for standardized, variants in TARGET_PRODUCTS.items():
        if any(variant in product for variant in variants):
            return standardized
    return None

# ----------------------------
# 🛠️ Main processing & EDA module
# ----------------------------
def process_complaints(
    input_path="../data/raw/complaints.csv",
    output_path="../data/processed/filtered_complaints.csv",
    plot_dir="../plots"
):
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    print("📊 Loading data...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Total complaints: {len(df):,}")

    print("\nProduct distribution (raw):")
    print(df['Product'].value_counts().head(10))
    has_narrative = df['Consumer complaint narrative'].notna()
    print(f"\nComplaints with narratives: {has_narrative.sum():,} ({has_narrative.mean():.1%})")
    print(f"Complaints without narratives: {(~has_narrative).sum():,}")

    df['product'] = df['Product'].apply(filter_products)
    df = df[df['product'].notna()]
    df = df[df['Consumer complaint narrative'].notna()]
    df['narrative'] = df['Consumer complaint narrative'].astype(str)
    df['cleaned_narrative'] = df['narrative'].apply(clean_text)
    df['word_count'] = df['cleaned_narrative'].apply(lambda x: len(x.split()))
    df = df[(df['word_count'] > 10) & (df['word_count'] < 1000)]

    print(f"\n✅ Complaints retained: {len(df):,}")
    print("\nProduct distribution (filtered):")
    print(df['product'].value_counts())

    plt.figure(figsize=(10, 5))
    sns.histplot(df['word_count'], bins=50, kde=True)
    plt.title("Distribution of Narrative Length (Words)")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/narrative_lengths.png")

    plt.figure(figsize=(10, 5))
    df['product'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Complaints by Product Category")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/product_distribution.png")

    plt.figure(figsize=(10, 5))
    df.groupby('product')['word_count'].mean().sort_values().plot(kind='barh', color='lightgreen')
    plt.title("Average Narrative Length by Product")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/mean_length_by_product.png")

    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved cleaned data to: {output_path}")
    return df

# ----------------------------
# ▶️ Run & preview
# ----------------------------
if __name__ == "__main__":
    df_clean = process_complaints()
    print(df_clean.head())