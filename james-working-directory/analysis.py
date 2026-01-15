import logging

import pandas as pd
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the CSV file
df = pd.read_csv("./output/Combined_Wide_Data.csv")

# Identify regions and their sub-types
regions = {}
region_subtypes = {}
all_subtypes = set()

for col in df.columns[1:]:  # Skip 'date' column
    parts = col.split("_")
    if len(parts) >= 2:
        region = parts[0]
        subtype = parts[1]

        if region not in regions:
            regions[region] = []
            region_subtypes[region] = set()
        regions[region].append(col)
        region_subtypes[region].add(subtype)
        all_subtypes.add(subtype)


# Function to analyze a region
def analyze_region(region_name, columns):
    if not columns:
        return None

    # Log the columns for UnknownRegion
    if region_name == "UnknownRegion":
        logging.info(f"UnknownRegion columns: {columns}")

    # Subset the dataframe
    sub_df = df[["date"] + columns].copy()

    # Convert date to datetime
    sub_df["date"] = pd.to_datetime(sub_df["date"])

    # Sum across all columns for the region (assuming additive metrics)
    sub_df["total"] = sub_df[columns].sum(axis=1)

    # Compute summary statistics
    stats = sub_df["total"].describe()

    return {
        "Region": region_name,
        "Num_Columns": len(columns),
        "SubTypes": ", ".join(sorted(region_subtypes[region_name])),
        "Mean": stats["mean"],
        "Std": stats["std"],
        "Min": stats["min"],
        "Max": stats["max"],
        "Count": stats["count"],
    }


# Analyze each region
results = []
for region, cols in regions.items():
    result = analyze_region(region, cols)
    if result:
        results.append(result)

# Create a dataframe from results
results_df = pd.DataFrame(results)

# Print tabulated results
print(tabulate(results_df, headers="keys", tablefmt="grid", floatfmt=".2f"))

# Analyze common sub-types
common_subtypes = set()
for subtype in all_subtypes:
    if all(subtype in region_subtypes[region] for region in region_subtypes):
        common_subtypes.add(subtype)

missing_summary = {}
for subtype in all_subtypes:
    missing_regions = [
        region for region in region_subtypes if subtype not in region_subtypes[region]
    ]
    if missing_regions:
        missing_summary[subtype] = missing_regions

# Summary with warnings
logging.info(
    f"Common sub-types across all regions: {', '.join(sorted(common_subtypes))}"
)
for subtype, missing in missing_summary.items():
    logging.warning(f"Sub-type '{subtype}' is missing in regions: {', '.join(missing)}")
