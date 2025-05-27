README: Company Data Deduplication and Aggregation Script
========================================================

Overview
--------
This script deduplicates and aggregates company data from a Parquet file, merging rows by `company_name` and intelligently combining all other fields using custom logic. It is optimized for large datasets and ensures compatibility with Parquet serialization.

Key Features
------------
- **Deduplication by Company Name:**
  - Splits the dataset into unique and duplicate company names.
  - Aggregates only the duplicates using custom rules, then recombines with unique entries.
- **Custom Aggregation Logic:**
  - Uses fuzzy matching, mode, sum, min/max, and other user-specified rules for each field.
  - Handles lists, URLs, tags, and descriptive fields with appropriate merging strategies.
- **Advanced Address/Location Merging:**
  - Reads the existing `locations` field (format: `{country code}, {country}, {region}, {city}, {postcode}, {street}, {street number}, {latitude}, {longitude}`; multiple locations separated by `|`).
  - Deduplicates locations by comparing country, city, street, and latitude/longitude (within 2 decimals), allowing for nulls and fuzzy matching on city/street.
  - Outputs merged locations in the same format, with empty values for nulls.
- **Parallel Processing:**
  - Uses `pandarallel` for fast groupby aggregation on large datasets.
- **Parquet Compatibility:**
  - Ensures all columns are serializable (converts lists/dicts to JSON, dates to string, etc.).

Usage
-----
1. **Install Requirements:**
   - Python 3.7+
   - pandas, numpy, fastparquet, pandarallel
   - (Optional) tqdm for progress bars
   - Install with: `pip install pandas numpy fastparquet pandarallel tqdm`

2. **Prepare Input:**
   - Place your input Parquet file at the path specified by `parquet_file` in the script.
   - Ensure the file contains a `company_name` column and a `locations` field in the expected format.

3. **Run the Script:**
   - Execute the script: `python main.py`
   - The deduplicated and aggregated output will be saved as `deduped_output.snappy.parquet` in the specified output directory.

4. **Output:**
   - The output Parquet file contains one row per unique company, with all fields merged according to the logic described above.
   - The `locations` field is a deduplicated, merged string; `num_locations` reflects the number of unique locations per company.

Customization
-------------
- You can adjust the fuzzy matching threshold and minimum length in the helper functions for more or less strict deduplication.
- The aggregation logic for each field can be customized in the `aggregate_group` function.

File Structure
--------------
- `main.py` : Main script with all logic, helper functions, and aggregation rules.
- Input Parquet file: Must contain company data with a `company_name` and `locations` field.
- Output Parquet file: Deduplicated, aggregated company data.
