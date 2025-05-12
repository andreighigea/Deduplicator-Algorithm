import pandas as pd

parquet_file = ''
output_directory = ''

# I first concluded that company names have to be unique for each company,
# thus a redundant values would mean duplicated company data
# some tuples have values for some attributes, and some don't
# so I used a merge aproach where I grouped tuples by their "company_name" value
# and merged their other values accordingly

# First i read the parquet file using pandas library
df = pd.read_parquet(parquet_file, engine='fastparquet')

# This line groups values by company name, and sets other column data to the first non None value
filtered_df = df.groupby('company_name', as_index=False).agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None)

# Output the new file with snappy compression
filtered_df.to_parquet(
    output_directory + 'filtered_output.parquet',
    engine='fastparquet',
    compression='snappy',
    index=False
)
