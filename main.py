import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from pandarallel import pandarallel
import json

# File paths for input and output
parquet_file = ''
output_directory = ''

# Utility functions for fuzzy matching and merging

def fuzzy_unique(values, threshold=0.9):
    """
    Return a list of unique values, using fuzzy string matching to deduplicate similar entries.
    """
    values = [str(v).strip() for v in values if pd.notnull(v) and str(v).strip()]
    unique = []
    for v in values:
        if not any(SequenceMatcher(None, v.lower(), u.lower()).ratio() > threshold for u in unique):
            unique.append(v)
    return unique

# Fuzzy merge function that combines similar values into a single string

def fuzzy_merge(values, threshold=0.9):
    """
    Merge a list of values into a single string, deduplicating similar values using fuzzy matching.
    """
    unique = fuzzy_unique(values, threshold)
    return ' | '.join(unique) if unique else None

# Helper functions for aggregating fields

def get_mode(series):
    """
    Return the most frequent (mode) value in a pandas Series, ignoring NaNs and stripping whitespace.
    """
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return None
    return s.mode().iloc[0]

# Helper functions to get the shortest and longest non-empty strings

def get_shortest(series):
    """
    Return the shortest non-empty string in a pandas Series.
    """
    s = series.dropna().astype(str)
    s = s[s.str.strip() != '']
    if s.empty:
        return None
    # Defensive: if all are empty after filtering, return None
    try:
        idx = s.str.len().idxmin()
        return s.loc[idx] if idx in s.index else None
    except (ValueError, KeyError):
        return None

# Helper function to get the longest non-empty string

def get_longest(series):
    """
    Return the longest non-empty string in a pandas Series.
    """
    s = series.dropna().astype(str)
    s = s[s.str.strip() != '']
    if s.empty:
        return None
    try:
        idx = s.str.len().idxmax()
        return s.loc[idx] if idx in s.index else None
    except (ValueError, KeyError):
        return None

# Function to split and merge values in a series, removing duplicates and sorting

def split_and_merge(series):
    """
    Split string values in a Series by common delimiters, deduplicate, and merge back into a single string.
    """
    import re
    vals = []
    for x in series.dropna().unique():
        if isinstance(x, str):
            vals.extend([v.strip() for v in re.split(r'[;,|]', x) if v.strip()])
        else:
            vals.append(str(x))
    return ' | '.join(sorted(set(vals))) if vals else None

# Function to normalize URLs by removing protocol and www, and ensuring lowercase

def normalize_url(url):
    """
    Normalize a URL by removing protocol, www, and trailing slashes, and converting to lowercase.
    """
    if not isinstance(url, str):
        return url
    url = url.lower().replace('https://', '').replace('http://', '')
    if url.startswith('www.'):
        url = url[4:]
    return url.rstrip('/')

# Function to parse the locations field into a list of tuples

def parse_locations_field(loc_str):
    """
    Parse a locations string into a list of 9-field tuples (country_code, country, region, city, postcode, street, street_number, lat, lon).
    Handles multiple locations separated by ' | '.
    """
    # Split by ' | ' for multiple locations
    if not isinstance(loc_str, str) or not loc_str.strip():
        return []
    locs = [l.strip() for l in loc_str.split(' | ') if l.strip()]
    parsed = []
    for l in locs:
        # Split by comma, preserve empty fields
        parts = [p.strip() if p.strip() else None for p in l.split(',')]
        # Pad to 9 fields if needed
        while len(parts) < 9:
            parts.append(None)
        parsed.append(tuple(parts[:9]))
    return parsed

# Function to merge locations with fuzzy matching and deduplication

def locations_fuzzy_merge(locations_list, threshold=0.9):
    """
    Deduplicate and merge a list of location tuples using strict and fuzzy logic:
    - Remove duplicates based on country, city, street, and lat/lon (within 2 decimals).
    - Then merge similar locations using fuzzy matching on city and street.
    """
    def is_duplicate(loc1, loc2):
        # Compare country
        if loc1[0] != loc2[0]:
            return False
        # Compare city (index 3) and street (index 5), allow nulls as matches
        def match_or_null(a, b):
            return (a == b) or (not a and not b) or (not a and b) or (a and not b)
        if not match_or_null(loc1[3], loc2[3]):
            return False
        if not match_or_null(loc1[5], loc2[5]):
            return False
        # Compare lat/lon within 2 decimal places if both present
        def round_or_none(x):
            try:
                return round(float(x), 2)
            except (TypeError, ValueError):
                return None
        lat1, lat2 = round_or_none(loc1[7]), round_or_none(loc2[7])
        lon1, lon2 = round_or_none(loc1[8]), round_or_none(loc2[8])
        if lat1 is not None and lat2 is not None and lat1 != lat2:
            return False
        if lon1 is not None and lon2 is not None and lon1 != lon2:
            return False
        return True
    # Remove duplicates first
    unique = []
    for loc in locations_list:
        if not any(is_duplicate(loc, u) for u in unique):
            unique.append(loc)
    # Now do fuzzy merge as before
    def is_similar(loc1, loc2):
        for i in range(9):
            a, b = loc1[i], loc2[i]
            if i in [3, 5]:  # city, street: fuzzy
                if a and b:
                    if SequenceMatcher(None, a.lower(), b.lower()).ratio() < threshold:
                        return False
                elif a or b:
                    return False
            else:
                if a != b:
                    return False
        return True
    merged = []
    for loc in unique:
        if not any(is_similar(loc, u) for u in merged):
            merged.append(loc)
    return merged

# Function to format merged locations back to the original string format

def format_locations_field(locations_tuples):
    """
    Convert a list of location tuples back to the string format for output.
    """
    loc_strs = []
    for loc in locations_tuples:
        loc_strs.append(', '.join([v if v is not None else '' for v in loc]))
    return ' | '.join(loc_strs)

def aggregate_group(group):
    """
    Aggregate a group of company records, deduplicating and merging fields using fuzzy matching and other heuristics.
    """
    # Main country
    main_country = get_mode(group['main_country'])
    main_country_code = get_mode(group['main_country_code'])
    # Main region/city/district/street/number
    main_region = get_mode(group.loc[group['main_country'] == main_country, 'main_region'])
    main_city = get_mode(group.loc[group['main_country'] == main_country, 'main_city'])
    main_city_district = get_mode(group.loc[(group['main_country'] == main_country) & (group['main_city'] == main_city), 'main_city_district'])
    main_street = get_mode(group.loc[(group['main_country'] == main_country) & (group['main_city'] == main_city), 'main_street'])
    main_street_number = get_mode(group.loc[(group['main_country'] == main_country) & (group['main_city'] == main_city) & (group['main_street'] == main_street), 'main_street_number'])
    # Main lat/lon
    main_latitude = group.loc[(group['main_country'] == main_country) & (group['main_city'] == main_city) & (group['main_street'] == main_street), 'main_latitude']
    main_longitude = group.loc[(group['main_country'] == main_country) & (group['main_city'] == main_city) & (group['main_street'] == main_street), 'main_longitude']
    main_latitude = main_latitude.iloc[0] if not main_latitude.empty else None
    main_longitude = main_longitude.iloc[0] if not main_longitude.empty else None
    # Main address raw text
    main_address_raw_text = ', '.join(group.loc[(group['main_country'] == main_country) & (group['main_city'] == main_city) & (group['main_street'] == main_street), 'main_address_raw_text'].dropna().unique())
    # Locations: merge using only the existing locations field
    all_locations = []
    for loc_str in group['locations'].dropna():
        all_locations.extend(parse_locations_field(loc_str))
    merged_locations = locations_fuzzy_merge(all_locations)
    locations = format_locations_field(merged_locations)
    num_locations = len(merged_locations)
    # Legal/commercial names: fuzzy unique, join with |
    company_legal_names = fuzzy_merge(group['company_legal_names'])
    company_commercial_names = fuzzy_merge(group['company_commercial_names'])
    # Company type: split and merge unique
    company_type = split_and_merge(group['company_type'])
    # Year founded: mode
    year_founded = get_mode(group['year_founded'])
    lnk_year_founded = get_mode(group['lnk_year_founded']) if 'lnk_year_founded' in group else None
    # Descriptions
    short_description = get_shortest(group['short_description'])
    long_description = get_longest(group['long_description'])
    # Business tags: split and merge unique
    business_tags = split_and_merge(group['business_tags'])
    generated_business_tags = split_and_merge(group['generated_business_tags'])
    # Business model, product type, naics_vertical, etc: split and merge unique
    business_model = split_and_merge(group['business_model'])
    product_type = split_and_merge(group['product_type'])
    naics_vertical = split_and_merge(group['naics_vertical'])
    naics_2022_primary_code = get_mode(group['naics_2022_primary_code'])
    naics_2022_primary_label = get_mode(group['naics_2022_primary_label'])
    naics_2022_secondary_codes = split_and_merge(group['naics_2022_secondary_codes'])
    naics_2022_secondary_labels = split_and_merge(group['naics_2022_secondary_labels'])
    main_business_category = split_and_merge(group['main_business_category'])
    main_industry = split_and_merge(group['main_industry'])
    main_sector = split_and_merge(group['main_sector'])
    # Phones/emails
    primary_phone = get_mode(group['primary_phone'])
    phone_numbers = split_and_merge(group['phone_numbers'])
    primary_email = get_mode(group['primary_email'])
    emails = split_and_merge(group['emails'])
    other_emails = split_and_merge(group['other_emails'])
    # URLs/domains: mode of normalized
    def norm_mode(col):
        if col in group:
            normed = group[col].dropna().map(normalize_url)
            return normed.mode().iloc[0] if not normed.empty else None
        return None
    website_url = norm_mode('website_url')
    website_domain = norm_mode('website_domain')
    website_tld = split_and_merge(group['website_tld'])
    website_language_code = split_and_merge(group['website_language_code'])
    facebook_url = norm_mode('facebook_url')
    twitter_url = norm_mode('twitter_url')
    instagram_url = norm_mode('instagram_url')
    linkedin_url = norm_mode('linkedin_url')
    ios_app_url = norm_mode('ios_app_url')
    android_app_url = norm_mode('android_app_url')
    youtube_url = norm_mode('youtube_url')
    tiktok_url = norm_mode('tiktok_url')
    # Domains: mode and all unique
    domains = split_and_merge(group['domains'])
    all_domains = split_and_merge(group['all_domains'])
    # Alexa rank: mode
    alexa_rank = get_mode(group['alexa_rank'])
    # SICS/ISIC/NACE: split and merge unique
    sics_codified_industry = split_and_merge(group['sics_codified_industry'])
    sics_codified_industry_code = split_and_merge(group['sics_codified_industry_code'])
    sics_codified_subsector = split_and_merge(group['sics_codified_subsector'])
    sics_codified_subsector_code = split_and_merge(group['sics_codified_subsector_code'])
    sics_codified_sector = split_and_merge(group['sics_codified_sector'])
    sics_codified_sector_code = split_and_merge(group['sics_codified_sector_code'])
    sic_codes = split_and_merge(group['sic_codes'])
    sic_labels = split_and_merge(group['sic_labels'])
    isic_v4_codes = split_and_merge(group['isic_v4_codes'])
    isic_v4_labels = split_and_merge(group['isic_v4_labels'])
    nace_rev2_codes = split_and_merge(group['nace_rev2_codes'])
    nace_rev2_labels = split_and_merge(group['nace_rev2_labels'])
    # Dates
    created_at = pd.to_datetime(group['created_at'], errors='coerce').min()
    last_updated_at = pd.to_datetime(group['last_updated_at'], errors='coerce').max()
    # Website number of pages: max
    website_number_of_pages = pd.to_numeric(group['website_number_of_pages'], errors='coerce').max() if 'website_number_of_pages' in group else None
    # Generated description: longest
    generated_description = get_longest(group['generated_description']) if 'generated_description' in group else None
    # Status, domains, revenue_type, employee_count_type: unique values joined by |
    def unique_join(col):
        vals = group[col].dropna().astype(str).str.strip().unique() if col in group else []
        return ' | '.join(sorted(set(vals))) if len(vals) else None
    status = unique_join('status')
    revenue_type = unique_join('revenue_type')
    employee_count_type = unique_join('employee_count_type')
    # Revenue, employee_count, inbound_links_count: sum of fields with unique address
    address_cols_sum = ['main_street', 'main_city', 'main_country_code', 'main_country', 'main_region', 'main_city_district', 'main_postcode', 'main_street_number']
    unique_addresses = group[address_cols_sum].drop_duplicates()
    revenue = group.loc[unique_addresses.index, 'revenue'].dropna().astype(float).sum() if 'revenue' in group else None
    employee_count = group.loc[unique_addresses.index, 'employee_count'].dropna().astype(float).sum() if 'employee_count' in group else None
    inbound_links_count = group.loc[unique_addresses.index, 'inbound_links_count'].dropna().astype(float).sum() if 'inbound_links_count' in group else None
    return pd.Series({
        'company_legal_names': company_legal_names,
        'company_commercial_names': company_commercial_names,
        'main_country_code': main_country_code,
        'main_country': main_country,
        'main_region': main_region,
        'main_city_district': main_city_district,
        'main_city': main_city,
        'main_postcode': get_mode(group['main_postcode']),
        'main_street': main_street,
        'main_street_number': main_street_number,
        'main_latitude': main_latitude,
        'main_longitude': main_longitude,
        'main_address_raw_text': main_address_raw_text,
        'locations': locations,
        'num_locations': num_locations,
        'company_type': company_type,
        'year_founded': year_founded,
        'lnk_year_founded': lnk_year_founded,
        'short_description': short_description,
        'long_description': long_description,
        'business_tags': business_tags,
        'business_model': business_model,
        'product_type': product_type,
        'naics_vertical': naics_vertical,
        'naics_2022_primary_code': naics_2022_primary_code,
        'naics_2022_primary_label': naics_2022_primary_label,
        'naics_2022_secondary_codes': naics_2022_secondary_codes,
        'naics_2022_secondary_labels': naics_2022_secondary_labels,
        'main_business_category': main_business_category,
        'main_industry': main_industry,
        'main_sector': main_sector,
        'primary_phone': primary_phone,
        'phone_numbers': phone_numbers,
        'primary_email': primary_email,
        'emails': emails,
        'other_emails': other_emails,
        'website_url': website_url,
        'website_domain': website_domain,
        'website_tld': website_tld,
        'website_language_code': website_language_code,
        'facebook_url': facebook_url,
        'twitter_url': twitter_url,
        'instagram_url': instagram_url,
        'linkedin_url': linkedin_url,
        'ios_app_url': ios_app_url,
        'android_app_url': android_app_url,
        'youtube_url': youtube_url,
        'tiktok_url': tiktok_url,
        'alexa_rank': alexa_rank,
        'sics_codified_industry': sics_codified_industry,
        'sics_codified_industry_code': sics_codified_industry_code,
        'sics_codified_subsector': sics_codified_subsector,
        'sics_codified_subsector_code': sics_codified_subsector_code,
        'sics_codified_sector': sics_codified_sector,
        'sics_codified_sector_code': sics_codified_sector_code,
        'sic_codes': sic_codes,
        'sic_labels': sic_labels,
        'isic_v4_codes': isic_v4_codes,
        'isic_v4_labels': isic_v4_labels,
        'nace_rev2_codes': nace_rev2_codes,
        'nace_rev2_labels': nace_rev2_labels,
        'created_at': created_at,
        'last_updated_at': last_updated_at,
        'website_number_of_pages': website_number_of_pages,
        'generated_description': generated_description,
        'generated_business_tags': generated_business_tags,
        'status': status,
        'domains': domains,
        'all_domains': all_domains,
        'revenue': revenue,
        'revenue_type': revenue_type,
        'employee_count': employee_count,
        'employee_count_type': employee_count_type,
        'inbound_links_count': inbound_links_count,
    })

df = pd.read_parquet(parquet_file, engine='fastparquet')

# Precompute normalized columns for vectorized deduplication
if 'website_url' in df:
    df['normalized_website_url'] = df['website_url'].map(normalize_url)
if 'website_domain' in df:
    df['normalized_website_domain'] = df['website_domain'].str.lower()
if 'website_language_code' in df:
    df['normalized_website_language_code'] = df['website_language_code'].str.lower()
url_fields = ['facebook_url', 'twitter_url', 'instagram_url', 'linkedin_url', 'ios_app_url', 'android_app_url', 'youtube_url', 'tiktok_url']
for field in url_fields:
    if field in df:
        df['normalized_' + field] = df[field].map(normalize_url)

# Initialize pandarallel for parallel processing
pandarallel.initialize(progress_bar=True)

# Split df into singles and dupes
counts = df.groupby('company_name').size()
single_names = counts[counts == 1].index
dup_names = counts[counts > 1].index
singles = df[df['company_name'].isin(single_names)].copy()
dupes = df[df['company_name'].isin(dup_names)].copy()

# Set num_locations for singles: 1 if locations is not null/empty, else 0
if 'locations' in singles:
    singles['num_locations'] = singles['locations'].apply(lambda x: 1 if pd.notnull(x) and str(x).strip() not in ('', '[]', 'null', 'None') else 0)

# For singles, deduplicate and merge locations
if 'locations' in singles:
    def merge_single_locations(row):
        locs = parse_locations_field(row['locations'])
        merged = locations_fuzzy_merge(locs)
        return format_locations_field(merged)
    singles['locations'] = singles['locations'].apply(lambda x: merge_single_locations({'locations': x}) if pd.notnull(x) and str(x).strip() not in ('', '[]', 'null', 'None') else x)
    singles['num_locations'] = singles['locations'].apply(lambda x: len(parse_locations_field(x)) if pd.notnull(x) and str(x).strip() not in ('', '[]', 'null', 'None') else 0)

# Deduplicate only the repeating company names
if not dupes.empty:
    deduped_dupes = dupes.groupby(['company_name'], dropna=False).parallel_apply(aggregate_group).reset_index()
    # For deduped_dupes, set num_locations to 0 if locations is null/empty after aggregation
    if 'locations' in deduped_dupes and 'num_locations' in deduped_dupes:
        mask = deduped_dupes['locations'].isnull() | (deduped_dupes['locations'].astype(str).str.strip().isin(['', '[]', 'null', 'None']))
        deduped_dupes.loc[mask, 'num_locations'] = 0
    result_df = pd.concat([singles, deduped_dupes], ignore_index=True, sort=False)
else:
    result_df = singles.copy()

# After aggregation, convert locations to JSON string for Parquet compatibility

def locations_to_json(df):
    """
    Convert locations and other fields to JSON-compatible format for Parquet output:
    - Locations: JSON string
    - num_locations: integer
    - created_at, last_updated_at: string
    - website_number_of_pages: int or float
    - All other columns: convert lists/dicts to JSON, others to str
    """
    if 'locations' in df:
        df['locations'] = df['locations'].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
    if 'num_locations' in df:
        df['num_locations'] = pd.to_numeric(df['num_locations'], errors='coerce').fillna(0).astype(int)
    # Ensure created_at and last_updated_at are strings for Parquet compatibility
    for col in ['created_at', 'last_updated_at']:
        if col in df:
            df[col] = df[col].astype(str)
    # Ensure website_number_of_pages is int or float (not numpy type/object)
    if 'website_number_of_pages' in df:
        df['website_number_of_pages'] = df['website_number_of_pages'].apply(lambda x: int(x) if pd.notnull(x) and not isinstance(x, str) and float(x).is_integer() else (float(x) if pd.notnull(x) and x != '' else None))
    # Ensure all columns are either str, int, float, or bool (no objects, lists, dicts, numpy types)
    for col in df.columns:
        if df[col].dtype == 'O':
            # Convert lists/dicts to JSON, others to str
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else (str(x) if pd.notnull(x) else None))
    return df

result_df = locations_to_json(result_df)

result_df.to_parquet(
    output_directory + 'deduped_output.snappy.parquet',
    engine='fastparquet',
    compression='snappy',
    index=False
)


