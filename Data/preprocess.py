import pandas as pd
import re

def clean_km_values(df):
    """
    Remove 'Km' unit from the 'Km' column and convert to numeric values
    """
    print("Cleaning kilometer values...")
    
    # Make a copy of the column for reporting
    if 'Km' in df.columns:
        # Check the first few values with Km to show examples of changes
        sample_values = df['Km'].dropna().head(3).tolist()
        
        # Remove 'Km' and convert to numeric
        df['Km'] = df['Km'].astype(str).str.replace(' Km', '', regex=False)
        
        # Remove commas in numbers and convert to float
        df['Km'] = df['Km'].str.replace(',', '', regex=False)
        df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
        
        print(f"Kilometer values cleaned. Example conversions:")
        for val in sample_values:
            cleaned = val.replace(' Km', '').replace(',', '')
            print(f"  '{val}' → {cleaned}")
    else:
        print("Warning: 'Km' column not found")
    
    return df

def clean_seats_values(df):
    """
    Clean 'Seat' column by removing the word 'chỗ' from each value
    """
    print("Cleaning seat values...")
    
    if 'Seat' in df.columns:
        # Store original values for reporting
        sample_values = df['Seat'].dropna().head(5).tolist()
        
        # Remove the word 'chỗ' from each value
        df['Seat'] = df['Seat'].astype(str).str.replace(' chỗ', '', regex=False)
        
        # Try to convert to numeric
        df['Seat'] = pd.to_numeric(df['Seat'], errors='coerce')
        
        # Report examples of the conversion
        print("Seat values cleaned. Example conversions:")
        for val in sample_values:
            if isinstance(val, str):  # Check if value is a string before replacing
                cleaned = val.replace(' chỗ', '')
                print(f"  '{val}' → '{cleaned}'")
    else:
        print("Warning: 'Seat' column not found")
    
    return df

def clean_doors_values(df):
    """
    Clean 'Window' column by removing the word 'cửa' from each value
    """
    print("Cleaning door values...")
    
    if 'Window' in df.columns:
        # Store original values for reporting
        sample_values = df['Window'].dropna().head(5).tolist()
        
        # Remove the word 'cửa' from each value
        df['Window'] = df['Window'].astype(str).str.replace(' cửa', '', regex=False)
        
        # Try to convert to numeric
        df['Window'] = pd.to_numeric(df['Window'], errors='coerce')
        
        # Report examples of the conversion
        print("Door values cleaned. Example conversions:")
        for val in sample_values:
            if isinstance(val, str):  # Check if value is a string before replacing
                cleaned = val.replace(' cửa', '')
                print(f"  '{val}' → '{cleaned}'")
    else:
        print("Warning: 'Window' column not found")
    
    return df

def clean_window_values(df):
    """
    Clean a potential 'Window' column by removing descriptive text and keeping numeric values
    """
    print("Checking for Window column...")
    
    if 'Window' in df.columns:
        print("Cleaning window values...")
        # Store original values for reporting
        sample_values = df['Window'].dropna().head(5).tolist()
        
        # Extract numeric values using regex
        df['Window'] = df['Window'].astype(str).apply(lambda x: re.search(r'(\d+)', x).group(1) if re.search(r'(\d+)', x) else x)
        
        # Try to convert to numeric
        df['Window'] = pd.to_numeric(df['Window'], errors='coerce')
        
        # Report examples of the conversion
        print("Window values cleaned. Example conversions:")
        for val in sample_values:
            if isinstance(val, str):
                cleaned = re.search(r'(\d+)', val)
                cleaned = cleaned.group(1) if cleaned else val
                print(f"  '{val}' → '{cleaned}'")
    else:
        print("Note: 'Window' column not found in the dataset")
    
    return df


############################
# def convert_condition_values(df):
    """
    Convert 'Condition' values from 'Xe đã dùng' to 0 and 'Xe mới' to 1
    """
    print("Converting car condition values...")
    
    if 'Condition' in df.columns:
        # Store original values for reporting
        condition_counts = df['Condition'].value_counts().to_dict()
        sample_values = df['Condition'].dropna().unique().tolist()
        
        # Create a mapping dictionary
        condition_map = {
            'Xe đã dùng': 0,
            'Xe mới': 1
        }
        
        # Apply the mapping to convert values
        df['Condition'] = df['Condition'].map(condition_map)
        
        # Report the conversion
        print("Condition values converted:")
        for condition, value in condition_map.items():
            count = condition_counts.get(condition, 0)
            print(f"  '{condition}' → {value}: {count} records")
    else:
        print("Warning: 'Condition' column not found")
    
    return df





# def convert_origin_values(df):
#     """
#     Convert 'Origin' values from 'Lắp ráp trong nước' to 0 and 'Nhập khẩu' to 1
#     """
#     print("Converting car origin values...")
    
#     if 'Origin' in df.columns:
#         # Store original values for reporting
#         origin_counts = df['Origin'].value_counts().to_dict()
        
#         # Create a mapping dictionary
#         origin_map = {
#             'Lắp ráp trong nước': 0,  # Domestically assembled
#             'Nhập khẩu': 1            # Imported
#         }
        
#         # Apply the mapping to convert values
#         df['Origin'] = df['Origin'].map(origin_map)
        
#         # Report the conversion
#         print("Origin values converted:")
#         for origin, value in origin_map.items():
#             count = origin_counts.get(origin, 0)
#             print(f"  '{origin}' → {value}: {count} records")
#     else:
#         print("Warning: 'Origin' column not found")
    
#     return df
############################










def convert_price_to_millions(df):
    """
    Convert Price values from mixed 'Tỷ' (billion) and 'Triệu' (million) format to numeric values
    with standardized 'Triệu' (million) as the unit
    """
    print("Converting price values to millions (Triệu)...")
    
    if 'Price' in df.columns:
        # Store original values for reporting
        sample_values = df['Price'].dropna().head(5).tolist()
        
        # Function to extract numeric values and convert to millions
        def extract_price_in_millions(price_str):
            if pd.isna(price_str):
                return None
                
            # Initialize values
            billions = 0
            millions = 0
            
            # Extract billions (Tỷ)
            ty_match = re.search(r'(\d+)\s*Tỷ', price_str)
            if ty_match:
                billions = int(ty_match.group(1))
            
            # Extract millions (Triệu)
            trieu_match = re.search(r'(\d+)\s*Triệu', price_str)
            if trieu_match:
                millions = int(trieu_match.group(1))
            
            # Convert to millions and return
            return billions * 1000 + millions
        
        # Apply the conversion function
        df['Price'] = df['Price'].apply(extract_price_in_millions)
        
        # Report examples of the conversion
        print("Price values converted to millions. Example conversions:")
        for val in sample_values:
            converted = extract_price_in_millions(val)
            print(f"  '{val}' → {converted} (Triệu)")
    else:
        print("Warning: 'Price' column not found")
    
    return df

def filter_by_location():
    # Read the CSV file
    print("Reading the CSV file...")
    df = pd.read_csv('rawdata.csv')
    
    # Get total number of records before filtering
    total_records = len(df)
    print(f"Total records before filtering: {total_records}")
    
    # Filter records to keep only those from Hà Nội and TP HCM
    df_filtered = df[df['Location'].isin(['Hà Nội', 'TP HCM'])]
    
    # Get number of records after filtering
    filtered_records = len(df_filtered)
    print(f"Records after filtering (Hà Nội and TP HCM only): {filtered_records}")
    print(f"Removed records: {total_records - filtered_records}")
    
    # Clean kilometer values by removing "Km" unit
    df_filtered = clean_km_values(df_filtered)
    
    
    # Clean seat values by removing "chỗ"
    df_filtered = clean_seats_values(df_filtered)
    
    # Clean door values by removing "cửa"
    df_filtered = clean_doors_values(df_filtered)
    
    # Convert price values to millions (Triệu)
    df_filtered = convert_price_to_millions(df_filtered)
    
    # Check and clean window values if column exists
    df_filtered = clean_window_values(df_filtered)
    
    # Convert condition values (Xe đã dùng -> 0, Xe mới -> 1)
    # df_filtered = convert_condition_values(df_filtered)
    
    # Convert origin values (Lắp ráp trong nước -> 0, Nhập khẩu -> 1)
    # df_filtered = convert_origin_values(df_filtered)
    
    # Remove the 'Condition' column if it exists
    if 'Condition' in df_filtered.columns:
        print(f"Removing 'Condition' column")
        df_filtered = df_filtered.drop('Condition', axis=1)
    
    # Remove the 'Car_code' column if it exists
    if 'Car_code' in df_filtered.columns:
        print(f"Removing 'Car_code' column")
        df_filtered = df_filtered.drop('Car_code', axis=1)
    
    # Save the filtered and cleaned data to a new CSV file
    output_file = 'preprocessed-data.csv'
    df_filtered.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")
    
    # Display count by location
    location_counts = df_filtered['Location'].value_counts()
    print("\nCount by location:")
    for location, count in location_counts.items():
        print(f"{location}: {count} records")

if __name__ == "__main__":
    filter_by_location()