import pandas as pd
import numpy as np


"""
First stage transormation from old format file to new
"""


def transorm_sales(sales_value):
    try:
        return float(str(sales_value).strip().replace(',', '.'))
    except ValueError:
        return 0


if __name__ == '__main__':
    df = pd.read_csv('stores_with_addresses.csv')
    month_columns = [(c, pd.datetime.strptime(c, ' %Y / %B ')) for c in df.columns[3:]]

    new_rows = []
    store_format = None
    address = None
    skipped = 0

    for index, row in df.iterrows():
        store_format = row['Store Format'] if not pd.isnull(row['Store Format']) else store_format
        if not store_format:
            raise ValueError('No store format!')
        address = row['Address'] if not pd.isnull(row['Address']) else address
        if not address:
            raise ValueError('No address!')
        if 'Total' in address:
            skipped += 1
            continue
        for col_name, col_date in month_columns:
            row_dict = dict(
                brand=row['Brand'], 
                store_format=store_format, 
                address=address,
                month=col_date.month,
                year=col_date.year,
                sales=row[col_name],
            )
            new_rows.append(row_dict)
    new_df = pd.DataFrame(new_rows, columns=['year', 'month', 'store_format', 'address', 'brand', 'sales'])  
    assert (len(df) - skipped) * 24 == (len(new_df)), True

    clean_new_df['sales'] = clean_new_df['sales'].apply(transorm_sales).astype(np.float32)
    clean_new_df.to_csv('sales_transformed.csv')
