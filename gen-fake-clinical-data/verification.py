import polars as pl

# Read each domain
ae = pl.read_parquet('clinical_data_output/AE/**/*.parquet')
dm = pl.read_parquet('clinical_data_output/DM/**/*.parquet')
vs = pl.read_parquet('clinical_data_output/VS/**/*.parquet')
cm = pl.read_parquet('clinical_data_output/CM/**/*.parquet')
lb = pl.read_parquet('clinical_data_output/LB/**/*.parquet')
tv = pl.read_parquet('clinical_data_output/TV/**/*.parquet')

print('=== AE (Adverse Events) ===')
print(f'Columns: {ae.columns}')
print(f'Record count: {len(ae)}')
print('Sample records:')
print(ae.head(3))

print('\n=== DM (Demographics) ===')
print(f'Columns: {dm.columns}')
print(f'Record count: {len(dm)}')
print('Sample records:')
print(dm.head(3))

print('\n=== VS (Vital Signs) ===')
print(f'Columns: {vs.columns}')
print(f'Record count: {len(vs)}')
print('Sample records:')
print(vs.head(3))

print('=== CM (Concomitant Medications) ===')
print(f'Columns: {cm.columns}')
print(f'Record count: {len(cm)}')
print('Sample records:')
print(cm.head(3))

print('\n=== LB (Laboratory) ===')
print(f'Columns: {lb.columns}')
print(f'Record count: {len(lb)}')
print('Sample records:')
print(lb.head(3))

print('\n=== TV (Trial Visits) ===')
print(f'Columns: {tv.columns}')
print(f'Record count: {len(tv)}')
print('Sample records:')
print(tv.head())