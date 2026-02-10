import duckdb
import os
from pathlib import Path

# Configuration
DB_FILE = "clinical_data.duckdb"
DATA_DIR = "clinical_data_output"
DOMAINS = ["DM", "AE", "VS", "CM", "LB", "TV"]

def create_database():
    """Create DuckDB database and load all clinical data domains."""
    
    # Connect to DuckDB (creates file if it doesn't exist)
    con = duckdb.connect(DB_FILE)
    
    print(f"Creating DuckDB database: {DB_FILE}")
    print("=" * 60)
    
    for domain in DOMAINS:
        domain_path = os.path.join(DATA_DIR, domain)
        
        if not os.path.exists(domain_path):
            print(f"⚠️  Skipping {domain}: Directory not found")
            continue
        
        # Read all parquet files for this domain and create table
        parquet_pattern = f"{domain_path}/**/*.parquet"
        
        print(f"\n📊 Loading {domain} domain...")
        
        # Create table from parquet files
        # DuckDB can directly read partitioned parquet files
        # Use hive_partitioning=1 to automatically parse partition columns
        con.execute(f"""
            CREATE OR REPLACE TABLE {domain} AS 
            SELECT * FROM read_parquet('{parquet_pattern}', hive_partitioning=1)
        """)
        
        # Get record count
        count = con.execute(f"SELECT COUNT(*) FROM {domain}").fetchone()[0]
        print(f"   ✅ Created table {domain} with {count:,} records")
        
        # Show sample of columns
        columns = con.execute(f"PRAGMA table_info({domain})").fetchall()
        col_names = [col[1] for col in columns]
        print(f"   📋 Columns ({len(col_names)}): {', '.join(col_names[:10])}")
        if len(col_names) > 10:
            print(f"      ... and {len(col_names) - 10} more")
    
    # Create summary view
    print("\n" + "=" * 60)
    print("📈 Database Summary:")
    print("=" * 60)
    
    for domain in DOMAINS:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {domain}").fetchone()[0]
            print(f"   {domain}: {count:,} records")
        except:
            pass
    
    # Get total database size
    total_records = con.execute(f"""
        SELECT 
            {' + '.join([f'(SELECT COUNT(*) FROM {d})' for d in DOMAINS if os.path.exists(os.path.join(DATA_DIR, d))])}
    """).fetchone()[0]
    
    print(f"\n   TOTAL: {total_records:,} records")
    print("=" * 60)
    
    # Show database file size
    if os.path.exists(DB_FILE):
        size_mb = os.path.getsize(DB_FILE) / (1024 * 1024)
        print(f"\n💾 Database file size: {size_mb:.2f} MB")
    
    print(f"\n✅ Database created successfully: {os.path.abspath(DB_FILE)}")
    
    # Close connection
    con.close()
    
    return DB_FILE

def query_examples(db_file):
    """Show some example queries."""
    
    print("\n" + "=" * 60)
    print("📝 Example Queries:")
    print("=" * 60)
    
    con = duckdb.connect(db_file, read_only=True)
    
    # Example 1: Count subjects by sex and arm
    print("\n1️⃣  Subjects by Sex and Treatment Arm:")
    result = con.execute("""
        SELECT ARM, SEX, COUNT(*) as subject_count
        FROM DM
        GROUP BY ARM, SEX
        ORDER BY ARM, SEX
    """).fetchall()
    for row in result:
        print(f"   {row[0]} - {row[1]}: {row[2]} subjects")
    
    # Example 2: Most common adverse events
    print("\n2️⃣  Top 5 Most Common Adverse Events:")
    result = con.execute("""
        SELECT AETERM, COUNT(*) as event_count
        FROM AE
        GROUP BY AETERM
        ORDER BY event_count DESC
        LIMIT 5
    """).fetchall()
    for i, row in enumerate(result, 1):
        print(f"   {i}. {row[0]}: {row[1]} events")
    
    # Example 3: Most common medications
    print("\n3️⃣  Top 5 Most Common Medications:")
    result = con.execute("""
        SELECT CMTRT, COUNT(*) as med_count
        FROM CM
        GROUP BY CMTRT
        ORDER BY med_count DESC
        LIMIT 5
    """).fetchall()
    for i, row in enumerate(result, 1):
        print(f"   {i}. {row[0]}: {row[1]} prescriptions")
    
    # Example 4: Lab tests with abnormal values
    print("\n4️⃣  Abnormal Lab Results (outside normal range):")
    result = con.execute("""
        SELECT 
            LBTEST,
            COUNT(*) as abnormal_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM LB WHERE LBTEST = l.LBTEST), 2) as pct_abnormal
        FROM LB l
        WHERE LBORRES < LBSTNRLO OR LBORRES > LBSTNRHI
        GROUP BY LBTEST
        ORDER BY abnormal_count DESC
        LIMIT 5
    """).fetchall()
    for row in result:
        print(f"   {row[0]}: {row[1]} abnormal ({row[2]}%)")
    
    con.close()
    
    print("\n" + "=" * 60)

def show_usage():
    """Show how to use the database."""
    
    print("\n" + "=" * 60)
    print("🔧 How to Use the Database:")
    print("=" * 60)
    
    print("""
# Python Example:
import duckdb

con = duckdb.connect('clinical_data.duckdb', read_only=True)

# Query any domain
df = con.execute("SELECT * FROM DM LIMIT 10").df()
print(df)

# Join domains
query = '''
    SELECT 
        dm.USUBJID,
        dm.AGE,
        dm.SEX,
        COUNT(ae.AESEQ) as adverse_event_count
    FROM DM dm
    LEFT JOIN AE ae ON dm.USUBJID = ae.USUBJID
    GROUP BY dm.USUBJID, dm.AGE, dm.SEX
    LIMIT 10
'''
result = con.execute(query).df()
print(result)

con.close()

# Command Line Example:
# duckdb clinical_data.duckdb "SELECT * FROM DM LIMIT 5"
""")

if __name__ == "__main__":
    # Create database and load data
    db_file = create_database()
    
    # Show example queries
    query_examples(db_file)
    
    # Show usage instructions
    show_usage()
