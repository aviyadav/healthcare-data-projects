# DuckDB Quick Reference Guide

## Database Overview

**File**: `clinical_data.duckdb`  
**Size**: ~7.5 MB  
**Total Records**: 1,138,247  

### Tables

| Table | Records | Description |
|-------|---------|-------------|
| DM | 5,000 | Demographics |
| AE | 37,728 | Adverse Events |
| VS | 676,710 | Vital Signs |
| CM | 12,575 | Concomitant Medications |
| LB | 406,026 | Laboratory Tests |
| TV | 208 | Trial Visits |

## Quick Start

### Load the Database

```bash
# From command line
source .venv/bin/activate
python load_to_duckdb.py
```

### Query from Python

```python
import duckdb

# Connect to database
con = duckdb.connect('clinical_data.duckdb', read_only=True)

# Simple query
df = con.execute("SELECT * FROM DM LIMIT 10").df()
print(df)

# Close connection
con.close()
```

### Query from Command Line

```bash
# Activate virtual environment
source .venv/bin/activate

# Run a query
duckdb clinical_data.duckdb "SELECT * FROM DM LIMIT 5"

# Interactive mode
duckdb clinical_data.duckdb
```

## Common Queries

### 1. Subject Demographics

```sql
-- Count subjects by treatment arm and sex
SELECT ARM, SEX, COUNT(*) as count
FROM DM
GROUP BY ARM, SEX
ORDER BY ARM, SEX;

-- Age distribution by treatment arm
SELECT 
    ARM,
    MIN(AGE) as min_age,
    AVG(AGE) as avg_age,
    MAX(AGE) as max_age
FROM DM
GROUP BY ARM;
```

### 2. Adverse Events Analysis

```sql
-- Most common adverse events
SELECT AETERM, COUNT(*) as event_count
FROM AE
GROUP BY AETERM
ORDER BY event_count DESC
LIMIT 10;

-- Severe adverse events by body system
SELECT AEBODSYS, COUNT(*) as severe_count
FROM AE
WHERE AESEV IN ('SEVERE', 'SEVERE AND LIFE THREATENING')
GROUP BY AEBODSYS
ORDER BY severe_count DESC;

-- AE rate by treatment arm
SELECT 
    dm.ARM,
    COUNT(DISTINCT dm.USUBJID) as total_subjects,
    COUNT(ae.AESEQ) as total_events,
    ROUND(COUNT(ae.AESEQ) * 1.0 / COUNT(DISTINCT dm.USUBJID), 2) as events_per_subject
FROM DM dm
LEFT JOIN AE ae ON dm.USUBJID = ae.USUBJID
GROUP BY dm.ARM;
```

### 3. Vital Signs Analysis

```sql
-- Average vital signs by visit
SELECT 
    VISIT,
    VSTESTCD,
    ROUND(AVG(VSORRES), 2) as avg_value,
    VSORRESU as unit
FROM VS
WHERE VSTESTCD IN ('SYSBP', 'DIABP', 'HR', 'TEMP')
GROUP BY VISIT, VSTESTCD, VSORRESU
ORDER BY VISIT, VSTESTCD;

-- Blood pressure trends
SELECT 
    VISIT,
    ROUND(AVG(CASE WHEN VSTESTCD = 'SYSBP' THEN VSORRES END), 1) as avg_systolic,
    ROUND(AVG(CASE WHEN VSTESTCD = 'DIABP' THEN VSORRES END), 1) as avg_diastolic
FROM VS
WHERE VSTESTCD IN ('SYSBP', 'DIABP')
GROUP BY VISIT
ORDER BY VISIT;
```

### 4. Concomitant Medications

```sql
-- Most prescribed medications
SELECT CMTRT, CMCAT, COUNT(*) as prescription_count
FROM CM
GROUP BY CMTRT, CMCAT
ORDER BY prescription_count DESC
LIMIT 10;

-- Medications by route of administration
SELECT CMROUTE, COUNT(*) as count
FROM CM
GROUP BY CMROUTE
ORDER BY count DESC;

-- Average number of medications per subject
SELECT 
    ROUND(AVG(med_count), 2) as avg_medications_per_subject
FROM (
    SELECT USUBJID, COUNT(*) as med_count
    FROM CM
    GROUP BY USUBJID
);
```

### 5. Laboratory Analysis

```sql
-- Abnormal lab results
SELECT 
    LBTEST,
    COUNT(*) as total_tests,
    SUM(CASE WHEN LBORRES < LBSTNRLO OR LBORRES > LBSTNRHI THEN 1 ELSE 0 END) as abnormal_count,
    ROUND(SUM(CASE WHEN LBORRES < LBSTNRLO OR LBORRES > LBSTNRHI THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_abnormal
FROM LB
GROUP BY LBTEST
ORDER BY abnormal_count DESC;

-- Lab trends over visits
SELECT 
    VISIT,
    LBTEST,
    ROUND(AVG(LBORRES), 2) as avg_value,
    LBORRESU as unit
FROM LB
WHERE LBTESTCD IN ('WBC', 'HGB', 'PLT', 'GLUC')
GROUP BY VISIT, LBTEST, LBORRESU
ORDER BY LBTEST, VISIT;

-- Subjects with critical lab values
SELECT 
    USUBJID,
    LBTEST,
    LBORRES,
    LBSTNRLO,
    LBSTNRHI,
    VISIT
FROM LB
WHERE LBORRES < LBSTNRLO * 0.5 OR LBORRES > LBSTNRHI * 1.5
ORDER BY USUBJID, VISIT;
```

### 6. Cross-Domain Analysis

```sql
-- Subject profile with AE and medication counts
SELECT 
    dm.USUBJID,
    dm.AGE,
    dm.SEX,
    dm.ARM,
    COUNT(DISTINCT ae.AESEQ) as ae_count,
    COUNT(DISTINCT cm.CMSEQ) as medication_count
FROM DM dm
LEFT JOIN AE ae ON dm.USUBJID = ae.USUBJID
LEFT JOIN CM cm ON dm.USUBJID = cm.USUBJID
GROUP BY dm.USUBJID, dm.AGE, dm.SEX, dm.ARM
LIMIT 20;

-- Correlation between age and adverse events
SELECT 
    CASE 
        WHEN dm.AGE < 30 THEN '18-29'
        WHEN dm.AGE < 50 THEN '30-49'
        WHEN dm.AGE < 70 THEN '50-69'
        ELSE '70+'
    END as age_group,
    COUNT(DISTINCT dm.USUBJID) as subjects,
    COUNT(ae.AESEQ) as total_aes,
    ROUND(COUNT(ae.AESEQ) * 1.0 / COUNT(DISTINCT dm.USUBJID), 2) as aes_per_subject
FROM DM dm
LEFT JOIN AE ae ON dm.USUBJID = ae.USUBJID
GROUP BY age_group
ORDER BY age_group;

-- Lab abnormalities by treatment arm
SELECT 
    dm.ARM,
    lb.LBTEST,
    SUM(CASE WHEN lb.LBORRES < lb.LBSTNRLO OR lb.LBORRES > lb.LBSTNRHI THEN 1 ELSE 0 END) as abnormal_count,
    COUNT(*) as total_tests,
    ROUND(SUM(CASE WHEN lb.LBORRES < lb.LBSTNRLO OR lb.LBORRES > lb.LBSTNRHI THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_abnormal
FROM DM dm
JOIN LB lb ON dm.USUBJID = lb.USUBJID
GROUP BY dm.ARM, lb.LBTEST
HAVING COUNT(*) > 100
ORDER BY dm.ARM, pct_abnormal DESC;
```

### 7. Trial Visit Schedule

```sql
-- View planned visit schedule
SELECT 
    VISITNUM,
    VISIT,
    TVSTRL as window_start,
    TVENRL as window_end,
    TVENRL - TVSTRL as window_days
FROM TV
WHERE STUDYID = 'STUDY-001'
ORDER BY VISITNUM;
```

## Export Data

### Export to CSV

```python
import duckdb

con = duckdb.connect('clinical_data.duckdb', read_only=True)

# Export a table
con.execute("COPY DM TO 'dm_export.csv' (HEADER, DELIMITER ',')")

# Export query results
con.execute("""
    COPY (
        SELECT * FROM AE WHERE AESEV = 'SEVERE'
    ) TO 'severe_aes.csv' (HEADER, DELIMITER ',')
""")

con.close()
```

### Export to Parquet

```python
import duckdb

con = duckdb.connect('clinical_data.duckdb', read_only=True)

# Export to parquet
con.execute("COPY DM TO 'dm_export.parquet' (FORMAT PARQUET)")

con.close()
```

## Performance Tips

1. **Use indexes for frequent queries**:
```sql
CREATE INDEX idx_dm_usubjid ON DM(USUBJID);
CREATE INDEX idx_ae_usubjid ON AE(USUBJID);
```

2. **Use EXPLAIN to analyze queries**:
```sql
EXPLAIN SELECT * FROM DM WHERE AGE > 65;
```

3. **Use read_only mode when not modifying data**:
```python
con = duckdb.connect('clinical_data.duckdb', read_only=True)
```

## Troubleshooting

### Database is locked
If you get a "database is locked" error, make sure no other processes are accessing the database.

### Memory issues
For large queries, increase DuckDB's memory limit:
```python
con.execute("SET memory_limit='4GB'")
```

### Slow queries
Add appropriate indexes or use EXPLAIN to optimize queries.

## Additional Resources

- [DuckDB Documentation](https://duckdb.org/docs/)
- [DuckDB SQL Reference](https://duckdb.org/docs/sql/introduction)
- [CDISC SDTM Standards](https://www.cdisc.org/standards/foundational/sdtm)
