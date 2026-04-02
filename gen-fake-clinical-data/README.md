# Clinical Data Generator

A high-performance Python tool for generating realistic fake clinical trial data following **CDISC SDTM** (Study Data Tabulation Model) standards. Uses multiprocessing to efficiently generate millions of records across multiple clinical domains, writing partitioned Parquet files with all attributes embedded in each file.

## Features

- **Six CDISC SDTM Domains**: DM, AE, VS, CM, LB, and TV
- **Mandatory Context Attributes**: Every record carries `STUDY`, `SITE`, `SUBJECT`, `VISIT`, and `FORM` for consistent cross-domain filtering
- **High Performance**: Leverages multiprocessing to generate 1M+ records in seconds
- **Realistic Data**: Proper distributions, normal ranges, and clinical relationships
- **Hive-Partitioned Output**: Parquet files partitioned by key columns; partition values are **embedded in each file** so they are readable without partition-awareness
- **DuckDB Ready**: Load all domains into DuckDB for SQL querying and analysis

## Mandatory Attributes

Every row in every domain includes these five standard context attributes, enabling uniform filtering and joining across all datasets:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `STUDY`   | Study identifier | `STUDY-001` |
| `SITE`    | Site identifier  | `SITE-003` |
| `SUBJECT` | Subject identifier (USUBJID) | `STUDY-001-SITE-003-abc12345` |
| `VISIT`   | Visit label | `VISIT 2`, `SCREENING` |
| `FORM`    | Domain/form name | `DM`, `AE`, `VS`, `CM`, `LB`, `TV` |

> **Note**: For TV (Trial Visits), which is a study-level dataset, `SITE` and `SUBJECT` are set to `"ALL"`.

## Supported Domains

### 1. **DM** (Demographics)
Subject-level demographic information including age, sex, race, country, and treatment arm.
- `VISIT` = `"SCREENING"` (fixed for the demographics form)

### 2. **AE** (Adverse Events)
Adverse events experienced by subjects with severity, relationship to treatment, body system, and outcomes.
- Generated 0–15 adverse events per subject
- `VISIT` aligned to the event sequence number

### 3. **VS** (Vital Signs)
Vital sign measurements including blood pressure, heart rate, temperature, and weight across multiple visits.
- 1–8 visits per subject
- Tests: SYSBP, DIABP, HR, TEMP, WEIGHT, BMI, SBP, DBP

### 4. **CM** (Concomitant Medications)
Medications taken by subjects during the study with dosing information, routes, and frequencies.
- 15 common medications (Aspirin, Metformin, Lisinopril, etc.)
- Realistic dosing, routes (ORAL, IV, TOPICAL, etc.)
- Mix of ongoing and completed medications

### 5. **LB** (Laboratory)
Laboratory test results covering hematology and chemistry panels.
- 18 lab tests (WBC, RBC, HGB, HCT, PLT, GLUC, BUN, CREAT, ALT, AST, BILI, etc.)
- Realistic values: 80% within normal range, 20% abnormal (high/low)
- Normal range references (`LBSTNRLO`, `LBSTNRHI`) included

### 6. **TV** (Trial Visits)
Planned visit schedule — **study-level**, generated once per unique `STUDYID`, not per subject.
- 8 planned visits: SCREENING → BASELINE → WEEK 2/4/8/12/16 → END OF TREATMENT
- Study days ranging from day −14 to day 140
- Visit windows (tolerance periods) encoded in `TVSTRL` / `TVENRL`

## Installation

This project uses `uv` for dependency management and requires **Python ≥ 3.14**.

```bash
# Clone the repository
git clone <repository-url>
cd gen-fake-clinical-data

# Install dependencies with uv (recommended)
uv sync

# Or with pip in a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install polars pyarrow faker numpy duckdb
```

### Dependencies

| Package    | Version   | Purpose                              |
|------------|-----------|--------------------------------------|
| `polars`   | ≥ 1.38.1  | DataFrame construction and I/O       |
| `pyarrow`  | ≥ 23.0.0  | Partitioned Parquet writing          |
| `faker`    | ≥ 40.4.0  | Realistic synthetic data generation  |
| `numpy`    | ≥ 2.4.2   | Numeric distributions                |
| `duckdb`   | ≥ 1.4.4   | SQL querying over Parquet files      |

## Usage

### Generate Data

```bash
source .venv/bin/activate
python gen_clinical.py
```

### Configure Volume

Edit `gen_clinical.py` to adjust the number of subjects:

```python
total_subjects = 5000  # Change this value
```

**Estimated Record Counts (5,000 subjects):**

| Domain | Records |
|--------|---------|
| DM     | 5,000   |
| AE     | ~38,000 |
| VS     | ~679,000 |
| CM     | ~12,500 |
| LB     | ~407,000 |
| TV     | ~200    |
| **Total** | **~1.1M** |

### Output Structure

```
clinical_data_output/
├── DM/          # Demographics          → partitioned by STUDY/SITE/SUBJECT
├── AE/          # Adverse Events        → partitioned by STUDY/SITE/SUBJECT/AE_INCIDENT_GROUP
├── VS/          # Vital Signs           → partitioned by STUDY/SITE/SUBJECT
├── CM/          # Concomitant Meds      → partitioned by STUDY/SITE/SUBJECT
├── LB/          # Laboratory            → partitioned by STUDY/SITE/SUBJECT
└── TV/          # Trial Visits          → partitioned by STUDY
```

Partition columns use **hive-style** directory naming (`col=value/`) and are **also kept inside each Parquet file**, so data is fully readable without a partition-aware reader.

## Load into DuckDB

```bash
python load_to_duckdb.py
```

This creates `clinical_data.duckdb` (~15 MB) with all six domains as tables.

### Query with Python (Polars)

```python
import duckdb

con = duckdb.connect('clinical_data.duckdb', read_only=True)

# Return results as a Polars DataFrame
df = con.execute("SELECT * FROM DM LIMIT 10").pl()
print(df)

# Cross-domain join → Polars DataFrame
query = '''
    SELECT
        dm.STUDY,
        dm.ARM,
        COUNT(DISTINCT dm.USUBJID) AS subjects,
        COUNT(ae.AESEQ)            AS total_aes,
        ROUND(COUNT(ae.AESEQ) * 1.0 / COUNT(DISTINCT dm.USUBJID), 2) AS aes_per_subject
    FROM DM dm
    LEFT JOIN AE ae ON dm.USUBJID = ae.USUBJID
    GROUP BY dm.STUDY, dm.ARM
    ORDER BY dm.STUDY, dm.ARM
'''
result = con.execute(query).pl()
print(result)

con.close()
```

### Query from the Command Line

```bash
# Quick query
duckdb clinical_data.duckdb "SELECT ARM, COUNT(*) FROM DM GROUP BY ARM"

# Interactive REPL
duckdb clinical_data.duckdb
```

**See [DUCKDB_GUIDE.md](DUCKDB_GUIDE.md) for comprehensive query examples.**

### Query Parquet Directly with Polars

```python
import polars as pl

# Read any domain (all attributes available, including mandatory ones)
df = pl.read_parquet('clinical_data_output/CM/**/*.parquet')
print(df.select(["STUDY", "SITE", "SUBJECT", "VISIT", "FORM", "CMTRT"]).head())

# Filter using mandatory attributes
aes = (
    pl.read_parquet('clinical_data_output/AE/**/*.parquet')
    .filter(pl.col("STUDY") == "STUDY-001")
)
```

## Data Verification

```bash
python verification.py
```

## Technical Details

- **Language**: Python ≥ 3.14
- **Parallelization**: `multiprocessing.Pool` with `cpu_count() - 1` workers
- **Output Format**: Hive-partitioned Parquet (partition columns embedded in files)
- **DataFrame Library**: Polars (build) + PyArrow (write)
- **Data Standard**: CDISC SDTM-aligned

## Customization

### Adding Medications

```python
CM_MEDICATIONS = [
    ("MEDCODE", "Medication Name", "Indication"),
    # Add entries here
]
```

### Adding Lab Tests

```python
LB_TESTS = [
    ("TESTCD", "Test Name", "Unit", low_normal, high_normal),
    # Add entries here
]
```

### Modifying the Visit Schedule

```python
TV_VISITS = [
    (visit_num, "VISIT NAME", planned_day, window),
    # Add or modify visits here
]
```

## License

[Add your license here]
