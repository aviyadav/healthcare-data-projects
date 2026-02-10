# Clinical Data Generator

A high-performance Python tool for generating realistic fake clinical trial data following CDISC SDTM (Study Data Tabulation Model) standards. Uses multiprocessing to efficiently generate millions of records across multiple clinical domains.

## Features

- **Six CDISC SDTM Domains**: Generates comprehensive clinical trial data
- **High Performance**: Leverages multiprocessing to generate 1M+ records in seconds
- **Realistic Data**: Includes proper distributions, normal ranges, and clinical relationships
- **Partitioned Output**: Data written as partitioned Parquet files for efficient querying
- **Scalable**: Easily configurable to generate datasets of any size

## Supported Domains

### 1. **DM** (Demographics)
Subject-level demographic information including age, sex, race, country, and treatment arm.

### 2. **AE** (Adverse Events)
Adverse events experienced by subjects with severity, relationship to treatment, body system, and outcomes.

### 3. **VS** (Vital Signs)
Vital sign measurements including blood pressure, heart rate, temperature, weight, and BMI across multiple visits.

### 4. **CM** (Concomitant Medications)
Medications taken by subjects during the study with dosing information, routes, and frequencies.
- 15 common medications (Aspirin, Metformin, Lisinopril, etc.)
- Realistic dosing, routes (ORAL, IV, TOPICAL, etc.)
- Ongoing and completed medications

### 5. **LB** (Laboratory)
Laboratory test results covering hematology and chemistry panels.
- 18 lab tests (WBC, RBC, HGB, HCT, PLT, GLUC, BUN, CREAT, ALT, AST, BILI, etc.)
- Realistic values: 80% within normal range, 20% abnormal
- Normal range references included

### 6. **TV** (Trial Visits)
Planned visit schedule for the study with visit windows.
- 8 planned visits from SCREENING to END OF TREATMENT
- Visit windows (tolerance periods)
- Study days ranging from day -14 to day 140

## Installation

This project uses `uv` for dependency management. Make sure you have Python 3.x installed.

```bash
# Clone the repository
git clone <repository-url>
cd gen-fake-clinical-data

# Install dependencies (if using uv)
uv sync

# Or create a virtual environment manually
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install polars pyarrow faker numpy
```

## Usage

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the generator
python gen_clinical.py
```

### Configuration

Edit `gen_clinical.py` to adjust the number of subjects:

```python
# Line ~180
total_subjects = 5000  # Change this value
```

**Estimated Record Counts:**
- 5,000 subjects → ~1.1M records
- 10,000 subjects → ~2.2M records
- 40,000 subjects → ~8.8M records

### Output

Data is written to `clinical_data_output/` directory with the following structure:

```
clinical_data_output/
├── DM/          # Demographics
├── AE/          # Adverse Events
├── VS/          # Vital Signs
├── CM/          # Concomitant Medications
├── LB/          # Laboratory
└── TV/          # Trial Visits
```

Each domain is partitioned for efficient querying:
- **DM, VS, CM, LB**: Partitioned by `STUDYID / SITEID / USUBJID`
- **AE**: Partitioned by `STUDYID / SITEID / USUBJID / AE_INCIDENT_GROUP`
- **TV**: Partitioned by `STUDYID` (study-level data)

## Performance

**Benchmark (5,000 subjects):**
- **Total Records**: 1,142,203
- **Generation Time**: ~15 seconds
- **Processes Used**: 13 (CPU count - 1)

**Record Breakdown:**
- DM: 5,000 records
- AE: ~38,000 records
- VS: ~679,000 records
- CM: ~12,500 records
- LB: ~407,000 records
- TV: ~200 records

## Data Verification

Use the included `verification.py` script to inspect the generated data:

```bash
python verification.py
```

Or use Polars directly:

```python
import polars as pl

# Read any domain
cm = pl.read_parquet('clinical_data_output/CM/**/*.parquet')
print(cm.head())
print(f"Total CM records: {len(cm)}")
```

## Loading Data into DuckDB

For easier querying and analysis, you can load all the data into a DuckDB database:

```bash
# Load all domains into DuckDB
python load_to_duckdb.py
```

This creates a `clinical_data.duckdb` file (~7.5 MB) with all six domains as tables.

### Query the Database

**From Python:**
```python
import duckdb

con = duckdb.connect('clinical_data.duckdb', read_only=True)

# Simple query
df = con.execute("SELECT * FROM DM LIMIT 10").df()
print(df)

# Cross-domain analysis
query = '''
    SELECT 
        dm.ARM,
        COUNT(DISTINCT dm.USUBJID) as subjects,
        COUNT(ae.AESEQ) as total_aes,
        ROUND(COUNT(ae.AESEQ) * 1.0 / COUNT(DISTINCT dm.USUBJID), 2) as aes_per_subject
    FROM DM dm
    LEFT JOIN AE ae ON dm.USUBJID = ae.USUBJID
    GROUP BY dm.ARM
'''
result = con.execute(query).df()
print(result)

con.close()
```

**From Command Line:**
```bash
# Run a query
duckdb clinical_data.duckdb "SELECT ARM, COUNT(*) FROM DM GROUP BY ARM"

# Interactive mode
duckdb clinical_data.duckdb
```

**See [DUCKDB_GUIDE.md](DUCKDB_GUIDE.md) for comprehensive query examples and usage tips.**

## Domain Details

### CM (Concomitant Medications)
- **Medications**: 15 common drugs across therapeutic areas
- **Fields**: Medication name, indication, dose, route, frequency, start/end dates
- **Generation**: 0-5 medications per subject
- **Realism**: Mix of ongoing and completed medications

### LB (Laboratory)
- **Test Panels**: Hematology (7 tests) + Chemistry (11 tests)
- **Fields**: Test code, test name, result value, units, normal ranges
- **Generation**: All tests for each visit (aligned with VS visits)
- **Realism**: 80% values within normal range, 20% abnormal (high/low)

### TV (Trial Visits)
- **Visits**: 8 planned visits over 154 days
- **Fields**: Visit number, visit name, planned day, visit window
- **Generation**: Study-level (not subject-level)
- **Schedule**: SCREENING (-14d) → BASELINE (1d) → Weekly visits → END OF TREATMENT (140d)

## Technical Details

- **Language**: Python 3.x
- **Dependencies**: Polars, PyArrow, Faker, NumPy
- **Parallelization**: Multiprocessing with automatic CPU detection
- **Output Format**: Partitioned Parquet files
- **Data Standards**: CDISC SDTM compliant

## Customization

### Adding More Medications

Edit the `CM_MEDICATIONS` constant in `gen_clinical.py`:

```python
CM_MEDICATIONS = [
    ("MEDCODE", "Medication Name", "Indication"),
    # Add more medications here
]
```

### Adding More Lab Tests

Edit the `LB_TESTS` constant:

```python
LB_TESTS = [
    ("TESTCD", "Test Name", "Unit", low_normal, high_normal),
    # Add more tests here
]
```

### Modifying Visit Schedule

Edit the `TV_VISITS` constant:

```python
TV_VISITS = [
    (visit_num, "VISIT NAME", planned_day, window),
    # Add or modify visits here
]
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]
