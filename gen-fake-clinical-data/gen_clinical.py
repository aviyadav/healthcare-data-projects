import sys
import os
import random
import time
import shutil
import uuid
import datetime
from multiprocessing import Pool, cpu_count

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import numpy as np
from faker import Faker

# Constants
DOMAINS = ["DM", "AE", "VS", "CM", "LB", "TV"]
STUDIES = [f"STUDY-{i:03d}" for i in range(1, 3)] 
SITES = [f"SITE-{i:03d}" for i in range(1, 6)] 
AE_TERMS = ["Headache", "Nausea", "Dizziness", "Fatigue", "Rash", "Pyrexia", "Vomiting", "Pain"]
AE_BODY_SYSTEMS = ["Nervous System", "Gastrointestinal", "General", "Skin", "Respiratory System", "Cardiovascular System", "Musculoskeletal System", "Renal System", "Gastrointestinal System", "Hematologic System", "Neurological System", "Psychiatric System", "Renal System", "Respiratory System", "Skin System", "Urinary System", "Vascular System", "Endocrine System", "Immune System", "Infectious System", "Musculoskeletal System", "Neurological System", "Psychiatric System", "Renal System", "Respiratory System", "Skin System", "Urinary System", "Vascular System"]
SEVERITY = ["MILD", "MODERATE", "SEVERE", "SEVERE AND LIFE THREATENING", "CRITICAL", "FATAL", "MILD AND LIFE THREATENING", "MODERATE AND LIFE THREATENING", "SEVERE AND LIFE THREATENING", "CRITICAL AND LIFE THREATENING", "FATAL AND LIFE THREATENING", "MILD AND LIFE THREATENING", "MODERATE AND LIFE THREATENING", "SEVERE AND LIFE THREATENING", "CRITICAL AND LIFE THREATENING", "FATAL AND LIFE THREATENING"]
RELATIONSHIP = ["NOT RELATED", "POSSIBLY RELATED", "RELATED"]
VS_TESTS = [
    ("SYSBP", "Systolic Blood Pressure", "mmHg"),
    ("DIABP", "Diastolic Blood Pressure", "mmHg"),
    ("HR", "Heart Rate", "beats/min"),
    ("TEMP", "Temperature", "C"),
    ("WEIGHT", "Weight", "kg"),
    ("BMI", "Body Mass Index", "kg/m^2"),
    ("SBP", "Systolic Blood Pressure", "mmHg"),
    ("DBP", "Diastolic Blood Pressure", "mmHg"),
    ("HR", "Heart Rate", "beats/min"),
    ("TEMP", "Temperature", "C"),
    ("WEIGHT", "Weight", "kg"),
    ("BMI", "Body Mass Index", "kg/m^2"),
    ("SBP", "Systolic Blood Pressure", "mmHg"),
    ("DBP", "Diastolic Blood Pressure", "mmHg"),
    ("HR", "Heart Rate", "beats/min"),
    ("TEMP", "Temperature", "C"),
    ("WEIGHT", "Weight", "kg"),
    ("BMI", "Body Mass Index", "kg/m^2"),
    ("SBP", "Systolic Blood Pressure", "mmHg"),
    ("DBP", "Diastolic Blood Pressure", "mmHg"),
    ("HR", "Heart Rate", "beats/min"),
    ("TEMP", "Temperature", "C"),
    ("WEIGHT", "Weight", "kg"),
    ("BMI", "Body Mass Index", "kg/m^2"),
    ("SBP", "Systolic Blood Pressure", "mmHg"),
    ("DBP", "Diastolic Blood Pressure", "mmHg"),
    ("HR", "Heart Rate", "beats/min"),
    ("TEMP", "Temperature", "C"),
    ("WEIGHT", "Weight", "kg"),
    ("BMI", "Body Mass Index", "kg/m^2")
]

# CM (Concomitant Medications) Constants
CM_MEDICATIONS = [
    ("ASPIRIN", "Aspirin", "Cardiovascular"),
    ("IBUPROFEN", "Ibuprofen", "Pain/Inflammation"),
    ("METFORMIN", "Metformin", "Diabetes"),
    ("LISINOPRIL", "Lisinopril", "Hypertension"),
    ("ATORVASTATIN", "Atorvastatin", "Cholesterol"),
    ("LEVOTHYROXINE", "Levothyroxine", "Thyroid"),
    ("OMEPRAZOLE", "Omeprazole", "Gastric"),
    ("AMLODIPINE", "Amlodipine", "Hypertension"),
    ("METOPROLOL", "Metoprolol", "Cardiovascular"),
    ("LOSARTAN", "Losartan", "Hypertension"),
    ("GABAPENTIN", "Gabapentin", "Neuropathic Pain"),
    ("SERTRALINE", "Sertraline", "Depression"),
    ("SIMVASTATIN", "Simvastatin", "Cholesterol"),
    ("MONTELUKAST", "Montelukast", "Asthma"),
    ("ESCITALOPRAM", "Escitalopram", "Anxiety/Depression")
]
CM_ROUTES = ["ORAL", "IV", "TOPICAL", "SUBCUTANEOUS", "INTRAMUSCULAR"]
CM_FORMS = ["TABLET", "CAPSULE", "INJECTION", "CREAM", "SOLUTION"]
CM_FREQUENCIES = ["QD", "BID", "TID", "QID", "PRN", "Q12H", "WEEKLY"]

# LB (Laboratory) Constants
LB_TESTS = [
    # Hematology
    ("WBC", "White Blood Cell Count", "10^9/L", 4.0, 11.0),
    ("RBC", "Red Blood Cell Count", "10^12/L", 4.5, 5.9),
    ("HGB", "Hemoglobin", "g/dL", 13.5, 17.5),
    ("HCT", "Hematocrit", "%", 38.0, 50.0),
    ("PLT", "Platelet Count", "10^9/L", 150.0, 400.0),
    ("NEUT", "Neutrophils", "%", 40.0, 70.0),
    ("LYMPH", "Lymphocytes", "%", 20.0, 40.0),
    # Chemistry
    ("GLUC", "Glucose", "mg/dL", 70.0, 100.0),
    ("BUN", "Blood Urea Nitrogen", "mg/dL", 7.0, 20.0),
    ("CREAT", "Creatinine", "mg/dL", 0.7, 1.3),
    ("ALT", "Alanine Aminotransferase", "U/L", 7.0, 56.0),
    ("AST", "Aspartate Aminotransferase", "U/L", 10.0, 40.0),
    ("BILI", "Total Bilirubin", "mg/dL", 0.1, 1.2),
    ("ALB", "Albumin", "g/dL", 3.5, 5.5),
    ("CA", "Calcium", "mg/dL", 8.5, 10.5),
    ("NA", "Sodium", "mmol/L", 135.0, 145.0),
    ("K", "Potassium", "mmol/L", 3.5, 5.0),
    ("CL", "Chloride", "mmol/L", 96.0, 106.0)
]

# TV (Trial Visits) Constants
TV_VISITS = [
    (1, "SCREENING", -14, 7),
    (2, "BASELINE", 1, 0),
    (3, "WEEK 2", 14, 3),
    (4, "WEEK 4", 28, 3),
    (5, "WEEK 8", 56, 5),
    (6, "WEEK 12", 84, 7),
    (7, "WEEK 16", 112, 7),
    (8, "END OF TREATMENT", 140, 7)
]

def generate_chunk(args):
    chunk_id, n_subjects, output_base = args
    fake = Faker()
    
    dm_rows = []
    ae_rows = []
    vs_rows = []
    cm_rows = []
    lb_rows = []
    tv_rows = []
    
    for _ in range(n_subjects):
        # DM Generation
        study = random.choice(STUDIES)
        site = random.choice(SITES)
        usubjid = f"{study}-{site}-{uuid.uuid4().hex[:8]}"
        age = random.randint(18, 85)
        sex = random.choice(["M", "F"])
        race = random.choice(["WHITE", "BLACK", "ASIAN", "OTHER"])
        
        dm_rows.append({
            # Mandatory context attributes
            "STUDY": study,
            "SITE": site,
            "SUBJECT": usubjid,
            "VISIT": "SCREENING",
            "FORM": "DM",
            # Domain-specific attributes
            "STUDYID": study,
            "SITEID": site,
            "USUBJID": usubjid,
            "DOMAIN": "DM",
            "AGE": age,
            "SEX": sex,
            "RACE": race,
            "COUNTRY": fake.country_code(),
            "DMDTC": fake.date_between(start_date='-2y', end_date='today'),
            "ARM": random.choice(["Placebo", "Active 10mg", "Active 20mg"])
        })
        
        # AE Generation
        num_aes = random.randint(0, 15)
        for seq in range(1, num_aes + 1):
            term = random.choice(AE_TERMS)
            start_date = fake.date_between(start_date='-1y', end_date='today')
            ae_rows.append({
                # Mandatory context attributes
                "STUDY": study,
                "SITE": site,
                "SUBJECT": usubjid,
                "VISIT": f"VISIT {seq}",
                "FORM": "AE",
                # Domain-specific attributes
                "STUDYID": study,
                "SITEID": site,
                "USUBJID": usubjid,
                "DOMAIN": "AE",
                "AESEQ": seq,
                "AETERM": term,
                "AEDECOD": term.upper(),
                "AEBODSYS": random.choice(AE_BODY_SYSTEMS),
                "AESTDTC": start_date,
                "AEENDTC": fake.date_between(start_date=start_date, end_date='today'),
                "AESEV": random.choice(SEVERITY),
                "AEREL": random.choice(RELATIONSHIP),
                "AEOUT": "RECOVERED",
                "AE_INCIDENT_GROUP": random.choice(["TypeA", "TypeB"])
            })

        # VS Generation
        num_visits = random.randint(1, 8)
        for visit_num in range(1, num_visits + 1):
            visit_label = f"VISIT {visit_num}"
            visit_date = fake.date_between(start_date='-1y', end_date='today')
            for testcd, testname, unit in VS_TESTS:
                val = 0.0
                if testcd == "SYSBP": val = float(random.randint(100, 160))
                elif testcd == "DIABP": val = float(random.randint(60, 100))
                elif testcd == "HR": val = float(random.randint(50, 100))
                elif testcd == "TEMP": val = round(random.uniform(36.0, 38.0), 1)
                elif testcd == "WEIGHT": val = round(random.uniform(50.0, 120.0), 1)

                vs_rows.append({
                    # Mandatory context attributes
                    "STUDY": study,
                    "SITE": site,
                    "SUBJECT": usubjid,
                    "VISIT": visit_label,
                    "FORM": "VS",
                    # Domain-specific attributes
                    "STUDYID": study,
                    "SITEID": site,
                    "USUBJID": usubjid,
                    "DOMAIN": "VS",
                    "VSTESTCD": testcd,
                    "VSTEST": testname,
                    "VSORRES": val,
                    "VSORRESU": unit,
                    "VSDTC": visit_date
                })

        # CM Generation (Concomitant Medications)
        num_meds = random.randint(0, 5)
        for seq in range(1, num_meds + 1):
            medcd, medname, indication = random.choice(CM_MEDICATIONS)
            start_date = fake.date_between(start_date='-2y', end_date='today')
            # Some medications are ongoing, some have ended
            ongoing = random.choice([True, False])
            end_date = None if ongoing else fake.date_between(start_date=start_date, end_date='today')

            cm_rows.append({
                # Mandatory context attributes
                "STUDY": study,
                "SITE": site,
                "SUBJECT": usubjid,
                "VISIT": f"VISIT {seq}",
                "FORM": "CM",
                # Domain-specific attributes
                "STUDYID": study,
                "SITEID": site,
                "USUBJID": usubjid,
                "DOMAIN": "CM",
                "CMSEQ": seq,
                "CMTRT": medname,
                "CMDECOD": medcd,
                "CMCAT": indication,
                "CMSTDTC": start_date,
                "CMENDTC": end_date,
                "CMDOSE": round(random.uniform(5.0, 500.0), 1),
                "CMDOSU": "mg",
                "CMDOSFRM": random.choice(CM_FORMS),
                "CMROUTE": random.choice(CM_ROUTES),
                "CMDOSFRQ": random.choice(CM_FREQUENCIES)
            })

        # LB Generation (Laboratory Tests)
        # Generate lab results for each visit (aligned with VS visits)
        for visit_num in range(1, num_visits + 1):
            visit_label = f"VISIT {visit_num}"
            visit_date = fake.date_between(start_date='-1y', end_date='today')
            for testcd, testname, unit, low_normal, high_normal in LB_TESTS:
                # Generate realistic values: 80% within normal range, 20% outside
                if random.random() < 0.8:
                    # Within normal range
                    val = round(random.uniform(low_normal, high_normal), 2)
                else:
                    # Outside normal range (either low or high)
                    if random.choice([True, False]):
                        val = round(random.uniform(low_normal * 0.5, low_normal), 2)
                    else:
                        val = round(random.uniform(high_normal, high_normal * 1.5), 2)

                lb_rows.append({
                    # Mandatory context attributes
                    "STUDY": study,
                    "SITE": site,
                    "SUBJECT": usubjid,
                    "VISIT": visit_label,
                    "FORM": "LB",
                    # Domain-specific attributes
                    "STUDYID": study,
                    "SITEID": site,
                    "USUBJID": usubjid,
                    "DOMAIN": "LB",
                    "LBTESTCD": testcd,
                    "LBTEST": testname,
                    "LBORRES": val,
                    "LBORRESU": unit,
                    "LBSTNRLO": low_normal,
                    "LBSTNRHI": high_normal,
                    "LBDTC": visit_date
                })

    # TV Generation (Trial Visits) — Study-level, one schedule per unique STUDYID.
    # Collect unique studies seen during subject generation, then emit TV rows.
    studies_in_chunk = {row["STUDYID"] for row in dm_rows}
    for study_id in sorted(studies_in_chunk):
        for visitnum, visit, planned_day, window in TV_VISITS:
            tv_rows.append({
                # Mandatory context attributes
                "STUDY": study_id,
                "SITE": "ALL",
                "SUBJECT": "ALL",
                "VISIT": visit,
                "FORM": "TV",
                # Domain-specific attributes
                "STUDYID": study_id,
                "DOMAIN": "TV",
                "VISITNUM": visitnum,
                "TVSTRL": planned_day - window,
                "TVENRL": planned_day + window,
                "ARMCD": "ALL"
            })

    # Convert to Polars DataFrames and Write
    if dm_rows:
        df_dm = pl.DataFrame(dm_rows)
        _write_dataset(df_dm, output_base, "DM", ["STUDYID", "SITEID", "USUBJID"], chunk_id)
        
    if ae_rows:
        df_ae = pl.DataFrame(ae_rows)
        _write_dataset(df_ae, output_base, "AE", ["STUDYID", "SITEID", "USUBJID", "AE_INCIDENT_GROUP"], chunk_id)

    if vs_rows:
        df_vs = pl.DataFrame(vs_rows)
        _write_dataset(df_vs, output_base, "VS", ["STUDYID", "SITEID", "USUBJID"], chunk_id)

    if cm_rows:
        df_cm = pl.DataFrame(cm_rows)
        _write_dataset(df_cm, output_base, "CM", ["STUDYID", "SITEID", "USUBJID"], chunk_id)

    if lb_rows:
        df_lb = pl.DataFrame(lb_rows)
        _write_dataset(df_lb, output_base, "LB", ["STUDYID", "SITEID", "USUBJID"], chunk_id)

    if tv_rows:
        df_tv = pl.DataFrame(tv_rows)
        _write_dataset(df_tv, output_base, "TV", ["STUDYID"], chunk_id)

    return len(dm_rows), len(ae_rows), len(vs_rows), len(cm_rows), len(lb_rows), len(tv_rows)

def _write_dataset(df, base_path, domain, partition_cols, chunk_id):
    path = os.path.join(base_path, domain)

    # Use chunk_id and uuid in filename to avoid collisions
    unique_id = uuid.uuid4().hex[:6]
    fname = "part-{i}-" + f"{chunk_id}-{unique_id}.parquet"

    # Build a partitioning object that preserves partition columns in the files.
    # Using DirectoryPartitioning so column values appear in directory names AND
    # remain embedded in each parquet file — mandatory attributes are always readable
    # regardless of whether the consumer is partition-aware.
    partitioning = ds.partitioning(
        pa.schema([pa.field(col, pa.string()) for col in partition_cols]),
        flavor="hive"
    )

    table = df.to_arrow()

    ds.write_dataset(
        table,
        base_dir=path,
        partitioning=partitioning,
        format="parquet",
        existing_data_behavior="overwrite_or_ignore",
        basename_template=fname
    )

def main():
    output_dir = "clinical_data_output"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except OSError:
            pass # Handle case where files are locked or busy
            
    # Target: To get near 1M records
    # ~25 records per subject (1 DM + 7 AE + 15 VS)
    # 40,000 subjects => ~1M records
    # For testing, we use 2,000 subjects (~50k records)
    total_subjects = 5000 
    
    num_procs = max(1, cpu_count() - 1) # Leave one core free
    chunk_size = total_subjects // num_procs
    
    tasks = []
    for i in range(num_procs):
        count = chunk_size + (1 if i < total_subjects % num_procs else 0)
        tasks.append((i, count, output_dir))
        
    print(f"Starting generation for {total_subjects} subjects using {num_procs} processes...")
    start_time = time.time()
    
    with Pool(num_procs) as pool:
        results = pool.map(generate_chunk, tasks)
        
    end_time = time.time()
    
    total_dm = sum(r[0] for r in results)
    total_ae = sum(r[1] for r in results)
    total_vs = sum(r[2] for r in results)
    total_cm = sum(r[3] for r in results)
    total_lb = sum(r[4] for r in results)
    total_tv = sum(r[5] for r in results)
    total_records = total_dm + total_ae + total_vs + total_cm + total_lb + total_tv
    
    print("-" * 30)
    print(f"Generation Complete in {end_time - start_time:.2f} seconds.")
    print("-" * 30)
    print(f"Total DM Records: {total_dm}")
    print(f"Total AE Records: {total_ae}")
    print(f"Total VS Records: {total_vs}")
    print(f"Total CM Records: {total_cm}")
    print(f"Total LB Records: {total_lb}")
    print(f"Total TV Records: {total_tv}")
    print(f"Total Records: {total_records}")
    print("-" * 30)
    print(f"Data written to: {os.path.abspath(output_dir)}")
    print("Partitioning Scheme:")
    print("  DM: STUDYID / SITEID / USUBJID")
    print("  AE: STUDYID / SITEID / USUBJID / AE_INCIDENT_GROUP")
    print("  VS: STUDYID / SITEID / USUBJID")
    print("  CM: STUDYID / SITEID / USUBJID")
    print("  LB: STUDYID / SITEID / USUBJID")
    print("  TV: STUDYID")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds.")
