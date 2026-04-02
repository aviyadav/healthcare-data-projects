duckdb clinical_data.duckdb "SELECT * FROM DM LIMIT 5"

duckdb clinical_data.duckdb "
SELECT
    dm.USUBJID,
    dm.AGE,
    dm.SEX,
    COUNT(ae.AESEQ) as adverse_event_count
FROM DM dm
LEFT JOIN AE ae ON dm.USUBJID = ae.USUBJID
GROUP BY dm.USUBJID, dm.AGE, dm.SEX
LIMIT 10
"