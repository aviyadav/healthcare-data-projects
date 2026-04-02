import duckdb

def main():
    con = duckdb.connect('clinical_data.duckdb', read_only=True)

    # Query any domain
    df = con.execute("SELECT * FROM DM LIMIT 10").pl()
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
    result = con.execute(query).pl()
    print(result)

    con.close()

if __name__ == "__main__":
    main()