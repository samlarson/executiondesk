import io
import psycopg2
from sqlalchemy import create_engine

try:
    connect_str = "dbname='dbfin' user='slars' host='localhost' " + \
                  "password='ayylmao'"
    conn = psycopg2.connect(connect_str)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE users(
        time timestamp PRIMARY KEY,
        open float,
        high float,
        low float,
        close float,
        volume int
    )
    """)

    conn.commit()


except Exception as e:
    print("WARNING: Cannot connect to database. Please check credentials")
    print(e)


class LoadDB:
    def load_table(self, df):

        engine = create_engine('postgresql+psycopg2://slars:ayylmao@slars:5432/dbfin')
        df.head(0).to_sql('ohlc_table', engine, if_exists='replace', index=False)
        conn = engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        # contents = output.getvalue()
        cur.copy_from(output, 'ohlc_table', null="")
        conn.commit()
