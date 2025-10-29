import os, datetime
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NovaPredict API (beta)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS","*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL","")  # ne bloque pas le boot
# Conseil Supabase : terminer par ?sslmode=require

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.datetime.utcnow().isoformat()}

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# --------- modèles & routes DB ----------
class Prediction(BaseModel):
    match_id: int
    league_id: Optional[str] = None
    match_date: datetime.date
    home: str
    away: str
    p_home: Optional[float] = None
    p_draw: Optional[float] = None
    p_away: Optional[float] = None
    p_over25: Optional[float] = None
    p_under25: Optional[float] = None
    version: Optional[str] = None
    value_flag: Optional[bool] = None

def db_conn():
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL absente (configure la variable sur Render).")
    import psycopg2
    from psycopg2.extras import RealDictCursor
    return psycopg2.connect(DATABASE_URL), RealDictCursor

@app.get("/predictions", response_model=List[Prediction])
def predictions(date: Optional[str] = Query(None), league: Optional[str] = Query(None)):
    conn, RealDictCursor = db_conn()
    qdate = date or datetime.date.today().isoformat()
    sql = """
    select m.id as match_id, m.league_id, m.match_date,
           m.home_team as home, m.away_team as away,
           p.p_home, p.p_draw, p.p_away, p.p_over25, p.p_under25, p.version
    from matches m
    join predictions p on p.match_id = m.id
    where m.match_date = %s
      and (%s is null or m.league_id = %s)
    order by m.match_date, m.league_id, m.id;
    """
    with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (qdate, league, league))
        rows = cur.fetchall()
    out = []
    for r in rows:
        mx = max([v for v in [r.get("p_home"), r.get("p_draw"), r.get("p_away")] if v is not None], default=None)
        r["value_flag"] = (mx is not None and mx > 0.5)
        out.append(r)
    conn.close()
    return out

@app.get("/metrics")
def metrics():
    conn, RealDictCursor = db_conn()
    sql = "select * from model_metrics order by created_at desc limit 50;"
    with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    conn.close()
    return {"metrics": rows}


@app.post("/seed")
def seed():
    """Seed a sample match and prediction for testing."""
    # connect to DB
    conn, RealDictCursor = db_conn()
    try:
        with conn.cursor() as cur:
            # insert league
            cur.execute("INSERT INTO leagues (id, name) VALUES ('L1','Ligue 1') ON CONFLICT (id) DO NOTHING;")
            # prepare match date (today)
            match_date = datetime.date.today().isoformat()
            # insert or get match id
            cur.execute(
                """
                INSERT INTO matches (league_id, match_date, home_team, away_team, status)
                VALUES ('L1', %s, 'PSG', 'OM', 'scheduled')
                ON CONFLICT (league_id, match_date, home_team, away_team) DO NOTHING
                RETURNING id
                """,
                (match_date,)
            )
            res = cur.fetchone()
            if res:
                match_id = res[0]
            else:
                # fetch existing match id
                cur.execute(
                    """
                    SELECT id FROM matches WHERE league_id='L1' AND match_date=%s AND home_team='PSG' AND away_team='OM'
                    """,
                    (match_date,),
                )
                match_id = cur.fetchone()[0]
            # insert prediction (upsert)
            cur.execute(
                """
                INSERT INTO predictions (match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
                VALUES (%s, 'v0.1', 0.58, 0.24, 0.18, 0.62, 0.38)
                ON CONFLICT (match_id) DO UPDATE SET
                    version=EXCLUDED.version,
                    p_home=EXCLUDED.p_home,
                    p_draw=EXCLUDED.p_draw,
                    p_away=EXCLUDED.p_away,
                    p_over25=EXCLUDED.p_over25,
                    p_under25=EXCLUDED.p_under25;
                """,
                (match_id,),
            )
        conn.commit()
        return {"status": "seeded", "match_id": match_id}
    finally:
        conn.close()
