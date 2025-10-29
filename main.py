import os, datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Header, Body
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NovaPredict API (beta)")

# --- CORS (OK pour test, on restreindra plus tard) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENV ---
DATABASE_URL = os.getenv("DATABASE_URL", "")  # ne bloque pas le boot
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# --- DB helpers ---
def db_conn():
    """Retourne (connection, RealDictCursor) ou 503 si DATABASE_URL manquante."""
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL absente (configure la variable sur Render).")
    import psycopg2
    from psycopg2.extras import RealDictCursor
    return psycopg2.connect(DATABASE_URL), RealDictCursor

def _conn_raw():
    """Connection simple (sans RealDictCursor) pour usages internes."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL manquante")
    import psycopg2
    return psycopg2.connect(DATABASE_URL)

# --- Models ---
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

# --- Health & Root ---
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.datetime.utcnow().isoformat()}

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# --- Public: predictions & metrics ---
@app.get("/predictions", response_model=List[Prediction])
def predictions(date: Optional[str] = Query(None), league: Optional[str] = Query(None)):
    """
    Renvoie les prédictions pour une date (YYYY-MM-DD) et une ligue optionnelle.
    """
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

# --- Seed (corrigé pour utiliser les contraintes) ---
@app.post("/seed")
def seed():
    """Seed d'un match et prédiction de test (date = aujourd'hui)."""
    conn, RealDictCursor = db_conn()
    try:
        with conn.cursor() as cur:
            # ligue
            cur.execute("""
                INSERT INTO leagues (id, name)
                VALUES ('L1','Ligue 1')
                ON CONFLICT (id) DO NOTHING;
            """)

            match_date = datetime.date.today().isoformat()

            # match : utiliser la contrainte uq_match
            cur.execute("""
                INSERT INTO matches (league_id, match_date, home_team, away_team, status)
                VALUES ('L1', %s, 'PSG', 'OM', 'scheduled')
                ON CONFLICT ON CONSTRAINT uq_match DO NOTHING
                RETURNING id;
            """, (match_date,))
            row = cur.fetchone()
            if row:
                match_id = row[0]
            else:
                cur.execute("""
                    SELECT id FROM matches
                    WHERE league_id='L1' AND match_date=%s AND home_team='PSG' AND away_team='OM'
                    LIMIT 1;
                """, (match_date,))
                match_id = cur.fetchone()[0]

            # prediction : utiliser la contrainte uq_pred (match_id, version)
            cur.execute("""
                INSERT INTO predictions (match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
                VALUES (%s, 'v0.1', 0.58, 0.24, 0.18, 0.62, 0.38)
                ON CONFLICT ON CONSTRAINT uq_pred DO UPDATE SET
                    p_home=EXCLUDED.p_home,
                    p_draw=EXCLUDED.p_draw,
                    p_away=EXCLUDED.p_away,
                    p_over25=EXCLUDED.p_over25,
                    p_under25=EXCLUDED.p_under25,
                    version=EXCLUDED.version;
            """, (match_id,))
        conn.commit()
        return {"status": "seeded", "match_id": match_id}
    finally:
        conn.close()

# =========================
# ADMIN (sécurisé par token)
# =========================

def require_admin(x_admin_token: str = Header(default="")):
    """Vérifie le header 'X-Admin-Token'."""
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

def upsert_match(cur, league_id: str, d: str, home: str, away: str, status: str):
    """
    Insert/Update de match avec uq_match ; renvoie match_id.
    """
    from psycopg2.extras import RealDictCursor as _R
    cur.execute("""
        INSERT INTO matches (league_id, match_date, home_team, away_team, status)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT ON CONSTRAINT uq_match DO UPDATE
        SET status = EXCLUDED.status
        RETURNING id;
    """, (league_id, d, home, away, status))
    row = cur.fetchone()
    if not row:
        cur.execute("""
            SELECT id FROM matches
            WHERE league_id=%s AND match_date=%s AND home_team=%s AND away_team=%s
            LIMIT 1;
        """, (league_id, d, home, away))
        row = cur.fetchone()
    # row peut être tuple (int) ou dict selon le cursor
    if isinstance(row, dict):
        return row.get("id")
    return row[0]

def upsert_prediction(cur, match_id: int, probs: Dict[str, Any]):
    """
    Insert/Update de prediction avec uq_pred (match_id, version).
    """
    version = probs.get("version", "v0.1")
    cur.execute("""
        INSERT INTO predictions (match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT ON CONSTRAINT uq_pred DO UPDATE SET
            p_home=EXCLUDED.p_home,
            p_draw=EXCLUDED.p_draw,
            p_away=EXCLUDED.p_away,
            p_over25=EXCLUDED.p_over25,
            p_under25=EXCLUDED.p_under25,
            version=EXCLUDED.version;
    """, (
        match_id, version,
        probs.get("p_home"), probs.get("p_draw"), probs.get("p_away"),
        probs.get("p_over25"), probs.get("p_under25"),
    ))

@app.post("/admin/upsert", tags=["admin"])
def admin_upsert(payload: Dict[str, Any] = Body(...), _=require_admin()):
    """
    Charge / met à jour des matchs + prédictions via JSON.

    Body attendu :
    {
      "date": "YYYY-MM-DD",
      "league_id": "L1",
      "matches": [
        {
          "home": "PSG",
          "away": "OM",
          "status": "scheduled",
          "prediction": {
            "version": "v0.1",
            "p_home": 0.58, "p_draw": 0.24, "p_away": 0.18,
            "p_over25": 0.62, "p_under25": 0.38
          }
        }
      ]
    }
    """
    d = payload["date"]
    league_id = payload.get("league_id", "L1")
    matches = payload.get("matches", [])
    inserted = 0

    conn, RealDictCursor = db_conn()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ligue
            cur.execute("""
                INSERT INTO leagues (id, name)
                VALUES (%s, %s)
                ON CONFLICT (id) DO NOTHING;
            """, (league_id, league_id))

            for m in matches:
                mid = upsert_match(
                    cur, league_id, d,
                    m["home"], m["away"],
                    m.get("status", "scheduled")
                )
                if "prediction" in m:
                    upsert_prediction(cur, mid, m["prediction"])
                inserted += 1
    finally:
        conn.close()

    return {"ok": True, "count": inserted, "date": d, "league": league_id}
