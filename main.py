import os, datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Query, HTTPException, Header, Body, Depends
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NovaPredict API (beta)")

# --- CORS (permissif pour tests ; restreindre plus tard à ton domaine Vercel) ---
allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed if allowed else ["*"],
    allow_origin_regex=origin_regex or None,   # <--- support regex
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENV ---
DATABASE_URL = os.getenv("DATABASE_URL", "")  # ne bloque pas le boot
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# --- Helpers DB ---
def db_conn():
    """
    Retourne (connection, RealDictCursor) ou 503 si DATABASE_URL manquante.
    psycopg2 est importé localement pour réduire les erreurs à l'import uvicorn.
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL absente (configure la variable sur Render).")
    import psycopg2
    from psycopg2.extras import RealDictCursor
    return psycopg2.connect(DATABASE_URL), RealDictCursor

# --- Pydantic models ---
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
    # Redirige vers la doc Swagger
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

# --- Seed (exemple) ---
@app.post("/seed")
def seed():
    """
    Seed d'un match et d'une prédiction de test (date = aujourd'hui).
    Utilise les contraintes uq_match / uq_pred pour éviter les doublons.
    """
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

            # match (uq_match)
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

            # prediction (uq_pred)
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
    """Vérifie le header 'X-Admin-Token' pour les routes admin."""
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

def upsert_match(cur, league_id: str, d: str, home: str, away: str, status: str):
    """
    Insert/Update de match avec uq_match ; renvoie match_id.
    """
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
    return row["id"] if isinstance(row, dict) else row[0]

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
def admin_upsert(
    payload: Dict[str, Any] = Body(...),
    _ = Depends(require_admin),   # <-- la dépendance correcte (pas d'appel direct)
):
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
    # ============
# BASELINE (cotes -> probabilités) + /admin/refresh
# ============

# Helpers sûrs (compatibles Python 3.11+ ; tu es en 3.13 sur Render)
def _inv_safe(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        x = float(x)
        if x <= 0:
            return None
        return 1.0 / x
    except Exception:
        return None

def _normalize_triple(a: float | None, b: float | None, c: float | None):
    vals = [v for v in (a, b, c) if v is not None]
    if not vals:
        return None, None, None
    s = sum(vals)
    if s <= 0:
        return None, None, None
    def nz(v):
        return (v / s) if v is not None else None
    return nz(a), nz(b), nz(c)

def _odds_to_probs(odds_home: float | None, odds_draw: float | None, odds_away: float | None):
    ih = _inv_safe(odds_home)
    idr = _inv_safe(odds_draw)
    ia  = _inv_safe(odds_away)
    return _normalize_triple(ih, idr, ia)

def _odds_to_probs_over25(odds_over25: float | None, odds_under25: float | None):
    io = _inv_safe(odds_over25)
    iu = _inv_safe(odds_under25)
    if io is None and iu is None:
        return None, None
    s = (io or 0.0) + (iu or 0.0)
    if s <= 0:
        return None, None
    return (io or 0.0)/s, (iu or 0.0)/s


@app.post("/admin/refresh", tags=["admin"])
def admin_refresh(
    payload: Dict[str, Any] = Body(...),
    _ = Depends(require_admin),   # ne pas appeler require_admin()
):
    """
    Reçoit des matchs + cotes, calcule des probabilités baseline et upsert en DB.

    Body attendu :
    {
      "date": "YYYY-MM-DD",
      "league_id": "L1",
      "matches": [
        {
          "home": "PSG", "away": "OM", "status": "scheduled",
          "odds": { "home": 1.75, "draw": 3.80, "away": 4.50, "over25": 1.85, "under25": 1.95 },
          "version": "baseline-v1"
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
            # ligue présente
            cur.execute("""
                INSERT INTO leagues (id, name)
                VALUES (%s, %s)
                ON CONFLICT (id) DO NOTHING;
            """, (league_id, league_id))

            for m in matches:
                home = m["home"]; away = m["away"]
                status  = m.get("status", "scheduled")
                version = m.get("version", "baseline-v1")

                # 1) upsert match (uq_match)
                cur.execute("""
                    INSERT INTO matches (league_id, match_date, home_team, away_team, status)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT ON CONSTRAINT uq_match DO UPDATE
                    SET status = EXCLUDED.status
                    RETURNING id;
                """, (league_id, d, home, away, status))
                row = cur.fetchone()
                if row:
                    match_id = row[0] if not isinstance(row, dict) else row.get("id")
                else:
                    cur.execute("""
                        SELECT id FROM matches
                        WHERE league_id=%s AND match_date=%s AND home_team=%s AND away_team=%s
                        LIMIT 1;
                    """, (league_id, d, home, away))
                    row = cur.fetchone()
                    match_id = row[0] if row and not isinstance(row, dict) else (row.get("id") if row else None)

                # 2) cotes -> probabilités
                odds = (m.get("odds") or {})
                p_home, p_draw, p_away = _odds_to_probs(
                    odds.get("home"), odds.get("draw"), odds.get("away"))
                p_over25, p_under25 = _odds_to_probs_over25(
                    odds.get("over25"), odds.get("under25"))

                # 3) upsert prediction (uq_pred: match_id, version)
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
                """, (match_id, version, p_home, p_draw, p_away, p_over25, p_under25))

                inserted += 1
    finally:
        conn.close()

    return {"ok": True, "count": inserted, "date": d, "league": league_id}

