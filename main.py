import os, datetime, math
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Query, HTTPException, Header, Body, Depends
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# ========================
# App & CORS
# ========================
app = FastAPI(title="NovaPredict API (beta)")

allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed if allowed else ["*"],
    allow_origin_regex=origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# ENV
# ========================
DATABASE_URL = os.getenv("DATABASE_URL", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# ========================
# DB helper
# ========================
def db_conn():
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL absente (configure sur Render).")
    import psycopg2
    from psycopg2.extras import RealDictCursor
    return psycopg2.connect(DATABASE_URL), RealDictCursor
    def _ensure_team(cur, alias: str) -> int:
    """
    Mappe un alias d’équipe -> team_id (crée si inconnu).
    """
    cur.execute("select team_id from team_aliases where alias=%s", (alias,))
    row = cur.fetchone()
    if row:
        return row["team_id"] if isinstance(row, dict) else row[0]

    # crée une team avec ce nom (simple au départ)
    cur.execute("insert into teams(name) values(%s) returning id", (alias,))
    team_id = cur.fetchone()[0]
    cur.execute(
        "insert into team_aliases(alias, team_id) values(%s,%s) on conflict do nothing",
        (alias, team_id)
    )
    return team_id

def _resolve_team_name(cur, alias: str) -> str:
    """
    Retourne le nom canonique (teams.name) à partir d’un alias, sinon l’alias.
    """
    cur.execute("""
        select t.name from team_aliases a
        join teams t on t.id=a.team_id
        where a.alias=%s
    """, (alias,))
    row = cur.fetchone()
    if not row:
        return alias
    return row["name"] if isinstance(row, dict) else row[0]


# ========================
# Schemas publics
# ========================
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

# ========================
# Health & root
# ========================
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.datetime.utcnow().isoformat()}

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# ========================
# Public endpoints
# ========================
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

# ========================
# Seed simple (optionnel)
# ========================
@app.post("/seed")
def seed():
    conn, RealDictCursor = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("insert into leagues(id,name) values('L1','Ligue 1') on conflict(id) do nothing;")
            match_date = datetime.date.today().isoformat()

            cur.execute("""
                insert into matches(league_id, match_date, home_team, away_team, status)
                values('L1', %s, 'PSG', 'OM', 'scheduled')
                on conflict on constraint uq_match do nothing
                returning id;
            """, (match_date,))
            row = cur.fetchone()
            if row: match_id = row[0]
            else:
                cur.execute("""
                    select id from matches
                    where league_id='L1' and match_date=%s and home_team='PSG' and away_team='OM' limit 1;
                """, (match_date,))
                match_id = cur.fetchone()[0]

            cur.execute("""
                insert into predictions(match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
                values(%s,'v0.1',0.58,0.24,0.18,0.62,0.38)
                on conflict on constraint uq_pred do update set
                    p_home=excluded.p_home,
                    p_draw=excluded.p_draw,
                    p_away=excluded.p_away,
                    p_over25=excluded.p_over25,
                    p_under25=excluded.p_under25,
                    version=excluded.version;
            """, (match_id,))
        conn.commit()
        return {"status": "seeded", "match_id": match_id}
    finally:
        conn.close()

# =========================
# ADMIN security
# =========================
def require_admin(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# =========================
# Helpers upsert match/pred
# =========================
def upsert_match(cur, league_id: str, d: str, home: str, away: str, status: str):
    cur.execute("""
        insert into matches(league_id, match_date, home_team, away_team, status)
        values (%s, %s, %s, %s, %s)
        on conflict on constraint uq_match do update
        set status = excluded.status
        returning id;
    """, (league_id, d, home, away, status))
    row = cur.fetchone()
    if not row:
        cur.execute("""
            select id from matches
            where league_id=%s and match_date=%s and home_team=%s and away_team=%s limit 1;
        """, (league_id, d, home, away))
        row = cur.fetchone()
    return row["id"] if isinstance(row, dict) else row[0]

def upsert_prediction(cur, match_id: int, probs: Dict[str, Any]):
    version = probs.get("version", "v0.1")
    cur.execute("""
        insert into predictions(match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict on constraint uq_pred do update set
            p_home=excluded.p_home,
            p_draw=excluded.p_draw,
            p_away=excluded.p_away,
            p_over25=excluded.p_over25,
            p_under25=excluded.p_under25,
            version=excluded.version;
    """, (
        match_id, version,
        probs.get("p_home"), probs.get("p_draw"), probs.get("p_away"),
        probs.get("p_over25"), probs.get("p_under25"),
    ))

# =========================
# /admin/upsert (manuel)
# =========================
@app.post("/admin/upsert", tags=["admin"])
def admin_upsert(payload: Dict[str, Any] = Body(...), _ = Depends(require_admin)):
    d = payload["date"]
    league_id = payload.get("league_id", "L1")
    matches = payload.get("matches", [])
    inserted = 0

    conn, RealDictCursor = db_conn()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("insert into leagues(id,name) values(%s,%s) on conflict(id) do nothing;", (league_id, league_id))
            for m in matches:
                mid = upsert_match(cur, league_id, d, m["home"], m["away"], m.get("status","scheduled"))
                if "prediction" in m:
                    upsert_prediction(cur, mid, m["prediction"])
                inserted += 1
    finally:
        conn.close()
    return {"ok": True, "count": inserted, "date": d, "league": league_id}

# =========================
# Baseline (odds -> probs)
# =========================
def _inv_safe(x: float | None) -> float | None:
    if x is None: return None
    try:
        x = float(x)
        if x <= 0: return None
        return 1.0 / x
    except Exception:
        return None

def _normalize_triple(a: float | None, b: float | None, c: float | None):
    vals = [v for v in (a,b,c) if v is not None]
    if not vals: return None, None, None
    s = sum(vals)
    if s <= 0: return None, None, None
    def nz(v): return (v/s) if v is not None else None
    return nz(a), nz(b), nz(c)

def _odds_to_probs(oh, od, oa):
    return _normalize_triple(_inv_safe(oh), _inv_safe(od), _inv_safe(oa))

def _odds_to_probs_over25(o, u):
    io, iu = _inv_safe(o), _inv_safe(u)
    if io is None and iu is None: return None, None
    s = (io or 0.0) + (iu or 0.0)
    if s <= 0: return None, None
    return (io or 0.0)/s, (iu or 0.0)/s

@app.post("/admin/refresh", tags=["admin"])
def admin_refresh(payload: Dict[str, Any] = Body(...), _ = Depends(require_admin)):
    d = payload["date"]
    league_id = payload.get("league_id", "L1")
    matches = payload.get("matches", [])
    inserted = 0

    conn, RealDictCursor = db_conn()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("insert into leagues(id,name) values(%s,%s) on conflict(id) do nothing;", (league_id, league_id))
            for m in matches:
                home, away = m["home"], m["away"]
                status  = m.get("status", "scheduled")
                version = m.get("version", "baseline-v1")

                cur.execute("""
                    insert into matches(league_id, match_date, home_team, away_team, status)
                    values (%s,%s,%s,%s,%s)
                    on conflict on constraint uq_match do update set status=excluded.status
                    returning id;
                """, (league_id, d, home, away, status))
                row = cur.fetchone()
                if row:
                    match_id = row[0] if not isinstance(row, dict) else row.get("id")
                else:
                    cur.execute("""
                        select id from matches where league_id=%s and match_date=%s and home_team=%s and away_team=%s limit 1;
                    """, (league_id, d, home, away))
                    row = cur.fetchone()
                    match_id = row[0] if row and not isinstance(row, dict) else (row.get("id") if row else None)

                odds = (m.get("odds") or {})
                p_home, p_draw, p_away = _odds_to_probs(odds.get("home"), odds.get("draw"), odds.get("away"))
                p_over25, p_under25 = _odds_to_probs_over25(odds.get("over25"), odds.get("under25"))

                cur.execute("""
                    insert into predictions(match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
                    values (%s,%s,%s,%s,%s,%s,%s)
                    on conflict on constraint uq_pred do update set
                        p_home=excluded.p_home,
                        p_draw=excluded.p_draw,
                        p_away=excluded.p_away,
                        p_over25=excluded.p_over25,
                        p_under25=excluded.p_under25,
                        version=excluded.version;
                """, (match_id, version, p_home, p_draw, p_away, p_over25, p_under25))
                inserted += 1
    finally:
        conn.close()

    return {"ok": True, "count": inserted, "date": d, "league": league_id}
    @app.post("/admin/ingest/odds", tags=["admin"])
def admin_ingest_odds(payload: Dict[str, Any] = Body(...), _=Depends(require_admin)):
    """
    Stocke des cotes + crée/MAJ les matches + snapshotte les cotes.
    Body:
    {
      "date":"YYYY-MM-DD", "league_id":"L1", "book":"BookName",
      "matches":[
        {"home":"PSG","away":"OM",
         "odds":{"home":1.70,"draw":3.90,"away":5.00,"over25":1.85,"under25":1.95},
         "ext":{"provider":"oddsapi","id":"12345"}}
      ]
    }
    """
    d = payload["date"]
    league_id = payload.get("league_id", "L1")
    book = payload.get("book")
    matches = payload.get("matches", [])
    if not matches:
        raise HTTPException(400, "matches vide")

    conn, RealDictCursor = db_conn()
    inserted = 0
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("insert into leagues(id,name) values(%s,%s) on conflict do nothing", (league_id, league_id))
            for m in matches:
                home_alias, away_alias = m["home"], m["away"]
                _ensure_team(cur, home_alias)
                _ensure_team(cur, away_alias)

                # upsert match (scheduled par défaut)
                cur.execute("""
                    insert into matches(league_id, match_date, home_team, away_team, status)
                    values (%s,%s,%s,%s,'scheduled')
                    on conflict on constraint uq_match do update set status=excluded.status
                    returning id
                """, (league_id, d, home_alias, away_alias))
                row = cur.fetchone()
                match_id = row["id"] if isinstance(row, dict) else row[0]

                # Optionnel: lier un ID externe
                ext = m.get("ext")
                if ext and ext.get("provider") and ext.get("id"):
                    cur.execute("""
                        insert into match_external_ids(match_id, provider, ext_id)
                        values (%s,%s,%s)
                        on conflict (match_id) do update set provider=excluded.provider, ext_id=excluded.ext_id
                    """, (match_id, ext["provider"], ext["id"]))

                # Snapshot des cotes
                odds = m.get("odds") or {}
                cur.execute("""
                    insert into odds_snapshots(match_id, book, home_odds, draw_odds, away_odds, over25_odds, under25_odds)
                    values (%s,%s,%s,%s,%s,%s,%s)
                """, (match_id, book, odds.get("home"), odds.get("draw"), odds.get("away"),
                      odds.get("over25"), odds.get("under25")))
                inserted += 1
    finally:
        conn.close()
    return {"ok": True, "snapshots": inserted}
@app.post("/admin/ingest/xg", tags=["admin"])
def admin_ingest_xg(payload: Dict[str, Any] = Body(...), _=Depends(require_admin)):
    """
    Ajoute/MAJ les xG par match (utilisé pour features).
    Body:
    {
      "date":"YYYY-MM-DD","league_id":"L1",
      "matches":[{"home":"PSG","away":"OM","xg_home":2.1,"xg_away":0.8}]
    }
    """
    d = payload["date"]
    league_id = payload.get("league_id", "L1")
    matches = payload.get("matches", [])
    if not matches:
        raise HTTPException(400, "matches vide")

    conn, RealDictCursor = db_conn()
    up = 0
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            for m in matches:
                home, away = m["home"], m["away"]

                # retrouve (ou crée) le match de cette date
                cur.execute("""
                    select id from matches
                    where league_id=%s and match_date=%s and home_team=%s and away_team=%s
                    limit 1
                """, (league_id, d, home, away))
                row = cur.fetchone()
                if not row:
                    cur.execute("""
                        insert into matches(league_id, match_date, home_team, away_team, status)
                        values (%s,%s,%s,%s,'finished')
                        on conflict on constraint uq_match do nothing
                        returning id
                    """, (league_id, d, home, away))
                    row = cur.fetchone()
                    if not row:
                        cur.execute("""
                            select id from matches
                            where league_id=%s and match_date=%s and home_team=%s and away_team=%s
                            limit 1
                        """, (league_id, d, home, away))
                        row = cur.fetchone()

                match_id = row["id"] if isinstance(row, dict) else row[0]

                cur.execute("""
                    insert into match_xg(match_id, xg_home, xg_away, provider)
                    values (%s,%s,%s,'manual')
                    on conflict (match_id) do update set
                      xg_home=excluded.xg_home,
                      xg_away=excluded.xg_away,
                      updated_at=now()
                """, (match_id, m.get("xg_home"), m.get("xg_away")))
                up += 1
    finally:
        conn.close()

    return {"ok": True, "xg_upserted": up}


# =========================
# Elo: params & helpers
# =========================
ELO_K = float(os.getenv("ELO_K", "20"))
ELO_HOME_ADV = float(os.getenv("ELO_HOME_ADV", "60"))  # avantage maison en points

def _get_team_rating(cur, team: str) -> float:
    cur.execute("select rating from elo_ratings where team=%s;", (team,))
    row = cur.fetchone()
    if row: return row["rating"] if isinstance(row, dict) else row[0]
    cur.execute("insert into elo_ratings(team, rating) values(%s,1500) on conflict(team) do nothing;", (team,))
    return 1500.0

def _set_team_rating(cur, team: str, rating: float):
    cur.execute("""
        insert into elo_ratings(team, rating) values(%s,%s)
        on conflict(team) do update set rating=excluded.rating, updated_at=now();
    """, (team, rating))

def _elo_expected(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(a - b) / 400.0))

def _elo_update_pair(cur, home: str, away: str, gh: int, ga: int):
    Rh = _get_team_rating(cur, home)
    Ra = _get_team_rating(cur, away)
    Rh_eff = Rh + ELO_HOME_ADV
    Ra_eff = Ra

    if gh > ga: Sh, Sa = 1.0, 0.0
    elif gh < ga: Sh, Sa = 0.0, 1.0
    else: Sh, Sa = 0.5, 0.5

    Eh = _elo_expected(Rh_eff, Ra_eff)
    Ea = 1.0 - Eh

    _set_team_rating(cur, home, Rh + ELO_K * (Sh - Eh))
    _set_team_rating(cur, away, Ra + ELO_K * (Sa - Ea))

def _elo_predict_probs(cur, home: str, away: str):
    Rh = _get_team_rating(cur, home) + ELO_HOME_ADV
    Ra = _get_team_rating(cur, away)
    p_home = _elo_expected(Rh, Ra)
    p_away = _elo_expected(Ra, Rh)
    gap = abs((Rh - ELO_HOME_ADV) - Ra)
    p_draw = 0.24 + max(0.0, 0.12 - (gap / 800.0))
    s = p_home + p_away + p_draw
    if s > 0:
        p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s
    over_bias = min(0.15, gap / 800.0)
    p_over25 = 0.50 + over_bias
    p_under25 = 1.0 - p_over25
    return p_home, p_draw, p_away, p_over25, p_under25

# =========================
# /admin/record_results
# =========================
@app.post("/admin/record_results", tags=["admin"])
def admin_record_results(payload: Dict[str, Any] = Body(...), _ = Depends(require_admin)):
    results = payload.get("results", [])
    if not results:
        raise HTTPException(400, "results vide")

    conn, RealDictCursor = db_conn()
    updated = 0
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            for r in results:
                d = r["date"]; league_id = r.get("league_id", "L1")
                home = r["home"]; away = r["away"]
                gh = int(r["goals_home"]); ga = int(r["goals_away"])

                cur.execute("insert into leagues(id,name) values(%s,%s) on conflict(id) do nothing;", (league_id, league_id))

                cur.execute("""
                    insert into matches(league_id, match_date, home_team, away_team, status, goals_home, goals_away)
                    values(%s,%s,%s,%s,'finished',%s,%s)
                    on conflict on constraint uq_match do update
                    set status='finished', goals_home=excluded.goals_home, goals_away=excluded.goals_away
                    returning id;
                """, (league_id, d, home, away, gh, ga))
                row = cur.fetchone()
                if not row:
                    cur.execute("""
                        select id from matches
                        where league_id=%s and match_date=%s and home_team=%s and away_team=%s limit 1;
                    """, (league_id, d, home, away))
                    row = cur.fetchone()

                _elo_update_pair(cur, home, away, gh, ga)
                updated += 1
    finally:
        conn.close()

    return {"ok": True, "updated": updated}

# =========================
# /admin/predict_elo
# =========================
@app.post("/admin/predict_elo", tags=["admin"])
def admin_predict_elo(payload: Dict[str, Any] = Body(...), _ = Depends(require_admin)):
    d = payload["date"]
    league_id = payload.get("league_id", "L1")
    version = payload.get("version", "elo-v1")
    matches = payload.get("matches", [])
    if not matches:
        raise HTTPException(400, "matches vide")

    conn, RealDictCursor = db_conn()
    inserted = 0
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("insert into leagues(id,name) values(%s,%s) on conflict(id) do nothing;", (league_id, league_id))
            for m in matches:
                home, away = m["home"], m["away"]
                status = m.get("status","scheduled")

                cur.execute("""
                    insert into matches(league_id, match_date, home_team, away_team, status)
                    values(%s,%s,%s,%s,%s)
                    on conflict on constraint uq_match do update set status=excluded.status
                    returning id;
                """, (league_id, d, home, away, status))
                row = cur.fetchone()
                if not row:
                    cur.execute("""
                        select id from matches where league_id=%s and match_date=%s and home_team=%s and away_team=%s limit 1;
                    """, (league_id, d, home, away))
                    row = cur.fetchone()
                match_id = row["id"] if isinstance(row, dict) else row[0]

                p_home, p_draw, p_away, p_over25, p_under25 = _elo_predict_probs(cur, home, away)

                cur.execute("""
                    insert into predictions(match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
                    values (%s,%s,%s,%s,%s,%s,%s)
                    on conflict on constraint uq_pred do update set
                      p_home=excluded.p_home,
                      p_draw=excluded.p_draw,
                      p_away=excluded.p_away,
                      p_over25=excluded.p_over25,
                      p_under25=excluded.p_under25,
                      version=excluded.version;
                """, (match_id, version, p_home, p_draw, p_away, p_over25, p_under25))
                inserted += 1
    finally:
        conn.close()

    return {"ok": True, "count": inserted, "date": d, "league": league_id, "version": version}

# =========================
# Features pour ML
# =========================
def _team_last_results(cur, team: str, upto_date: str, limit: int = 10) -> list[dict]:
    cur.execute("""
        select match_date, home_team, away_team, goals_home, goals_away
        from matches
        where status='finished'
          and match_date < %s
          and (home_team=%s or away_team=%s)
        order by match_date desc
        limit %s;
    """, (upto_date, team, team, limit))
    rows = cur.fetchall()
    return rows[::-1]

def _points_from_row(row: dict, team: str) -> int:
    gh, ga = row["goals_home"], row["goals_away"]
    if row["home_team"] == team:
        if gh > ga: return 3
        if gh == ga: return 1
        return 0
    else:
        if ga > gh: return 3
        if ga == gh: return 1
        return 0

def _gfga_from_row(row: dict, team: str):
    gh, ga = row["goals_home"], row["goals_away"]
    return (gh, ga) if row["home_team"] == team else (ga, gh)

@app.post("/admin/build_features", tags=["admin"])
def admin_build_features(payload: Dict[str, Any] = Body(...), _ = Depends(require_admin)):
    """
    Construit les features pour les matchs 'scheduled' d'une date.
    Table utilisée: match_features(match_id, elo_diff, form_diff_5, gfga_diff_10)
    Body: { "date":"YYYY-MM-DD", "league_id":"L1" }
    """
    d = payload["date"]
    league_id = payload.get("league_id")

    conn, RealDictCursor = db_conn()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                select id, home_team, away_team
                from matches
                where match_date=%s and status='scheduled'
                  and (%s is null or league_id=%s)
                order by id;
            """, (d, league_id, league_id))
            matches = cur.fetchall()

            def elo(team: str) -> float:
                cur.execute("select rating from elo_ratings where team=%s;", (team,))
                row = cur.fetchone()
                return float(row["rating"]) if row else 1500.0

            inserted = 0
            for m in matches:
                home, away = m["home_team"], m["away_team"]
                elo_diff = elo(home) - elo(away)

                last5_home = _team_last_results(cur, home, d, limit=5)
                last5_away = _team_last_results(cur, away, d, limit=5)
                form_home = sum(_points_from_row(r, home) for r in last5_home) if last5_home else 0
                form_away = sum(_points_from_row(r, away) for r in last5_away) if last5_away else 0
                form_diff_5 = float(form_home - form_away)

                last10_home = _team_last_results(cur, home, d, limit=10)
                last10_away = _team_last_results(cur, away, d, limit=10)
                gf_h = ga_h = 0
                for r in last10_home:
                    gf, ga = _gfga_from_row(r, home)
                    gf_h += gf; ga_h += ga
                gf_a = ga_a = 0
                for r in last10_away:
                    gf, ga = _gfga_from_row(r, away)
                    gf_a += gf; ga_a += ga
                gfga_diff_10 = float((gf_h - ga_h) - (gf_a - ga_a))

                cur.execute("""
                    insert into match_features(match_id, elo_diff, form_diff_5, gfga_diff_10, updated_at)
                    values(%s,%s,%s,%s,now())
                    on conflict (match_id) do update set
                      elo_diff=excluded.elo_diff,
                      form_diff_5=excluded.form_diff_5,
                      gfga_diff_10=excluded.gfga_diff_10,
                      updated_at=now();
                """, (m["id"], elo_diff, form_diff_5, gfga_diff_10))
                inserted += 1
    finally:
        conn.close()

    return {"ok": True, "built": inserted, "date": d, "league": league_id}

# =========================
# /admin/predict_ml (rule-based)
# =========================
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _softmax3(a: float, b: float, c: float):
    mx = max(a,b,c)
    ea, eb, ec = math.exp(a-mx), math.exp(b-mx), math.exp(c-mx)
    s = ea+eb+ec
    return ea/s, eb/s, ec/s

@app.post("/admin/predict_ml", tags=["admin"])
def admin_predict_ml(payload: Dict[str, Any] = Body(...), _=Depends(require_admin)):
    """
    Produit des proba 'ml-v1' depuis les features (rule-based) et upsert.
    Body: { "date":"YYYY-MM-DD", "league_id":"L1", "version":"ml-v1" }
    """
    d = payload["date"]
    league_id = payload.get("league_id")
    version = payload.get("version", "ml-v1")

    conn, RealDictCursor = db_conn()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                select m.id, m.league_id, m.match_date, m.home_team, m.away_team,
                       f.elo_diff, f.form_diff_5, f.gfga_diff_10
                from matches m
                join match_features f on f.match_id = m.id
                where m.match_date = %s
                  and m.status = 'scheduled'
                  and (%s is null or m.league_id = %s)
                order by m.id;
            """, (d, league_id, league_id))
            rows = cur.fetchall()

            inserted = 0
            for r in rows:
                z_home =  0.0025*(r["elo_diff"] or 0) + 0.08*(r["form_diff_5"] or 0) + 0.02*(r["gfga_diff_10"] or 0)
                z_away = -z_home
                z_draw = -0.5*abs(z_home)

                p_home, p_draw, p_away = _softmax3(z_home, z_draw, z_away)

                mu = 2.4 + 0.02*(r["gfga_diff_10"] or 0)
                p_over25 = _sigmoid(1.2*(mu - 2.5))
                p_under25 = 1.0 - p_over25

                cur.execute("""
                    insert into predictions (match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
                    values (%s,%s,%s,%s,%s,%s,%s)
                    on conflict on constraint uq_pred do update set
                      p_home=excluded.p_home,
                      p_draw=excluded.p_draw,
                      p_away=excluded.p_away,
                      p_over25=excluded.p_over25,
                      p_under25=excluded.p_under25,
                      version=excluded.version;
                """, (r["id"], version, p_home, p_draw, p_away, p_over25, p_under25))
                inserted += 1
    finally:
        conn.close()

    return {"ok": True, "count": inserted, "version": version, "date": d, "league": league_id}
   
def _safe_log(x: float) -> float:
    x = max(1e-12, min(1.0-1e-12, x if x is not None else 1e-12))
    return math.log(x)

def _brier(pw, pd, pl, w, d, l):
    # Brier multi-classes : sum (p_i - y_i)^2
    return (pw - w)**2 + (pd - d)**2 + (pl - l)**2

@app.post("/admin/recompute_metrics", tags=["admin"])
def admin_recompute_metrics(payload: Dict[str, Any] = Body(...), _=Depends(require_admin)):
    """
    Recalcule Brier & LogLoss par (mois, version) à partir des matchs 'finished'
    et des prédictions (toutes versions), puis upsert dans model_metrics.
    Body: { "from": "2025-10-01", "to": "2025-10-31" }
    """
    import math
    from decimal import Decimal

    def to_float(x):
        if x is None: return 0.0
        if isinstance(x, Decimal): return float(x)
        return float(x)

    d_from = payload.get("from") or "2025-01-01"
    d_to   = payload.get("to")   or datetime.date.today().isoformat()

    # Prépare l'agg ici (sinon risque de NameError dans le return)
    agg: Dict[tuple, Dict[str, float]] = {}

    conn, RealDictCursor = db_conn()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Récupère matches terminés + prédictions (toutes versions) sur la période
            cur.execute("""
                SELECT
                  date_trunc('month', m.match_date)::date AS bucket_month,
                  p.version,
                  m.goals_home, m.goals_away,
                  p.p_home, p.p_draw, p.p_away
                FROM matches m
                JOIN predictions p ON p.match_id = m.id
                WHERE m.status = 'finished'
                  AND m.goals_home IS NOT NULL
                  AND m.goals_away IS NOT NULL
                  AND m.match_date BETWEEN %s AND %s
                ORDER BY bucket_month, p.version;
            """, (d_from, d_to))
            rows = cur.fetchall()

            if not rows:
                # Rien à recalculer mais on renvoie ok
                return {"ok": True, "from": d_from, "to": d_to, "buckets": 0, "note": "aucun match fini dans l'intervalle"}

            # Agrégation en Python
            for r in rows:
                bm = r["bucket_month"]
                version = (r["version"] or "unknown")

                gh = int(r["goals_home"]); ga = int(r["goals_away"])
                if   gh > ga: y = (1.0, 0.0, 0.0); p_true = r["p_home"]
                elif gh == ga: y = (0.0, 1.0, 0.0); p_true = r["p_draw"]
                else:          y = (0.0, 0.0, 1.0); p_true = r["p_away"]

                ph = to_float(r["p_home"])
                pd = to_float(r["p_draw"])
                pa = to_float(r["p_away"])

                # Brier multi-classes
                brier = ((ph - y[0])**2 + (pd - y[1])**2 + (pa - y[2])**2) / 3.0

                # LogLoss = -log(p_true) (clip pour éviter -inf)
                eps = 1e-15
                pt = max(min(to_float(p_true), 1.0 - eps), eps)
                logloss = -math.log(pt)

                key = (bm, version)
                if key not in agg:
                    agg[key] = {"sum_brier": 0.0, "sum_ll": 0.0, "n": 0}
                agg[key]["sum_brier"] += brier
                agg[key]["sum_ll"]    += logloss
                agg[key]["n"]         += 1

            # Upsert dans model_metrics
            for (bm, v), s in agg.items():
                if s["n"] == 0: 
                    continue
                brier = s["sum_brier"] / s["n"]
                logloss = s["sum_ll"] / s["n"]
                cur.execute("""
                    INSERT INTO model_metrics(bucket_month, version, brier, logloss, n)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (bucket_month, version) DO UPDATE SET
                      brier = EXCLUDED.brier,
                      logloss = EXCLUDED.logloss,
                      n = EXCLUDED.n,
                      created_at = now();
                """, (bm, v, brier, logloss, s["n"]))
    finally:
        conn.close()

    return {"ok": True, "from": d_from, "to": d_to, "buckets": len(agg)}


@app.get("/metrics/monthly")
def metrics_monthly():
    """
    Agrège Brier Score et LogLoss par (mois, version) à partir des matches 'finished'.
    Renvoie: [{month:'YYYY-MM-01', version:'elo-v1', n:12, brier:..., logloss:...}, ...]
    """
    import math
    from decimal import Decimal

    def to_float(x):
        if x is None:
            return 0.0
        if isinstance(x, Decimal):
            return float(x)
        return float(x)

    conn, RealDictCursor = db_conn()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT date_trunc('month', m.match_date)::date AS month,
                       p.version,
                       m.goals_home, m.goals_away,
                       p.p_home, p.p_draw, p.p_away
                FROM matches m
                JOIN predictions p ON p.match_id = m.id
                WHERE m.goals_home IS NOT NULL
                  AND m.goals_away IS NOT NULL
                  AND m.status = 'finished'
                ORDER BY month ASC;
            """)
            rows = cur.fetchall()

        if not rows:
            return {"items": []}

        agg = {}
        for r in rows:
            month = r["month"].isoformat() if hasattr(r["month"], "isoformat") else str(r["month"])
            version = (r["version"] or "unknown")

            gh = int(r["goals_home"]); ga = int(r["goals_away"])
            if   gh > ga: y = (1.0, 0.0, 0.0); p_true = r["p_home"]
            elif gh == ga: y = (0.0, 1.0, 0.0); p_true = r["p_draw"]
            else:          y = (0.0, 0.0, 1.0); p_true = r["p_away"]

            ph = to_float(r["p_home"])
            pd = to_float(r["p_draw"])
            pa = to_float(r["p_away"])

            # Brier multi-classes
            brier = ((ph - y[0])**2 + (pd - y[1])**2 + (pa - y[2])**2) / 3.0

            # LogLoss = -log(p_true clip)
            eps = 1e-15
            p_true_val = to_float(p_true)
            p_true_clip = max(min(p_true_val, 1.0 - eps), eps)
            logloss = -math.log(p_true_clip)

            key = (month, version)
            if key not in agg:
                agg[key] = {"n": 0, "s_brier": 0.0, "s_logloss": 0.0}
            agg[key]["n"]       += 1
            agg[key]["s_brier"] += brier
            agg[key]["s_logloss"] += logloss

        items = []
        for (month, version), v in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
            n = v["n"]
            items.append({
                "month": month,
                "version": version,
                "n": n,
                "brier": v["s_brier"] / n,
                "logloss": v["s_logloss"] / n
            })
        return {"items": items}
    finally:
        conn.close()


@app.post("/admin/backfill_predictions", tags=["admin"])
def admin_backfill_predictions(payload: Dict[str, Any] = Body(...), _=Depends(require_admin)):
    """
    Crée/MAJ des prédictions 'elo-v1' pour tous les matches FINISHED
    entre 'from' et 'to' (inclus), afin que /metrics/monthly ait de la matière.
    Body:
      { "from": "2025-10-01", "to": "2025-10-31", "version": "elo-v1" }
    """
    d_from = payload.get("from") or "2025-01-01"
    d_to   = payload.get("to")   or datetime.date.today().isoformat()
    version = payload.get("version", "elo-v1")

    conn, RealDictCursor = db_conn()
    created = 0
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Récupère tous les matches terminés sur la période
            cur.execute("""
                SELECT id, league_id, match_date, home_team, away_team
                FROM matches
                WHERE status='finished'
                  AND match_date BETWEEN %s AND %s
                ORDER BY match_date, id;
            """, (d_from, d_to))
            rows = cur.fetchall()

            for m in rows:
                mid   = m["id"]
                home  = m["home_team"]
                away  = m["away_team"]

                # Probas via Elo existant
                p_home, p_draw, p_away, p_over25, p_under25 = _elo_predict_probs(cur, home, away)

                # Upsert prédiction (uq_pred: match_id, version)
                cur.execute("""
                    INSERT INTO predictions(match_id, version, p_home, p_draw, p_away, p_over25, p_under25)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT ON CONSTRAINT uq_pred DO UPDATE SET
                      p_home=EXCLUDED.p_home,
                      p_draw=EXCLUDED.p_draw,
                      p_away=EXCLUDED.p_away,
                      p_over25=EXCLUDED.p_over25,
                      p_under25=EXCLUDED.p_under25,
                      version=EXCLUDED.version;
                """, (mid, version, p_home, p_draw, p_away, p_over25, p_under25))
                created += 1
    finally:
        conn.close()

    return {"ok": True, "created_or_updated": created, "from": d_from, "to": d_to, "version": version}
