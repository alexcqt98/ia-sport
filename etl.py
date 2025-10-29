# etl.py
"""
ETL pour NovaPredict :
- Récupère fixtures + cotes via un fournisseur (API-Football, football-data.org, etc.).
- Insère/actualise : matches, odds_prematch.
- Calcule et insère les features (forme, Elo, etc.).
- Optionnel : déclenche l'entraîment du modèle après mise à jour.
"""

import os, datetime, requests
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "")

def db():
    return psycopg2.connect(DATABASE_URL)

def fetch_fixtures(date_from: str, date_to: str):
    """
    Exemple : récupérer les matchs entre date_from et date_to via API-Football.
    """
    url = f"https://v3.football.api-sports.io/fixtures?from={date_from}&to={date_to}&timezone=Europe/Paris"
    headers = {"X-RapidAPI-Key": FOOTBALL_API_KEY, "X-RapidAPI-Host": "v3.football.api-sports.io"}
    resp = requests.get(url, headers=headers)
    return resp.json().get("response", [])

def upsert_matches(fixtures):
    with db() as conn, conn.cursor() as cur:
        for f in fixtures:
            league_id = f["league"]["id"]
            match_date = f["fixture"]["date"][:10]
            home = f["teams"]["home"]["name"]
            away = f["teams"]["away"]["name"]
            cur.execute("""
                insert into matches(league_id, match_date, home_team, away_team, status)
                values (%s,%s,%s,%s,'scheduled')
                on conflict (league_id, match_date, home_team, away_team) do nothing
            """, (league_id, match_date, home, away))
        conn.commit()

def fetch_odds(match_id):
    """
    Exemple : récupérer les cotes pour un match via API-Football.
    """
    pass  # implémentation à faire selon ton fournisseur

def update_odds():
    """
    Parcourt les matches à venir et insère les cotes dans odds_prematch.
    """
    pass  # implémentation à faire

def compute_features():
    """
    Calcule les features (forme, Elo, etc.) pour chaque match.
    Ecrit un JSON dans la table 'features'.
    """
    pass  # implémentation à faire

def run_daily():
    # Récupère fixtures pour la semaine
    today = datetime.date.today()
    date_from = today.isoformat()
    date_to = (today + datetime.timedelta(days=7)).isoformat()
    fixtures = fetch_fixtures(date_from, date_to)
    upsert_matches(fixtures)
    # Met à jour les cotes
    update_odds()
    # Calcule les features
    compute_features()
    print("ETL daily completed")

if __name__ == "__main__":
    run_daily()
