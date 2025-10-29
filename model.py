# model.py
"""
Model training pour NovaPredict
- Lit les features, résultats et cotes depuis la base.
- Entraîne et calibre des modèles (1N2 + Over/Under 2.5).
- Écrit les probabilités dans la table predictions.
- Enregistre les métriques (Brier, log-loss) dans model_metrics.
"""

import os, datetime, numpy as np, pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

DATABASE_URL = os.getenv("DATABASE_URL", "")

def db_conn():
    return psycopg2.connect(DATABASE_URL)

def load_dataset():
    # Récupère features JSON + résultats + odds
    with db_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            select f.match_id, f.payload as feats, r.home_goals, r.away_goals,
                   o.o_home, o.o_draw, o.o_away,
                   o.o_over25, o.o_under25
            from features f
            join results r on r.match_id = f.match_id
            join odds_prematch o on o.match_id = f.match_id
        """)
        rows = cur.fetchall()
    if not rows:
        return None
    # Convert JSON payload to dataframe
    df_feats = pd.json_normalize([row['feats'] for row in rows])
    df = pd.DataFrame(rows)
    X = pd.concat([df_feats], axis=1)
    # labels
    y_match = df.apply(lambda row: 0 if row['home_goals']>row['away_goals'] else (1 if row['home_goals']==row['away_goals'] else 2), axis=1)
    y_over = df.apply(lambda row: 1 if row['home_goals'] + row['away_goals'] > 2 else 0, axis=1)
    # probabilités implicites du marché
    base_odds = df[['o_home','o_draw','o_away']].astype(float)
    inv = 1.0 / base_odds
    base_match_probs = (inv.T / inv.sum(axis=1)).T
    base_over_prob = 1.0 / df['o_over25'].astype(float) / (1.0/df['o_over25'].astype(float) + 1.0/df['o_under25'].astype(float))
    return X, y_match, y_over, base_match_probs, base_over_prob, df

def train_and_save():
    ds = load_dataset()
    if ds is None:
        print("Pas assez de données pour entraîner")
        return
    X, y_match, y_over, base_match_probs, base_over_prob, df = ds
    if len(X) < 10:
        print("Pas assez de lignes pour entraîner")
        return
    # Modèle logistique pour 1N2
    lr_match = LogisticRegression(max_iter=1000, multi_class='multinomial')
    clf_match = CalibratedClassifierCV(lr_match, method='isotonic', cv=3)
    clf_match.fit(X, y_match)
    probs_match = clf_match.predict_proba(X)
    # Modèle logistique pour Over/Under
    lr_over = LogisticRegression(max_iter=1000)
    clf_over = CalibratedClassifierCV(lr_over, method='isotonic', cv=3)
    clf_over.fit(X, y_over)
    probs_over = clf_over.predict_proba(X)[:,1]
    # Métriques
    # Brier multi-classe pour 1N2
    brier_match_model = np.mean(np.sum((probs_match - pd.get_dummies(y_match).values)**2, axis=1))
    brier_match_base = np.mean(np.sum((base_match_probs.values - pd.get_dummies(y_match).values)**2, axis=1))
    logloss_match_model = log_loss(y_match, probs_match)
    logloss_match_base = log_loss(y_match, base_match_probs.values)
    brier_over_model = brier_score_loss(y_over, probs_over)
    brier_over_base = brier_score_loss(y_over, base_over_prob)
    logloss_over_model = log_loss(y_over, np.column_stack((1-probs_over, probs_over)))
    logloss_over_base = log_loss(y_over, np.column_stack((1-base_over_prob, base_over_prob)))
    # Enregistre les métriques
    version = "v0.1"
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into model_metrics (version, league_id, time_window, brier_1x2, logloss_1x2, brier_ou, logloss_ou, created_at)
            values (%s,%s,%s,%s,%s,%s,%s, now())
            """,
            (version, 'ALL', 'all_time', brier_match_model, logloss_match_model, brier_over_model, logloss_over_model)
        )
        conn.commit()
    # Enregistre les prédictions
    with db_conn() as conn, conn.cursor() as cur:
        for i, row in df.iterrows():
            match_id = row['match_id']
            ph, pd_, pa = probs_match[i]
            pov = probs_over[i]
            puv = 1 - pov
            cur.execute(
                """
                insert into predictions (match_id, version, p_home, p_draw, p_away, p_over25, p_under25, created_at)
                values (%s,%s,%s,%s,%s,%s,%s, now())
                on conflict (match_id) do update set version = EXCLUDED.version,
                    p_home = EXCLUDED.p_home,
                    p_draw = EXCLUDED.p_draw,
                    p_away = EXCLUDED.p_away,
                    p_over25 = EXCLUDED.p_over25,
                    p_under25 = EXCLUDED.p_under25,
                    created_at = EXCLUDED.created_at
                """,
                (int(match_id), version, float(ph), float(pd_), float(pa), float(pov), float(puv))
            )
        conn.commit()
    print("Modèle entraîné et prédictions mises à jour")

if __name__ == "__main__":
    train_and_save()
