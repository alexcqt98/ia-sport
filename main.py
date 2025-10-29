from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import datetime

app = FastAPI(title="NovaPredict API (beta)")

# CORS permissif temporaire (on restreindra après)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.datetime.utcnow().isoformat()}

@app.get("/")
def root():
    # JSON d'accueil (décommente si tu préfères) :
    # return {"service": "NovaPredict API", "status": "live", "see": ["/health", "/docs"]}
    return RedirectResponse(url="/docs")
