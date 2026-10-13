import io, sqlite3
import numpy as np, pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest

DB = "power.db"
FEATURES = ["voltage", "current", "frequency", "power"]
app = FastAPI(title="Power Consumption Anomaly Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS readings(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, voltage REAL, current REAL,
        frequency REAL, power REAL, is_anomaly INTEGER DEFAULT 0, score REAL DEFAULT 0)""")
    return c

def save(df):
    c = conn(); c.execute("DELETE FROM readings")
    df[["ts"] + FEATURES].to_sql("readings", c, if_exists="append", index=False)
    c.commit(); c.close()

@app.post("/generate")
def generate(n: int = 2000):
    """Synthetic SCADA/PMU-style data: daily load cycle + noise + injected faults."""
    rng = np.random.default_rng(42)
    ts = pd.date_range("2025-01-01", periods=n, freq="5min")
    h = np.asarray(ts.hour + ts.minute / 60)
    load = 50 + 20 * np.sin((h - 6) / 24 * 2 * np.pi)           # MW
    voltage = 230 + rng.normal(0, 1.5, n)                        # V
    freq = 50 + rng.normal(0, 0.02, n)                           # Hz
    power = load + rng.normal(0, 1.5, n)
    current = power * 1000 / (voltage * 1.732 * 0.95) + rng.normal(0, 0.5, n)
    for i in rng.choice(n, int(n * 0.02), replace=False):        # inject anomalies
        kind = rng.integers(3)
        if kind == 0: power[i] *= 1.8                            # load spike
        elif kind == 1: voltage[i] -= 25                         # voltage sag
        else: freq[i] += rng.choice([-0.6, 0.6])                 # frequency deviation
    save(pd.DataFrame(dict(ts=ts.astype(str), voltage=voltage, current=current, frequency=freq, power=power)))
    return {"inserted": n}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """CSV with columns: ts,voltage,current,frequency,power"""
    df = pd.read_csv(io.BytesIO(await file.read()))
    save(df)
    return {"inserted": len(df)}

@app.post("/analyze")
def analyze(contamination: float = 0.02):
    c = conn()
    df = pd.read_sql("SELECT * FROM readings", c)
    if df.empty: return {"error": "no data"}
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    X = df[FEATURES]
    model.fit(X)
    df["is_anomaly"] = (model.predict(X) == -1).astype(int)     # -1 = anomaly
    df["score"] = -model.score_samples(X)                        # higher = more anomalous
    c.executemany("UPDATE readings SET is_anomaly=?, score=? WHERE id=?",
                  list(zip(df.is_anomaly.tolist(), df.score.tolist(), df.id.tolist())))
    c.commit(); c.close()
    return {"total": len(df), "anomalies": int(df.is_anomaly.sum())}

@app.get("/readings")
def readings(limit: int = 1000):
    c = conn()
    df = pd.read_sql("SELECT * FROM readings ORDER BY id LIMIT ?", c, params=(limit,))
    return df.to_dict("records")

@app.get("/summary")
def summary():
    c = conn()
    r = c.execute("SELECT COUNT(*), COALESCE(SUM(is_anomaly),0), AVG(power) FROM readings").fetchone()
    return {"total": r[0], "anomalies": r[1], "avg_power": r[2]}
