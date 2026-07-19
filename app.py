from fastapi import FastAPI

from autoguard_drift.demo import run

app = FastAPI(title="AutoGuard Fleet Drift", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/demo")
def demo() -> dict[str, object]:
    return run()
