from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.analytics import demo_summary
from app.core.config import settings
from app.db.database import Base, engine, get_db
from app.services import create_prediction_record, get_prediction_stats, get_recent_records, run_prediction, save_upload


app = FastAPI(title=settings.app_name)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.app_name})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.get("/api/dashboard")
def dashboard_payload(db: Session = Depends(get_db)) -> dict:
    payload = demo_summary()
    payload["prediction_summary"] = get_prediction_stats(db)
    payload["recent_predictions"] = get_recent_records(db)
    return payload


@app.get("/api/predictions")
def predictions(db: Session = Depends(get_db)) -> dict:
    return {
        "summary": get_prediction_stats(db),
        "records": get_recent_records(db),
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A histopathology tissue image file is required.")

    stored_path = save_upload(file)

    try:
        result = run_prediction(stored_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    payload = {"filename": stored_path.name, **result}
    record = create_prediction_record(db, stored_path.name, payload)
    return record
