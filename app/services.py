import json
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import PredictionRecord
from app.ml.inference import inference_service
from app.schemas import PredictionRecordRead
from app.core.config import settings


def save_upload(file: UploadFile) -> Path:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(file.filename or "upload.png").suffix or ".png"
    filename = f"{uuid4().hex}{extension}"
    destination = settings.upload_dir / filename
    with destination.open("wb") as buffer:
        buffer.write(file.file.read())
    return destination


def create_prediction_record(db: Session, stored_filename: str, result: dict) -> dict:
    record = PredictionRecord(
        filename=stored_filename,
        predicted_class=result["predicted_class"],
        predicted_label=result["predicted_label"],
        cancer_percentage=result["cancer_percentage"],
        risk_band=result["risk_band"],
        confidence_score=result["confidence_score"],
        top_probability=result["top_probability"],
        model_source=result["model_source"],
        class_probabilities_json=json.dumps(result["class_probabilities"]),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return serialize_record(record)


def serialize_record(record: PredictionRecord) -> dict:
    return PredictionRecordRead(
        id=record.id,
        filename=record.filename,
        predicted_class=record.predicted_class,
        predicted_label=record.predicted_label,
        cancer_percentage=record.cancer_percentage,
        risk_band=record.risk_band,
        confidence_score=record.confidence_score,
        top_probability=record.top_probability,
        model_source=record.model_source,
        class_probabilities=json.loads(record.class_probabilities_json),
        created_at=record.created_at,
    ).model_dump()


def get_recent_records(db: Session) -> list[dict]:
    records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(20).all()
    return [serialize_record(record) for record in records]


def get_prediction_stats(db: Session) -> dict:
    total_predictions = db.query(func.count(PredictionRecord.id)).scalar() or 0
    average_percentage = db.query(func.avg(PredictionRecord.cancer_percentage)).scalar() or 0.0
    high_risk_cases = (
        db.query(func.count(PredictionRecord.id))
        .filter(PredictionRecord.risk_band == "High")
        .scalar()
        or 0
    )
    top_class = (
        db.query(PredictionRecord.predicted_class, func.count(PredictionRecord.id).label("count"))
        .group_by(PredictionRecord.predicted_class)
        .order_by(func.count(PredictionRecord.id).desc())
        .first()
    )

    return {
        "total_predictions": int(total_predictions),
        "average_percentage": round(float(average_percentage), 2),
        "high_risk_cases": int(high_risk_cases),
        "most_common_class": top_class[0] if top_class else "N/A",
    }


def run_prediction(image_path: Path) -> dict:
    return inference_service.predict(image_path)
