CREATE TABLE IF NOT EXISTS prediction_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename VARCHAR(255) NOT NULL,
    predicted_class VARCHAR(16) NOT NULL,
    predicted_label VARCHAR(128) NOT NULL,
    cancer_percentage FLOAT NOT NULL,
    risk_band VARCHAR(32) NOT NULL,
    confidence_score FLOAT NOT NULL,
    top_probability FLOAT NOT NULL,
    model_source VARCHAR(64) NOT NULL,
    class_probabilities_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
