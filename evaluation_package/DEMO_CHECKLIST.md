# Demo checklist

1. Keep completed research artifacts intact. Do not start another campaign or activate research weights.
2. Run `.venv-research/Scripts/python.exe stain_research/demo_serve.py` from the project root; open http://127.0.0.1:8000.
3. The unchanged application uses isolated demo SQLite/uploads and CPU. Original database and GPU research checkpoints stay protected.
4. Identify the actual model source. Rehearsal used prototype-baseline; do not call its predictions research results.
5. Upload a TRAIN example, never a reserved test image. Show all eight probabilities and the added prediction-history row.
6. Explain FastAPI, the CNN/BiLSTM/attention architecture, the controller and SQL prediction_records storage.
7. Show the three-seed summary, tumor-recall regression, negative pilot and duplicate audit. Separate preprocessing, classifier and notebook metrics.
8. Use the offline slide/report PDFs if the demo fails. Do not fabricate a live result.
9. Stop the demo with Ctrl+C. No additional training is authorized by this checklist.
