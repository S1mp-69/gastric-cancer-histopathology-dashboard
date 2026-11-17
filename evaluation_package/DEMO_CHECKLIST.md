# Demonstration checklist

1. Keep the research supervisor running. Do not start another trainer or activate a research checkpoint.
2. Start the preserved app through `& .venv-research/Scripts/python.exe stain_research/demo_serve.py` (CPU default, isolated DB/uploads and training-only prototype references). Open http://127.0.0.1:8000.
3. Upload an HMU training example; identify the returned model_source. A prototype fallback must be called prototype-baseline, not a validated research model.
4. Explain eight tissue probabilities and prediction history; do not describe the heuristic weighted score as a clinical diagnosis, disease prevalence or tumor area.
5. Show saved scientific comparisons separately. HiESD ResNet18 is a different isolated research model with five groups.
6. Explain the frozen final-test protocol and actual current status. Do not tune after test access.
7. Stop only the demo server with Ctrl+C when finished; leave the research supervisor/checkpoints intact.

The original demo historically initialized prototypes from all_image, including 41 provisional HMU test paths. The wrapper fixes its reference source without editing the original application. This historical exposure is disclosed; it does not turn HMU into patient-independent data.
