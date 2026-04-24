TISSUE_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]

CLASS_LABELS = {
    "ADI": "Adipose Tissue",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Smooth Muscle",
    "NOR": "Normal Mucosa",
    "STR": "Cancer-Associated Stroma",
    "TUM": "Tumor",
}

CLASS_COLORS = {
    "ADI": "#1f4f8b",
    "DEB": "#4d8fd1",
    "LYM": "#2f9d8f",
    "MUC": "#d08f31",
    "MUS": "#8c6bb1",
    "NOR": "#6b7b8c",
    "STR": "#c4572e",
    "TUM": "#a61e4d",
}

# Weighted probability mix used to convert tissue composition into a demo-friendly cancer percentage.
CANCER_WEIGHTS = {
    "ADI": 0,
    "DEB": 18,
    "LYM": 35,
    "MUC": 28,
    "MUS": 8,
    "NOR": 2,
    "STR": 62,
    "TUM": 98,
}
