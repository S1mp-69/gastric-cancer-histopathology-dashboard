# Error review queue — expert review pending

The CSV records fixed standard/adaptive/within-class predictions for all three seeds on each reserved patch, sorted by model disagreement with the released annotation. It reads saved predictions and metadata only. It does not modify annotations, fit a model, select a new checkpoint or tune thresholds.

A frequent model disagreement is not proof of a bad label: the model may be wrong. Coarse released masks, mixed morphology and broad class definitions require a qualified reviewer to assess any proposed explanation. No local pathology expert has adjudicated these cases. Do not claim corrected ground truth or clinical utility from this queue. Keep the frozen test metrics unchanged; future relabeling would be a separate documented study with independently reviewed labels and a new development/evaluation protocol.

Coordinates and local image paths refer to the public HiESD release. Raw images are not included in this portable CSV. Dataset source: doi:10.1038/s41597-025-05679-1.
