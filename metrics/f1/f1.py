import evaluate
from sklearn.metrics import f1_score

_DESCRIPTION = "F1 score is the harmonic mean of precision and recall."
_KWARGS_DESCRIPTION = "Args: predictions: Predicted labels. references: Ground truth labels."
_CITATION = ""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class F1(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=evaluate.Features(
                {
                    "predictions": evaluate.Sequence(evaluate.Value("int32")),
                    "references": evaluate.Sequence(evaluate.Value("int32")),
                }
            ),
        )

    def _compute(self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None):
        score = f1_score(references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)
        return {"f1": float(score) if score.size == 1 else score}