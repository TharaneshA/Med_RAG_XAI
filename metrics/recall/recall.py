import evaluate
from sklearn.metrics import recall_score
from datasets import Features, Value, Sequence

_DESCRIPTION = "Recall is the fraction of relevant instances that were retrieved."
_KWARGS_DESCRIPTION = "Args: predictions: Predicted labels. references: Ground truth labels."
_CITATION = ""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Recall(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=Features(
                {
                    "predictions": Sequence(Value("int32")),
                    "references": Sequence(Value("int32")),
                }
            ),
        )

    def _compute(self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None):
        score = recall_score(references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)
        return {"recall": float(score) if score.size == 1 else score}