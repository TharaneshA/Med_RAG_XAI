import evaluate
from sklearn.metrics import accuracy_score
from datasets import Features, Value

_DESCRIPTION = "Accuracy is the proportion of correct predictions among the total number of cases processed."
_KWARGS_DESCRIPTION = "Args: predictions: Predicted labels. references: Ground truth labels."
_CITATION = ""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=Features(
                {
                    "predictions": Value("int32"),
                    "references": Value("int32"),
                }
            ),
        )

    def _compute(self, predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }