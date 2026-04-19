from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping


class CountingParadigm(str, Enum):
	"""Supported counting paradigms in this project."""

	DETECTION = "detection"
	DENSITY_MAP = "density_map"
	POINT_REGRESSION = "point_regression"


@dataclass
class PredictionResult:
	"""Unified prediction structure shared by all counting paradigms."""

	count: float | None = None
	boxes: Any | None = None
	points: Any | None = None
	density_map: Any | None = None
	scores: Any | None = None
	labels: Any | None = None
	metadata: dict[str, Any] = field(default_factory=dict)
	raw: Any | None = None


class BaseCountingModel(ABC):
	"""
	Base interface for all animal counting models.

	Concrete models (YOLOv8, CSRNet, P2PNet, ...) inherit this class and implement the training, inference and persistence methods.
	"""

	def __init__(
		self,
		name: str,
		paradigm: CountingParadigm | str,
		device: str | None = None,
		config: Mapping[str, Any] | None = None,
	) -> None:
		self.name = name
		self.paradigm = CountingParadigm(paradigm)
		self.device = device or "cpu"
		self.config = dict(config or {})

	@abstractmethod
	def fit(
		self,
		**kwargs: Any,
	) -> Mapping[str, float] | None:
		"""Train the model and optionally return a metrics summary."""
		raise NotImplementedError

	@abstractmethod
	def predict(self, image: Any, **kwargs: Any) -> PredictionResult | Mapping[str, Any]:
		"""
		Run inference for one input image.

		Implementations may return either a PredictionResult or a dict-like object.
		Dict outputs are normalized by normalize_prediction().
		"""
		raise NotImplementedError

	@abstractmethod
	def save(self, path: str | Path) -> None:
		"""Persist model weights/artifacts to disk."""
		raise NotImplementedError

	def predict_count(self, image: Any, **kwargs: Any) -> float:
		"""Return only the estimated count for one image."""
		prediction = self.predict(image, **kwargs)
		normalized = self.normalize_prediction(prediction)
		return float(normalized.count or 0.0)

	def normalize_prediction(
		self,
		prediction: PredictionResult | Mapping[str, Any],
	) -> PredictionResult:
		"""
		Convert model-specific outputs into PredictionResult.

		If count is missing, infer it from available outputs:
		- detection: number of boxes
		- point-based: number of points
		- density map: integral (sum) over all pixels
		"""
		if isinstance(prediction, PredictionResult):
			result = prediction
		elif isinstance(prediction, Mapping):
			result = PredictionResult(
				count=prediction.get("count"),
				boxes=prediction.get("boxes"),
				points=prediction.get("points"),
				density_map=prediction.get("density_map"),
				scores=prediction.get("scores"),
				labels=prediction.get("labels"),
				metadata=dict(prediction.get("metadata") or {}),
				raw=prediction.get("raw", prediction),
			)
		else:
			raise TypeError(
				"Prediction must be PredictionResult or mapping, "
				f"got {type(prediction).__name__}."
			)

		if result.count is None:
			result.count = self._infer_count_from_outputs(
				boxes=result.boxes,
				points=result.points,
				density_map=result.density_map,
			)
		return result

	def evaluate_counts(
		self,
		predicted_counts: Iterable[float],
		target_counts: Iterable[float],
	) -> dict[str, float]:
		"""Compute standard counting metrics on a list of predictions."""
		y_pred = [float(x) for x in predicted_counts]
		y_true = [float(x) for x in target_counts]

		if len(y_pred) != len(y_true):
			raise ValueError(
				"predicted_counts and target_counts must have the same length. "
				f"Got {len(y_pred)} and {len(y_true)}."
			)
		if not y_true:
			raise ValueError("Cannot evaluate on empty inputs.")

		errors = [p - t for p, t in zip(y_pred, y_true)]
		abs_errors = [abs(e) for e in errors]
		sq_errors = [e * e for e in errors]

		n = float(len(y_true))
		mae = sum(abs_errors) / n
		rmse = sqrt(sum(sq_errors) / n)
		bias = sum(errors) / n

		non_zero_targets = [t for t in y_true if t != 0.0]
		if non_zero_targets:
			ape_values = [
				abs((p - t) / t)
				for p, t in zip(y_pred, y_true)
				if t != 0.0
			]
			mape = 100.0 * sum(ape_values) / float(len(ape_values))
		else:
			mape = float("nan")

		return {
			"mae": mae,
			"rmse": rmse,
			"bias": bias,
			"mape": mape,
			"num_samples": float(len(y_true)),
		}

	def evaluate_dataset(
		self,
		samples: Iterable[Mapping[str, Any]],
		image_key: str = "image",
		target_key: str = "target",
		count_key: str = "count",
		**predict_kwargs: Any,
	) -> dict[str, float]:
		"""
		Run counting evaluation on an iterable of dataset-like samples.

		Each sample must provide:
		- sample[image_key]: model input image
		- sample[target_key][count_key] or sample[count_key]: target count
		"""
		predicted_counts: list[float] = []
		target_counts: list[float] = []

		for sample in samples:
			image = sample[image_key]
			predicted_counts.append(self.predict_count(image, **predict_kwargs))

			if target_key in sample and isinstance(sample[target_key], Mapping):
				target = sample[target_key]
				if count_key not in target:
					raise KeyError(
						f"Missing '{count_key}' in sample[{target_key!r}] during evaluation."
					)
				target_counts.append(float(target[count_key]))
			elif count_key in sample:
				target_counts.append(float(sample[count_key]))
			else:
				raise KeyError(
					f"Could not find target count in sample[{target_key!r}]['{count_key}'] "
					f"or sample['{count_key}']."
				)

		return self.evaluate_counts(predicted_counts, target_counts)

	@staticmethod
	def _infer_count_from_outputs(
		boxes: Any | None,
		points: Any | None,
		density_map: Any | None,
	) -> float:
		if boxes is not None:
			return float(BaseCountingModel._safe_len(boxes))
		if points is not None:
			return float(BaseCountingModel._safe_len(points))
		if density_map is not None:
			return float(BaseCountingModel._safe_sum(density_map))
		return 0.0

	@staticmethod
	def _safe_len(obj: Any) -> int:
		# Supports Python lists, NumPy arrays, and tensor-like objects.
		if hasattr(obj, "shape") and len(getattr(obj, "shape")) >= 1:
			return int(obj.shape[0])
		return int(len(obj))

	@staticmethod
	def _safe_sum(obj: Any) -> float:
		if hasattr(obj, "sum"):
			summed = obj.sum()
			if hasattr(summed, "item"):
				return float(summed.item())
			return float(summed)

		if isinstance(obj, (list, tuple)):
			total = 0.0
			for item in obj:
				total += BaseCountingModel._safe_sum(item)
			return total

		return float(obj)
