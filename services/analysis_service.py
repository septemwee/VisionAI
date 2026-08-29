"""Placeholder physical-analysis service.

The real analysis is not implemented yet. This stub keeps the trainer wiring
intact until it is replaced with measured package statistics.
"""


class AnalysisService:
    """Returns basic physical attributes of a package recipe."""

    def analyze(self, recipe_data, dataset_data):
        return {
            "family": recipe_data["package_family"],
            "pin_count": recipe_data["pin_count"],
            "aspect_ratio": 2.2,
        }
