"""Custom exception classes for the mlsynth library."""

class MlsynthError(Exception):
    """Base class for all custom exceptions in the mlsynth library."""
    pass

class MlsynthConfigError(MlsynthError):
    """Exception raised for errors in configuration."""
    pass

class MlsynthDataError(MlsynthError):
    """Exception raised for errors related to input data."""
    pass

class MlsynthEstimationError(MlsynthError):
    """Exception raised for errors during the estimation process."""
    pass

class MlsynthPlottingError(MlsynthError):
    """Exception raised for errors during plot generation."""
    pass
