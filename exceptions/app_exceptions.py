""" Application-specific exception types used across service boundaries. """

class ApplicationError(Exception):
    """Base application exception."""

class ConfigurationError(ApplicationError):
    """ ConfigurationError groups related application behavior. """
    pass

class ServiceUnavailableError(ApplicationError):
    """ ServiceUnavailableError groups related application behavior. """
    pass

class ValidationError(ApplicationError):
    """ ValidationError groups related application behavior. """
    pass

class IngestionError(ApplicationError):
    """ IngestionError groups related application behavior. """
    pass

class RetrievalError(ApplicationError):
    """ RetrievalError groups related application behavior. """
    pass
