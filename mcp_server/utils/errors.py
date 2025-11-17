"""
Custom Error Classes

Defines all custom exception types used by MCP Server.
"""

from typing import Optional


class MCPError(Exception):
    """MCP Tool Error Base Class"""

    def __init__(self, message: str, code: str = "MCP_ERROR", suggestion: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        error_dict = {
            "code": self.code,
            "message": self.message
        }
        if self.suggestion:
            error_dict["suggestion"] = self.suggestion
        return error_dict


class DataNotFoundError(MCPError):
    """Data Not Found Error"""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(
            message=message,
            code="DATA_NOT_FOUND",
            suggestion=suggestion or "Please check date range or wait for crawl task to complete"
        )


class InvalidParameterError(MCPError):
    """Invalid Parameter Error"""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(
            message=message,
            code="INVALID_PARAMETER",
            suggestion=suggestion or "Please check if parameter format is correct"
        )


class ConfigurationError(MCPError):
    """Configuration Error"""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            suggestion=suggestion or "Please check if configuration file is correct"
        )


class PlatformNotSupportedError(MCPError):
    """Platform Not Supported Error"""

    def __init__(self, platform: str):
        super().__init__(
            message=f"Platform '{platform}' is not supported",
            code="PLATFORM_NOT_SUPPORTED",
            suggestion="Supported platforms: cnn, bbc, reuters, nytimes, washingtonpost, npr, abc-news, cbs-news, nbc-news, fox-news, usatoday, theguardian"
        )


class CrawlTaskError(MCPError):
    """Crawl Task Error"""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(
            message=message,
            code="CRAWL_TASK_ERROR",
            suggestion=suggestion or "Please retry later or check logs"
        )


class FileParseError(MCPError):
    """File Parse Error"""

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message=f"Failed to parse file {file_path}: {reason}",
            code="FILE_PARSE_ERROR",
            suggestion="Please check if file format is correct"
        )
