"""Crawl refusals whose code and message are safe to show the user."""


class CrawlError(ValueError):
    """A crawl, or a change to a user's crawl sites, was refused."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def public_detail(self) -> dict:
        return {"code": self.code, "message": self.message}
