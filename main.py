"""Entry point kept for existing deploys: ``uvicorn main:app``."""

from gmp.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gmp.app:app", host="0.0.0.0", port=8000)
