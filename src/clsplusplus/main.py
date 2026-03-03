"""CLS++ entry point."""

import uvicorn

from clsplusplus.api import create_app
from clsplusplus.config import Settings


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
