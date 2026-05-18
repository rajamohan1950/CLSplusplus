"""CLS++ entry point."""

import uvicorn

from clsplusplus.api import create_app
from clsplusplus.config import Settings


def _init_sentry(settings: Settings) -> None:
    """Wire up Sentry error tracking when CLS_SENTRY_DSN is set; no-op otherwise.

    sentry-sdk auto-enables its FastAPI/Starlette integration, so unhandled
    exceptions are reported with request context once init() has run.
    """
    if not settings.sentry_dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        traces_sample_rate=0.0,
        send_default_pii=False,
    )


def main() -> None:
    settings = Settings()
    _init_sentry(settings)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
