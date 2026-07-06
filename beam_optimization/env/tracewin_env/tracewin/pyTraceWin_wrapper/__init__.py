from .files import Dst, Plt

__all__ = ["TraceWin", "tracewin_parser", "Dst", "Plt"]


def __getattr__(name):
    if name in {"TraceWin", "tracewin_parser"}:
        from .tracewin import TraceWin, tracewin_parser

        return {"TraceWin": TraceWin, "tracewin_parser": tracewin_parser}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
