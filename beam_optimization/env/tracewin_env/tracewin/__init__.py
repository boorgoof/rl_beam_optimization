from beam_optimization.env.simulation import BeamSimulationResult

__all__ = ["TraceWinSimulator", "BeamSimulationResult"]


def __getattr__(name):
    if name == "TraceWinSimulator":
        from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator

        return TraceWinSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
