#!/usr/bin/env python3

"""
Created on Mon Dec 13 17:28:11 2021

@author: davide.marcato@lnl.infn.it
"""
from __future__ import annotations

import argparse
import errno
import os
import signal
import subprocess

import pandas as pd

from .files import Dst, Plt


class TraceWin:
    """
    Wrapper for TraceWin operations
    """

    EXECUTABLE = os.path.join(os.path.dirname(__file__), "run_tracewin_with_permissions.sh")

    def __init__(self, project, outpath) -> None:
        self.project = os.path.abspath(project)
        self.outpath = os.path.abspath(outpath)
        self.last_stdout = ""
        self.last_stderr = ""
        self.last_run_success = None
        
        # File and folder must exists
        # Should be checked in run but may slow down computation
        if not os.path.exists(self.outpath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.outpath)
        if not os.path.exists(self.project):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.project)

    @staticmethod
    def _as_text(value) -> str:
        """Convert subprocess output payloads (str/bytes/None) to safe text."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def run(self, timeout, elem_params, other_params=None, num_threads=None) -> bool:
        """Execute TraceWin and wait for completion
        
        Args:
            timeout: Timeout in seconds
            elem_params: Dictionary of element parameters
            other_params: Dictionary of other parameters
            num_threads: Number of threads for TraceWin (default: all available CPUs)

        Returns:
            True if TraceWin exited successfully, False otherwise.
        """
        # Use all available CPUs if num_threads not specified
        if num_threads is None:
            num_threads = os.cpu_count() or 1
        if other_params is None:
            other_params = {}
        
        command_list = [
                self.EXECUTABLE,
                self.project,
                "hide",
                f"path_cal={self.outpath}",
                f"nbr_thread={num_threads}",
                *[f"{k}={v}" for k, v in other_params.items()],
                *[f"{k}={v}" for k, v in elem_params.items()],
            ]
        self.last_stdout = ""
        self.last_stderr = ""

        # start_new_session=True puts the wrapper script and the ssh client it
        # forks in the same new process group, so a timeout can kill the whole
        # group instead of only the immediate (wrapper) child.
        proc = subprocess.Popen(
            command_list,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._kill_process_group(proc, pgid)
            self._kill_remote_tracewin_processes()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired as cleanup_exc:
                self._kill_process_group(proc, pgid)
                self._kill_remote_tracewin_processes()
                stdout = cleanup_exc.output or exc.output or ""
                stderr = cleanup_exc.stderr or exc.stderr or ""
            self.last_stdout = self._as_text(stdout)
            self.last_stderr = self._as_text(stderr)
            self.last_run_success = False
            print(f"TraceWin execution timed out after {timeout} seconds")
            return False

        self.last_stdout = self._as_text(stdout)
        self.last_stderr = self._as_text(stderr)

        if proc.returncode != 0:
            self.last_run_success = False
            print(f"TraceWin execution failed with exit code {proc.returncode}")
            print(self.last_stdout, self.last_stderr)
            return False

        if self._reports_internal_error(self.last_stdout, self.last_stderr):
            self.last_run_success = False
            print("TraceWin reported an internal error despite exit code 0")
            print(self.last_stdout, self.last_stderr)
            return False

        self.last_run_success = True
        return True

    @staticmethod
    def _reports_internal_error(stdout: str, stderr: str) -> bool:
        """Return True when TraceWin reports a physics/internal failure in text output."""
        text = f"{stdout}\n{stderr}".lower()
        return "error:" in text or "transport failed" in text

    @staticmethod
    def _kill_process_group(proc: "subprocess.Popen", pgid: int) -> None:
        """Kill the whole process group started for ``proc``.

        ``proc.kill()`` alone only signals the immediate child (the wrapper
        script); it leaves the ssh client it spawned running, which keeps the
        stdout/stderr pipes open and makes the next ``communicate()`` block
        indefinitely instead of returning after the timeout.
        """
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    @staticmethod
    def _kill_remote_tracewin_processes() -> None:
        """Best-effort cleanup for TraceWin processes left alive over SSH."""
        try:
            subprocess.run(
                [
                    "ssh",
                    "-F", "/dev/null",
                    "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=5",
                    "comunian@localhost",
                    "pkill -u comunian -x TraceWin || true; "
                    "pkill -u comunian -f '[x]vfb-run.*TraceWin' || true",
                ],
                timeout=10,
                capture_output=True,
            )
        except Exception:
            pass



    def results(self) -> pd.DataFrame:
        """Read partran1.out file and return DataFrame with data"""
        outfile = self.outpath + "/partran1.out"
        if not os.path.exists(outfile):
            raise self._build_missing_output_error(outfile)
        return pd.read_csv(outfile, sep="\s+", header=0, skiprows=9)
    
    def dst(self, out=True) -> Dst:
        filename = "part_dtl1.dst" if out else "part_rfq.dst"
        dst_path = self.outpath + "/" + filename
        if not os.path.exists(dst_path):
            raise self._build_missing_output_error(dst_path)
        return Dst(dst_path)
    
    def _build_missing_output_error(self, expected_file: str) -> RuntimeError:
        stdout_preview = (self.last_stdout or "").strip()
        stderr_preview = (self.last_stderr or "").strip()
        status = (
            "success"
            if self.last_run_success is True
            else "failure"
            if self.last_run_success is False
            else "unknown"
        )
        details = [f"last_run_status={status}"]
        if stderr_preview:
            details.append(f"stderr={stderr_preview}")
        if stdout_preview:
            details.append(f"stdout={stdout_preview}")
        return RuntimeError(
            f"TraceWin output file not found: {expected_file}. "
            + " | ".join(details)
        )
    def plt(self) -> Plt:
        return Plt(self.outpath+"/dtl1.plt")


def tracewin_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one TraceWin batch simulation")

    parser.add_argument(
        "-p", "--project", dest="project", help="TraceWin project .ini file", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="outpath",
        help="Path to the TraceWin calculation/output directory",
        required=True,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum seconds to wait for TraceWin",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of TraceWin threads; defaults to all available CPUs",
    )
    parser.add_argument(
        "params",
        nargs="*",
        help="TraceWin parameter overrides as key=value, e.g. 'ele[5][6]=-2100'",
    )

    return parser


def _parse_cli_params(items: list[str]) -> dict[str, float | str]:
    """Parse CLI key=value items into a parameter dictionary."""
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid parameter {item!r}; expected key=value")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Invalid parameter {item!r}; empty key")
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    return params


if __name__ == "__main__":
    parser = tracewin_parser()
    args = parser.parse_args()

    tracewin = TraceWin(args.project, args.outpath)
    elem_params = _parse_cli_params(args.params)
    ok = tracewin.run(
        timeout=args.timeout,
        elem_params=elem_params,
        num_threads=args.threads,
    )
    if not ok:
        raise SystemExit(1)

    print(tracewin.results().tail(1).to_string(index=False))
