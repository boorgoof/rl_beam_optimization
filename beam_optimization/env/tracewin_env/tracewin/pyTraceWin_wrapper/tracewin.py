#!/usr/bin/env python3

"""
Created on Mon Dec 13 17:28:11 2021

@author: davide.marcato@lnl.infn.it
"""

import subprocess
import numpy as np
import pandas as pd
import os, errno
import argparse
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

    def run(self, timeout, elem_params, other_params = {}, num_threads=None) -> bool:
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
        
        command_list = [
                self.EXECUTABLE,
                self.project,
                "hide",
                f"path_cal={self.outpath}",
                f"nbr_thread={num_threads}",
                *[f"{k}={v}" for k, v in elem_params.items()],
            ]
        self.last_stdout = ""
        self.last_stderr = ""

        try:
            completed_proc = subprocess.run(
                command_list,
                check=True, # Raise CalledProcessError if TraceWin returns non-zero exit code
                timeout=timeout,
                text=True,
                capture_output=True,
            )
            self.last_stdout = self._as_text(completed_proc.stdout)
            self.last_stderr = self._as_text(completed_proc.stderr)
            self.last_run_success = True
            return True
        except subprocess.TimeoutExpired as e:
            self.last_stdout = self._as_text(e.stdout)
            self.last_stderr = self._as_text(e.stderr)
            self.last_run_success = False
            print(f"TraceWin execution timed out after {timeout} seconds")
            return False
        
        except subprocess.CalledProcessError as e:
            self.last_stdout = self._as_text(e.stdout)
            self.last_stderr = self._as_text(e.stderr)
            self.last_run_success = False
            print(f"TraceWin execution failed with exit code {e.returncode}")
            print(self.last_stdout, self.last_stderr)
            return False



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
    parser = argparse.ArgumentParser(description="Creates a TraceWin batch command")

    parser.add_argument(
        "-p", "--project", dest="project", help="Project file", required=True
    )
    # path_cal
    parser.add_argument(
        "-o",
        "--output",
        dest="outpath",
        help="Path to calculation directory",
        required=True,
    )

    return parser


if __name__ == "__main__":
    parser = tracewin_parser()
    args = parser.parse_args()

    tracewin = TraceWin(args.project, args.outpath)
    tracewin.run({"ele[5][6]": -2100})
    print(tracewin.results().loc[7])
