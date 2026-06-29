# TraceWin_program


This directory contains the real TraceWin executable used by the local Python wrapper

  
## Set up 
### See  this only the first time you want run TraceWin with the GUI Launch

The following is the set up for the licensed TraceWin valid for the Linux user `comunian`.

Create the `comunian` user if it does not already exist:

```bash
id comunian || sudo adduser comunian
```

Place the licensed TraceWin binary here:

```text
beam_optimization/env/tracewin_env/tracewin/pyTraceWin_wrapper/TraceWin_program/TraceWin
```

Make the binary executable:

```bash
chmod +x beam_optimization/env/tracewin_env/tracewin/pyTraceWin_wrapper/TraceWin_program/TraceWin
```

## Manual GUI launch: 
### Use this when you want to open the TraceWin graphical interface manually.

From this directory:
    xhost +SI:localuser:comunian
    sudo -u comunian DISPLAY=$DISPLAY ./TraceWin

Meaning:

    sudo -u comunian   run the program as user comunian
    DISPLAY=$DISPLAY   reuse the current graphical display
    ./TraceWin         start the TraceWin executable