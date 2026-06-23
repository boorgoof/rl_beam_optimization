TraceWin_program
================

This directory contains the real TraceWin executable used by the local Python
wrapper:

    TraceWin

In this project TraceWin is normally executed as the Linux user "comunian".
This is a local installation/licence choice, not a physics requirement.


(1) Manual GUI launch: use this only when you want to open the TraceWin graphical interface manually.

From this directory:

    sudo -u comunian DISPLAY=$DISPLAY ./TraceWin

Meaning:

    sudo -u comunian   run the program as user comunian
    DISPLAY=$DISPLAY   reuse the current graphical display
    ./TraceWin         start the TraceWin executable





