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


## TO run exnovo tracewin in a new vitrual machine

### Full from-scratch setup for a brand new VM (no `comunian` user, no libraries, nothing)

Steps below, in order. Example uses a hypothetical `TraceWin_workspace_6` and a new VM
with no prior setup at all.

**1. Create the `comunian` user:**

```bash
id comunian || sudo adduser comunian
```

**2. Install the X11/Qt runtime libraries TraceWin needs to even start.**
Without these the binary fails immediately with `error while loading shared
libraries: libXi.so.6: cannot open shared object file` (or a similar missing
`.so`) the first time you try to run it:

```bash
sudo dnf install -y xorg-x11-server-Xvfb \
  libXi libXrender libXrandr libXfixes libXcursor libXcomposite \
  libXtst libXScrnSaver libXinerama libXdamage mesa-libGL fontconfig
```

**3. Place the licensed TraceWin binary and license files here**, then make it
executable:

```text
TraceWin_program/
├── TraceWin
├── tracewin_key.log
└── toutatis_key.log
```

```bash
chmod +x beam_optimization/env/tracewin_env/tracewin/pyTraceWin_wrapper/TraceWin_program/TraceWin
```

**4. Enable passwordless SSH from your normal account to `comunian`:**

```bash
test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
sudo mkdir -p /home/comunian/.ssh
cat ~/.ssh/id_ed25519.pub | sudo tee -a /home/comunian/.ssh/authorized_keys >/dev/null
sudo chown -R comunian:comunian /home/comunian/.ssh
sudo chmod 700 /home/comunian/.ssh
sudo chmod 600 /home/comunian/.ssh/authorized_keys
```

On a genuinely fresh VM `localhost`'s host key is also not yet trusted by your
user, which fails with `Host key verification failed`. Fix it once:

```bash
ssh-keyscan -H localhost >> ~/.ssh/known_hosts
```

If SSH still refuses to connect, check that `sshd` is actually running:

```bash
sudo systemctl status sshd || sudo systemctl start sshd
```

**5. The launcher script** (`run_tracewin_with_permissions.sh`, next to this
file) **is shared by every workspace and every VM that mounts this same repo
checkout** — there is exactly one copy, referenced by `TraceWin.EXECUTABLE`
regardless of which `--workspace` is used. If it already exists and works on
another VM, you don't need to recreate it. If it's missing (truly first setup
ever), create it:

```bash
cat > beam_optimization/env/tracewin_env/tracewin/pyTraceWin_wrapper/run_tracewin_with_permissions.sh << 'EOF'
#!/bin/bash
set -e

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

quoted_args=""
for arg in "$@"; do
  quoted_args="$quoted_args $(printf '%q' "$arg")"
done

ssh -F /dev/null comunian@localhost \
  "cd '$SCRIPT_DIR'; \
   if command -v xvfb-run >/dev/null 2>&1; then \
     xvfb-run -a ./TraceWin_program/TraceWin $quoted_args; \
   else \
     ./TraceWin_program/TraceWin $quoted_args; \
   fi"
EOF
chmod +x beam_optimization/env/tracewin_env/tracewin/pyTraceWin_wrapper/run_tracewin_with_permissions.sh
```

The `printf '%q'` quoting of every argument locally, then sending ONE fully
built command string to `ssh`, is important: `ssh host "cmd" -- "$@"` does
**not** forward `"$@"` to the remote shell the way a local `bash -c "cmd" --
"$@"` would — `ssh` just concatenates every word after the destination into
one string, so a trailing `-- "$@"` produces a broken remote command
(`syntax error near unexpected token '--'`) the moment real arguments (the
project `.ini` path, `path_cal=...`, `ele[...]=...`) are involved.

**6. Give `comunian` access to the new workspace directory**
(`TraceWin_workspace_6` here). Try ACLs first:

```bash
sudo setfacl -R -m u:comunian:rx beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6
sudo setfacl -m u:comunian:rwx beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6/calc
```

If that fails with `setfacl: ... Operation not supported` on every file, the
filesystem underneath (e.g. an NFS mount shared across VMs) doesn't support
POSIX ACLs at all. Fall back to plain world-writable permissions instead —
this is the same approach `TraceWinSimulator._reset_calc_dir()` already uses
for the calc directories it manages:

```bash
chmod -R a+rwX beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6
```

(`X` upper-case only sets execute on directories, not on data files.)

**7. Required checks**, in order:

```bash
# comunian exists
id comunian

# passwordless SSH works
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=5 comunian@localhost echo OK

# comunian can read the project and write to calc/
mkdir -p beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6/calc
sudo -u comunian test -r beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6/CB_newMRMS_RFQ_Fields_1.ini
sudo -u comunian test -w beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6/calc
```

**8. Final end-to-end test — run a real simulation through the Python
wrapper** (headless-safe: prints a clear `success = True/False` instead of
requiring a GUI you can't see on a VM with no graphical interface):

```bash
cd /mnt/meneghetti/FEDERICO_TESI/rl_beam_optimization
source beam_optimization/.venv/bin/activate
python -c "
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator
from beam_optimization.config.adige import default_params

sim = TraceWinSimulator(
    project_file='beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6/CB_newMRMS_RFQ_Fields_1.ini',
    calc_dir='beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6/calc',
    timeout=60.0,
    retries=0,
)
result = sim.simulate(default_params())
print('success =', result.success)
print('score   =', result.score_val)
print('error   =', result.error)
"
```

`success = True` with a real (non `-999`) score means the VM is fully set up.
`success = False` with `error = None` and a `QXcbConnection: Could not
connect to display` in the traceback means `xvfb-run` isn't installed or
isn't being picked up — re-check step 2. A bash `syntax error near
unexpected token '--'` means the shared launcher script (step 5) still has
the old broken `-- "$@"` form.

**9. Build a dataset on the new workspace:**

```bash
python -m beam_optimization build_dataset \
  --target-samples 4000 \
  --workspace beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_6 \
  --dataset-root beam_optimization/env/dataset \
  --timeout 180.0 --retries 2 --retry-sleep 5.0
```