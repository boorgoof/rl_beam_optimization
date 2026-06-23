# Guida pratica a TraceWin

Questa guida spiega come usare TraceWin nel progetto, quali file guardare e come
salvare/visualizzare le distribuzioni del fascio. Dataset e ambienti RL verranno
documentati separatamente.

## 1. A cosa serve TraceWin

TraceWin e' il simulatore fisico usato per propagare il fascio lungo la linea.
Dato un progetto TraceWin, il programma legge il lattice, trasporta le particelle
e produce i risultati nella cartella `calc/`.

Nel workflow manuale tipico:

1. apri il progetto TraceWin;
2. controlli o modifichi il lattice;
3. inserisci eventuali diagnostiche o `PLOT_DST`;
4. lanci il calcolo multiparticle/PARTRAN;
5. controlli gli output in `calc/`;
6. usi `visualize_distributions.ipynb` per plottare le distribuzioni `.dst`.

## 2. File importanti

I file principali sono:

| File/cartella | Ruolo |
| --- | --- |
| `pyTraceWin_wrapper/TraceWin_program/TraceWin` | Eseguibile TraceWin usato dal wrapper Python. |
| `pyTraceWin_wrapper/run_tracewin_with_permissions.sh` | Script locale che avvia TraceWin con utente/permessi corretti. Non e' versionato perche' dipende dalla macchina. |
| `pyTraceWin_wrapper/tracewin.py` | Wrapper Python che costruisce il comando, lancia TraceWin e legge `partran1.out`. |
| `TraceWin_workspace/condensed.ini` | File progetto TraceWin. E' il punto di ingresso del progetto. |
| `TraceWin_workspace/condensed.dat` | Lattice della linea: drift, steerers, mappe, diagnostiche, ecc. |
| `TraceWin_workspace/16O5.dst` | Distribuzione iniziale/reservoir di particelle. |
| `calc/` | Cartella di output generata o aggiornata da TraceWin. |
| `calc/partran1.out` | Tabella con i risultati elemento per elemento lungo la linea. |
| `calc/part_rfq.dst` | Distribuzione in ingresso usata da PARTRAN. |
| `calc/part_dtl1.dst` | Distribuzione finale in uscita. |
| `calc/dtl1.plt` | File multiparticle con traiettorie/distribuzioni lungo la linea. |
| `calc/1.dst`, `calc/2.dst`, ... | Distribuzioni intermedie prodotte da `PLOT_DST`. |

Nel caso che stai usando ora, i path principali sono:

```text
Workspace:
/mnt/shared_volume/FEDERICO_TESI/DANIELE/TRACEWIN_WORKSPACE_test/TRACEWIN_WORKSPACE_0

Calc folder:
/mnt/shared_volume/FEDERICO_TESI/DANIELE/TRACEWIN_WORKSPACE_test/TRACEWIN_WORKSPACE_0/calc

Input distribution:
/mnt/shared_volume/FEDERICO_TESI/DANIELE/TRACEWIN_WORKSPACE_test/TRACEWIN_WORKSPACE_0/16O5.dst

Output distribution:
/mnt/shared_volume/FEDERICO_TESI/DANIELE/TRACEWIN_WORKSPACE_test/TRACEWIN_WORKSPACE_0/calc/part_dtl1.dst
```

## 3. Setup iniziale: utente `comunian` (leggi solo se e' la prima volta)

Questa sezione serve solo quando stai configurando la macchina o quando TraceWin
non parte. Se l'ambiente e' gia' configurato, puoi saltare direttamente alla
sezione successiva: **Come avviare TraceWin**.

### Perche' esiste l'utente `comunian`

In questo setup TraceWin viene eseguito come utente Linux `comunian`. Non e' un
dettaglio fisico della simulazione: e' una scelta pratica legata
all'installazione, alla licenza e all'ambiente grafico usato da TraceWin.

Ci sono quindi due utenti da tenere distinti:

- il tuo utente normale, con cui lavori nel repository, apri notebook e modifichi
  i file;
- l'utente `comunian`, con cui viene lanciato il programma TraceWin.

Questo significa che:

- `comunian` deve poter leggere il workspace e i file `.ini`, `.dat`, `.dst`;
- `comunian` deve poter scrivere nella cartella `calc/`;
- quando il codice Python lancia TraceWin, passa tramite `ssh comunian@localhost`.

Se TraceWin parte ma non genera output, spesso il problema non e' il lattice: e'
un permesso mancante sulla cartella `calc/`.

### Controlli da fare la prima volta

Verifica che l'utente esista:

```bash
id comunian
```

Se il comando risponde con un errore tipo `no such user`, sulla macchina non
esiste ancora l'utente `comunian`. In quel caso hai due possibilita'.

La soluzione consigliata e' chiedere a chi amministra la macchina di creare e
configurare l'utente `comunian`, perche' in questo progetto TraceWin e' stato
pensato per partire con quell'utente. L'amministratore deve controllare almeno
queste cose:

```bash
sudo adduser comunian
sudo usermod -aG <gruppo-del-workspace> comunian
```

Il nome del gruppo dipende dalla macchina. L'obiettivo non e' il nome del
gruppo in se', ma fare in modo che `comunian` possa leggere il repository e
scrivere nelle cartelle `calc/` dei workspace TraceWin.

Dopo aver creato l'utente, bisogna anche configurare l'accesso SSH locale senza
password, perche' lo script `run_tracewin_with_permissions.sh` di solito usa:

```bash
ssh comunian@localhost
```

Se invece vuoi usare il tuo utente personale al posto di `comunian`, devi
creare/modificare `run_tracewin_with_permissions.sh` e sostituire `comunian` con il tuo nome
utente. Questa strada e' piu' semplice solo se la licenza di TraceWin e i
permessi sui file funzionano anche con il tuo utente. Se TraceWin e' stato
installato/licenziato per `comunian`, usare un altro utente puo' far partire il
programma ma poi fallire sulla licenza o sulla scrittura degli output.

Verifica che SSH locale funzioni senza password:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 comunian@localhost echo OK
```

Il comando deve stampare `OK` senza chiedere password. Se chiede una password o
fallisce, il wrapper Python non riuscira' a lanciare TraceWin automaticamente.

Verifica che `comunian` possa scrivere nella cartella `calc/`:

```bash
sudo -u comunian test -w /mnt/shared_volume/FEDERICO_TESI/DANIELE/TRACEWIN_WORKSPACE_test/TRACEWIN_WORKSPACE_0/calc && echo writable
```

Se non stampa `writable`, TraceWin potrebbe partire ma non riuscire a generare
`partran1.out`, `part_dtl1.dst` o gli `.dst` intermedi.

### File legati alla licenza

Nella cartella dell'eseguibile ci sono anche:

```text
pyTraceWin_wrapper/TraceWin_program/tracewin_key.log
pyTraceWin_wrapper/TraceWin_program/toutatis_key.log
```

Sono file legati alla configurazione/licenza locale di TraceWin/Toutatis. Non
vanno modificati durante il normale uso del progetto.

## 4. Come avviare TraceWin

L'eseguibile si trova qui:

```text
beam_optimization/env/tracewin_env/tracewin/pyTraceWin_wrapper/TraceWin_program/TraceWin
```

### (1) Avvio manuale con interfaccia grafica

Usa questo quando vuoi aprire TraceWin dalla GUI, controllare il progetto,
modificare il lattice o lanciare PARTRAN manualmente.

```bash
cd /mnt/shared_volume/FEDERICO_TESI/beam_optimization/env/tracewin_env/tracewin/pyTraceWin_wrapper/TraceWin_program
sudo -u comunian DISPLAY=$DISPLAY ./TraceWin
```

Il comando significa:

- `sudo -u comunian`: esegui TraceWin come utente `comunian`;
- `DISPLAY=$DISPLAY`: usa il display grafico corrente;
- `./TraceWin`: avvia l'eseguibile nella cartella corrente.

Una volta aperta la GUI:

1. apri il file `.ini` del workspace;
2. controlla che punti al lattice `.dat` corretto;
3. controlla che la distribuzione iniziale `.dst` sia quella voluta;
4. modifica il lattice se necessario;
5. lancia il calcolo multiparticle/PARTRAN;
6. controlla gli output nella cartella `calc/`.

### (2) Avvio automatico da shell/Python

Il progetto include un wrapper:

```text
beam_optimization/env/tracewin_env/tracewin/pyTraceWin_wrapper/run_tracewin_with_permissions.sh
```

Il contenuto essenziale e':

```bash
ssh comunian@localhost "cd $SCRIPT_DIR; if command -v xvfb-run >/dev/null 2>&1; then xvfb-run -a ./TraceWin_program/TraceWin $@; else ./TraceWin_program/TraceWin $@; fi"
```

Questo wrapper:

1. entra via SSH locale come `comunian`;
2. si sposta nella cartella `pyTraceWin_wrapper`;
3. lancia `TraceWin_program/TraceWin`.

Se `xvfb-run` e' disponibile, TraceWin viene avviato dentro un display virtuale.
Questo e' utile per lanciare TraceWin in modalita' batch/headless, cioe' senza
dover aprire una finestra grafica visibile.

Il wrapper Python `pyTraceWin_wrapper` usa proprio questo script. In particolare:

```text
pyTraceWin_wrapper/tracewin.py
```

contiene:

```python
EXECUTABLE = "run_tracewin_with_permissions.sh"
```

Quando chiami `TraceWin.run(...)`, viene costruito un comando del tipo:

```text
run_tracewin_with_permissions.sh condensed.ini hide path_cal=/path/to/calc nbr_thread=N ele[i][j]=valore ...
```

I pezzi importanti sono:

- `condensed.ini`: progetto TraceWin da aprire;
- `hide`: modalita' senza GUI interattiva;
- `path_cal=...`: cartella dove TraceWin deve scrivere gli output;
- `nbr_thread=...`: numero di thread;
- `ele[i][j]=...`: eventuali parametri del lattice modificati da codice.

### Quando usare quale avvio

Usa la GUI manuale quando devi:

- guardare il progetto;
- modificare il lattice;
- aggiungere `PLOT_DST`;
- verificare visivamente le impostazioni di PARTRAN/multiparticle.

Usa il wrapper automatico quando devi:

- fare run ripetuti;
- chiamare TraceWin da Python;
- produrre output in una cartella `calc/` specifica;
- integrare TraceWin in script o notebook.


## 5. Modificare la configurazione dell'acceleratore

La configurazione dell'acceleratore e' descritta da il  file `.dat`, ad esempio:

```text
TraceWin_workspace/condensed.dat
```

Dentro trovi gli elementi della linea, per esempio:

```text
DRIFT 300 150 0 0 0
AD.ST.01 : THIN_STEERING 0.01 0 150 0
DRIFT 200 150 0 0 0
```

Ogni riga viene interpretata da TraceWin come un elemento o un comando. Se un
comando non e' riconosciuto, TraceWin mostra un errore tipo:

```text
Unknown element or command
```

Un caso importante: `SAVE_BEAM` non e' il comando giusto in questo setup. Per
salvare distribuzioni intermedie bisogna usare `PLOT_DST`.

## 6. Salvare distribuzioni intermedie con PLOT_DST

Per salvare la distribuzione del fascio in punti intermedi, inserisci righe
`PLOT_DST` nel lattice.

Esempio con due steerers:

```text
DRIFT 300 150 0 0 0
PLOT_DST 1

AD.ST.01 : THIN_STEERING 0.01 0 150 0
PLOT_DST 2

DRIFT 200 150 0 0 0
PLOT_DST 3

AD.ST.02 : THIN_STEERING -0.01 0 150 0
PLOT_DST 4

DRIFT 300 150 0 0 0
PLOT_DST 5

END
```

Interpretazione:

- `PLOT_DST 1`: distribuzione prima del primo steerer;
- `PLOT_DST 2`: subito dopo il primo steerer;
- `PLOT_DST 3`: dopo il drift, dove si vede l'effetto del primo kick sulla posizione;
- `PLOT_DST 4`: subito dopo il secondo steerer;
- `PLOT_DST 5`: dopo l'ultimo drift, con l'effetto combinato dei due steerers.

Se vuoi uno steerer orizzontale e uno verticale:

```text
AD.ST.X : THIN_STEERING 0.01 0 150 0
AD.ST.Y : THIN_STEERING 0 0.01 150 0
```

Nota fisica: subito dopo uno `THIN_STEERING` cambia soprattutto l'angolo del
fascio. Lo spostamento trasversale diventa evidente nel drift successivo.

## 7. Lanciare PARTRAN/multiparticle

Per ottenere file `.dst` e distribuzioni particellari devi lanciare il calcolo
multiparticle/PARTRAN, non solo envelope.

Dopo il run, controlla la cartella:

```text
.../TRACEWIN_WORKSPACE_0/calc
```

I file da guardare sono:

```text
partran1.out
part_rfq.dst
part_dtl1.dst
dtl1.plt
1.dst
2.dst
...
```

Se hai inserito `PLOT_DST 1`, `PLOT_DST 2`, ecc., TraceWin puo' produrre file
intermedi come:

```text
1.dst
2.dst
3.dst
```

Il nome dei file intermedi dipende da come TraceWin esporta i `PLOT_DST`, ma
nel tuo run attuale sono file numerici dentro `calc/`.

## 8. Leggere partran1.out

`partran1.out` e' una tabella con lo stato del fascio lungo la linea. Alcune
colonne utili sono:

| Colonna | Significato |
| --- | --- |
| `##` | indice/marker dell'elemento |
| `z(m)` | posizione longitudinale in metri |
| `x0`, `y0` | centroidi trasversali |
| `x'0`, `y'0` | angoli/divergenze medie |
| `SizeX`, `SizeY` | dimensioni RMS |
| `ex`, `ey` | emittanze RMS |
| `npart` | numero di particelle rimaste |
| `Aper` | apertura usata nei plot x-y |

Questo file e' utile per capire se lo steerer sta cambiando angolo e posizione,
se ci sono perdite, e dove il fascio cresce rispetto all'apertura.

## 9. Visualizzare le distribuzioni

Per plottare le distribuzioni usa:

```text
beam_optimization/env/tracewin_env/tracewin/visualize_distributions.ipynb
```

Il notebook legge:

```text
16O5.dst
calc/*.dst
calc/part_dtl1.dst
calc/partran1.out
```

e produce, per ogni distribuzione:

- plot `x-y`;
- plot `x-x'`;
- plot `y-y'`;
- aperture nel piano `x-y`;
- centroide del fascio.

Per usarlo:

1. lancia TraceWin/PARTRAN;
2. controlla che `calc/` sia aggiornata;
3. apri `visualize_distributions.ipynb`;
4. fai `Run All`.

I titoli delle figure sono:

- `Input distribution` per `16O5.dst`;
- il nome file, ad esempio `1.dst`, `2.dst`, per i punti intermedi;
- `Output distribution` per `part_dtl1.dst`.

## 10. Troubleshooting

### Il notebook non si aggiorna

Controlla che stia leggendo la stessa `calc/` aggiornata da TraceWin. Nel
notebook sono stampati i path:

```text
CALC_DIR
DST_INPUT
DST_OUTPUT
PARTRAN_OUT
```

Se i plot sono vecchi, fai `Run All` invece di rilanciare solo una cella.

### Non vedo i file intermedi .dst

Possibili cause:

- non hai inserito `PLOT_DST` nel lattice;
- hai lanciato solo envelope e non PARTRAN/multiparticle;
- TraceWin non ha completato il run;
- stai guardando una cartella `calc/` diversa.

### SAVE_BEAM da errore

In questo setup `SAVE_BEAM` non e' un comando riconosciuto nel lattice TraceWin.
Usa:

```text
PLOT_DST 1
PLOT_DST 2
...
```

### Input .dst e particelle simulate non coincidono

`16O5.dst` puo' contenere molte piu' particelle del numero effettivamente
tracciato nel run. Per esempio, il file input puo' essere un reservoir grande,
mentre PARTRAN simula solo `NPART` particelle.

Quindi:

- `16O5.dst` = distribuzione/reservoir iniziale completa;
- `NPART` in `partran1.out` = particelle effettivamente simulate;
- `part_dtl1.dst` = particelle sopravvissute alla fine.

Per calcolare perdite o trasmissione, usa il numero di particelle simulate come
riferimento, non necessariamente il numero totale nel reservoir.

