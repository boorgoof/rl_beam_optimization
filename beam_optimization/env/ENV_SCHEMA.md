# Schema Della Cartella `env`

Questa cartella contiene tutto cio che trasforma il problema fisico della linea
ADIGE in un ambiente Gymnasium usabile dagli algoritmi RL.

L'idea centrale e semplice:

```text
Agente RL
  -> ambiente Gymnasium
      -> simulatore fisico
          -> BeamSimulationResult
      -> obs, reward, terminated, truncated, info
```

Il codice mantiene separati due livelli:

- il **ciclo RL comune**, cioe reset, step, reward, action space e observation
  space;
- il **motore fisico**, che puo essere TraceWin reale oppure il surrogate
  neurale.

Per questo `TraceWinEnv` e `SurrogateEnv` hanno la stessa forma esterna per
gli algoritmi. Cambia solo il simulatore montato sotto.

## Struttura Logica

```text
beam_optimization/env/
├── simulation.py
├── base_beam_env.py
├── dataset/
│   ├── dataset.py
│   ├── utility.py
│   └── tracewin_dataset_builder.py
├── tracewin_env/
│   ├── tracewin_env.py
│   └── tracewin/
│       ├── tracewin_simulator.py
│       └── pyTraceWin_wrapper/
│           ├── tracewin.py
│           └── files.py
└── surrogate_env/
    ├── surrogate_env.py
    ├── surrogate_simulator.py
    └── surrogate/
        ├── modular_mlp.py
        ├── evaluator.py
        └── updater.py
```

`config/adige.py` non sta dentro `env`, ma e una dipendenza centrale: definisce
dimensioni, parametri controllabili, marker TraceWin, bound delle azioni,
conversione dei parametri e funzione di score.

## Contratto Comune: `simulation.py`

`simulation.py` contiene due pezzi comuni a entrambi i backend.

`BeamSimulationResult` e il contenitore standard prodotto da una simulazione.
Contiene:

```text
params       parametri macchina usati nella simulazione
beam_states  array degli stati del fascio, normalmente shape (12, 9)
score_val    score scalare del fascio finale
success      True se la simulazione e valida
source       "tracewin" oppure "surrogate"
error        messaggio di errore se la simulazione fallisce
final_beam   dizionario delle 9 feature finali
metadata     dettagli specifici del backend
timestamp    momento di creazione del risultato
```

Il campo piu importante e' `source`: impedisce di confondere dati fisici reali
con predizioni del surrogate.

`BeamSimulator` e' il contratto astratto dei simulatori. Ogni simulatore deve
esporre:

```python
reset_context(rng=None)
simulate(params) -> BeamSimulationResult
```

`reset_context()` prepara il contesto dell'episodio. TraceWin quasi non ne ha
bisogno, perche il fascio iniziale vive nei file progetto; il surrogate invece
lo usa per scegliere `beam0` e modello dell'ensemble.

## Ambiente Comune: `BaseBeamEnv`

`BaseBeamEnv` e la base Gymnasium condivisa. Eredita da `gymnasium.Env` e
gestisce tutto quello che non dipende dal motore fisico:

- `observation_space`;
- `action_space`;
- campionamento dei parametri iniziali a `reset()`;
- applicazione dell'azione come delta sui 16 parametri;
- chiamata a `self.simulator.simulate(params)`;
- conversione del `BeamSimulationResult` in osservazione;
- calcolo del reward;
- aggiornamento di `best_score` e `best_params`;
- render comune delle feature del fascio.

Il reward e:

```text
reward = score_nuovo - score_precedente
```

Gli episodi non hanno uno stato terminale fisico esplicito. Per questo
`terminated` e sempre `False`, mentre `truncated` diventa `True` quando
`_step_count >= max_steps`.

Le modalita di osservazione sono:

```text
obs_mode="full"              -> 12 stage * 9 feature = 108 valori
obs_mode="final"             -> solo fascio finale = 9 valori
obs_mode="final_with_beam0"  -> fascio iniziale + finale = 18 valori
```

`BaseBeamEnv` e astratta: non puo essere istanziata da sola. Le sottoclassi
devono implementare:

```python
_build_simulator() -> BeamSimulator
```

Il costruttore comune chiama `_build_simulator()`, salva il risultato in
`self.simulator` e verifica che sia davvero un `BeamSimulator`. In questo modo
un ambiente senza backend fisico fallisce subito all'istanziazione, non piu al
primo `reset()` o `step()`.

## Backend TraceWin

### `TraceWinEnv`

`TraceWinEnv` e l'ambiente Gymnasium che usa TraceWin reale. Non riscrive il
ciclo RL: implementa `_build_simulator()` restituendo un `TraceWinSimulator` e
poi usa il ciclo comune di `BaseBeamEnv`.

In piu, rispetto al render comune, puo visualizzare lo spazio delle fasi finale
leggendo i file `.dst` scritti da TraceWin. Queste immagini sono diagnostica:
non sono parte dell'osservazione RL.

### `TraceWinSimulator`

`TraceWinSimulator` e il backend fisico reale. Fa questo lavoro:

1. riceve un dizionario di parametri;
2. completa i parametri mancanti con i default;
3. pulisce e ricrea `calc_dir`;
4. opzionalmente prepara una copia locale della workspace TraceWin;
5. crea il wrapper `TraceWin`;
6. lancia il binario TraceWin tramite il launcher;
7. legge `partran1.out`;
8. estrae gli stati del fascio ai marker definiti in `config/adige.py`;
9. calcola lo score del fascio finale;
10. restituisce `BeamSimulationResult(source="tracewin")`.

Se TraceWin fallisce, il simulatore restituisce comunque un risultato
strutturato:

```text
success=False
source="tracewin"
score_val=-999.0
error=<messaggio>
```

Questo evita che il resto del codice debba gestire eccezioni sparse.

### `pyTraceWin_wrapper`

`pyTraceWin_wrapper/tracewin.py` contiene la classe tecnica `TraceWin`.
Il suo compito e lanciare l'eseguibile e leggere i file prodotti.

Espone:

```python
run(timeout, elem_params, other_params={}, num_threads=None)
results() -> DataFrame
dst(out=True) -> Dst
plt() -> Plt
```

`files.py` contiene classi di lettura dei formati TraceWin, soprattutto `Dst`
e `Plt`. Queste classi non sanno nulla di RL, reward o score: servono solo a
leggere file.

## Backend Surrogate

### `SurrogateEnv`

`SurrogateEnv` e l'ambiente Gymnasium veloce. Come `TraceWinEnv`, non riscrive
il ciclo RL: implementa `_build_simulator()` restituendo un
`SurrogateBeamSimulator` e poi delega a `BaseBeamEnv`.

Si usa per training, benchmark e rollout sintetici. Non lancia TraceWin e non
scrive file TraceWin.

### `SurrogateBeamSimulator`

`SurrogateBeamSimulator` e la controparte neurale di `TraceWinSimulator`.

Riceve:

- un singolo `ModularMLP` oppure una lista di `ModularMLP`;
- un `BeamDataset`;
- una modalita di campionamento di `beam0`, cioe `"dataset"` o `"gaussian"`.

A ogni reset:

1. sceglie un modello dell'ensemble;
2. campiona un fascio iniziale `beam0`;
3. mantiene quel contesto per l'episodio.

Quando deve simulare:

1. converte i 16 parametri in tensori per stage;
2. chiama il `ModularMLP`;
3. ricostruisce `beam_states` come `beam0 + 11 output`;
4. calcola `final_beam` e score;
5. restituisce `BeamSimulationResult(source="surrogate")`.

Il metodo `forward_differentiable()` esiste per algoritmi che devono
retropropagare attraverso il surrogate, come `SVGAgent`.

### `run_surrogate_forward`

`run_surrogate_forward()` e una funzione helper usata dal simulatore surrogate
per il percorso senza gradiente. Restituisce:

```text
beam_states, final_beam, score_val
```

E utile per tenere separata la logica di forward della rete dalla costruzione
del risultato di simulazione.

## Modello Surrogate: `ModularMLP`

`ModularMLP` e la rete neurale che approssima TraceWin. Segue la struttura a
stage della linea.

Input:

```text
beam_state_0 : Tensor (batch, 9)
stage_params : lista di 11 Tensor, uno per stage
```

Output:

```text
single_output=False -> lista di 11 stati predetti
single_output=True  -> solo stato finale
```

Internamente ha:

- `input_net`, che costruisce il primo stato latente;
- `stage_nets`, una rete per propagare il latente stage per stage;
- `output_nets`, una rete per produrre lo stato del fascio a ogni stage;
- normalizzazione opzionale dei parametri e degli stati del fascio.

Il modello puo essere salvato e caricato con:

```python
model.save(path)
ModularMLP.load(path)
```

## Dataset Del Surrogate

`BeamDataset` vive in `env/dataset/dataset.py`. Mantiene i dati nel formato
flat usato su disco e li converte al volo nel formato stage-wise richiesto da
`ModularMLP`.

Formato flat:

```text
X:      (N, 25) = beam0(9) + parametri(16)
Y:      (N, 99) = 11 stati successivi * 9 feature
scores: (N,)    = score finale del campione
```

Metodi principali:

- `append_flat_sample(x, y, score)`: aggiunge un campione gia convertito;
- `append_flat_samples(X, Y, scores)`: aggiunge un batch gia convertito;
- `get_training_batch(indices)`: produce `stage_params` e `beam_states`;
- `get_initial_beam_states()`: restituisce tutti i `beam0`;
- `get_param_vecs()`: restituisce tutti i vettori parametri;
- `load(path)`: carica dataset flat o legacy;
- `merge(other)`: unisce due dataset;
- `save_flat(path)`: salva in formato flat.

`SurrogateTrainingDataset` e un alias legacy di `BeamDataset`, ma vive nel
package comune `env.dataset`. Non esiste piu un `dataset.py` dentro
`surrogate/`: gli import devono passare da `beam_optimization.env.dataset`.

### `utility.py`

`env/dataset/utility.py` contiene la conversione condivisa:

```python
tracewin_result_to_flat_sample(result) -> x, y, score
```

Questa funzione e l'unico punto comune che trasforma un
`BeamSimulationResult(source="tracewin")` in:

```text
x:     (25,) = beam0(9) + parametri(16)
y:     (99,) = 11 output stage * 9 feature
score: scalare finale
```

La usano sia il builder offline sia `SurrogateDatasetUpdater`.

## Creazione Dataset Offline Da TraceWin

`TraceWinDatasetBuilder` vive in `env/dataset/tracewin_dataset_builder.py`.
Serve per creare dataset nuovi da zero usando TraceWin reale.

Flusso:

```text
parametri
  -> TraceWinSimulator.simulate(params)
  -> BeamSimulationResult(source="tracewin")
  -> tracewin_result_to_flat_sample(...)
  -> BeamDataset
  -> env/dataset/001/dataset_train.pt
  -> env/dataset/001/dataset_val.pt
  -> env/dataset/001/dataset_test.pt
```

Di default puo salvare split:

```text
train = 80%
val   = 10%
test  = 10%
```

Il builder non appende a dataset esistenti: ogni chiamata costruisce un nuovo
`BeamDataset` in memoria e salva nuovi file `.pt`. Se non viene passato un
`output_dir`, crea automaticamente la prossima cartella numerata sotto
`env/dataset`, per esempio `001`, `002`, `003`, ignorando `base`.

## Aggiornamento Del Surrogate

`SurrogateDatasetUpdater` e il punto unico che riceve nuovi risultati TraceWin,
li aggiunge al dataset online e fine-tuna uno o piu `ModularMLP`.

La regola critica e:

```python
if result.source != "tracewin":
    return False
```

Quindi il surrogate viene aggiornato solo con dati fisici reali, mai con le
sue stesse predizioni.

Quando ci sono abbastanza campioni:

1. usa `utility.tracewin_result_to_flat_sample()` per ottenere `X`, `Y`, `score`;
2. aggiunge il campione solo al dataset online;
3. campiona batch bootstrap dal dataset online o da offline+online;
4. aggiorna ogni surrogate con il proprio ottimizzatore Adam persistente;
5. puo salvare solo i dati online o il dataset merged offline+online; nel
   training MBPO online il default e salvare il merged sul dataset base caricato
   per `beam0`, cioe `env/dataset/base/dataset_train.pt`;
6. puo salvare i pesi aggiornati come `surrogate_0.pt`, `surrogate_1.pt`, ...

`collector.py` non e piu parte del flusso ufficiale.

## Valutazione Dei Surrogate

`surrogate/evaluator.py` misura la qualita dei checkpoint `surrogate_*.pt` su
un dataset flat, tipicamente validation o test.

Espone:

```python
evaluate_surrogate(model, dataset, batch_size=1024, device=None)
evaluate_surrogate_folder(model_dir, dataset_path, batch_size=1024, device=None)
```

Per ogni modello calcola:

```text
mse_all / rmse_all
mse_final_stage / rmse_final_stage
mse_per_stage / rmse_per_stage
n_samples
```

## Cosa Scrive Su Disco

| Componente | Scrive? | Cosa |
| --- | --- | --- |
| `BaseBeamEnv` | No | Stato episodio solo in memoria |
| `TraceWinSimulator` | Si | Output TraceWin in `calc_dir` |
| `TraceWinSimulator` | Si, se cache attiva | Copia locale della workspace |
| `TraceWin` wrapper | Indirettamente | Fa scrivere TraceWin in `calc_dir` |
| `SurrogateEnv` | No | Usa modello e dataset in RAM |
| `SurrogateBeamSimulator` | No | Produce risultati in RAM |
| `BeamDataset.save_flat` | Si | Dataset `.pt` flat |
| `TraceWinDatasetBuilder` | Si | Nuovi dataset train/val/test in cartelle numerate |
| `ModularMLP.save` | Si | Checkpoint del modello |
| `SurrogateDatasetUpdater.save_online_dataset` | Si | Solo nuovi campioni TraceWin |
| `SurrogateDatasetUpdater.save_merged_dataset` | Si | Dataset offline + online |
| `SurrogateDatasetUpdater.save_surrogates` | Si | Pesi surrogate fine-tunati in `models/updated` |
| `SurrogateEvaluator` | Opzionale | JSON con MSE/RMSE |

## Flusso Di `reset()`

```text
BaseBeamEnv.reset()
  -> campiona parametri iniziali
  -> simulator.reset_context(rng)
  -> simulator.simulate(params)
  -> BeamSimulationResult
  -> osservazione + score + info
```

Nel backend surrogate, `reset_context()` sceglie anche `beam0` e modello
dell'ensemble. Nel backend TraceWin, il fascio iniziale e definito dal progetto.

## Flusso Di `step(action)`

```text
BaseBeamEnv.step(action)
  -> clip dell'azione nei bound
  -> params = params + action
  -> simulator.simulate(params)
  -> BeamSimulationResult
  -> obs = slice(beam_states, obs_mode)
  -> reward = score_nuovo - score_precedente
  -> aggiorna best_score e best_params
  -> ritorna obs, reward, False, truncated, info
```

## Come Leggere Il Diagramma Di Classe

Il file modificabile e:

```text
beam_optimization/env/ENV_CLASS_DIAGRAM.drawio
```

Il diagramma e diviso in blocchi:

- **common env contract**: `BaseBeamEnv`, `BeamSimulator`,
  `BeamSimulationResult`, `AdigeConfig`;
- **TraceWin backend**: `TraceWinEnv`, `TraceWinSimulator`, `TraceWin`,
  `Dst/Plt`;
- **surrogate backend**: `SurrogateEnv`, `SurrogateBeamSimulator`,
  `run_surrogate_forward`, `ModularMLP`;
- **data pipeline**: `BeamDataset`, `TraceWinDatasetBuilder`, `utility.py`,
  `SurrogateDatasetUpdater`, `SurrogateEvaluator`, file flat `.pt`.

Le frecce principali sono:

| Soggetto | Relazione | Oggetto | Significato |
| --- | --- | --- | --- |
| `TraceWinEnv` | eredita da | `BaseBeamEnv` | Stesso ciclo Gym comune |
| `SurrogateEnv` | eredita da | `BaseBeamEnv` | Stesso ciclo Gym comune |
| `TraceWinEnv` | possiede | `TraceWinSimulator` | Backend reale |
| `SurrogateEnv` | possiede | `SurrogateBeamSimulator` | Backend neurale |
| `TraceWinSimulator` | implementa | `BeamSimulator` | Simulatore reale |
| `SurrogateBeamSimulator` | implementa | `BeamSimulator` | Simulatore neurale |
| `TraceWinSimulator` | produce | `BeamSimulationResult` | `source="tracewin"` |
| `SurrogateBeamSimulator` | produce | `BeamSimulationResult` | `source="surrogate"` |
| `SurrogateBeamSimulator` | usa | `ModularMLP` | Forward del modello |
| `SurrogateBeamSimulator` | usa | `BeamDataset` | Campiona `beam0` |
| `TraceWinDatasetBuilder` | usa | `TraceWinSimulator` | Crea dataset offline |
| `TraceWinDatasetBuilder` | scrive | `BeamDataset` | Nuovi split train/val/test |
| `SurrogateDatasetUpdater` | aggiorna | `ModularMLP` | Fine-tuning con dati reali |
| `utility.py` | converte | `BeamSimulationResult` | Crea righe `X/Y/scores` |
| `SurrogateEvaluator` | valuta | `ModularMLP` | MSE/RMSE su dataset |

## Regola Mentale

Se lavori sul ciclo RL, parti da `BaseBeamEnv`.

Se lavori con TraceWin reale, guarda `TraceWinEnv`, `TraceWinSimulator` e
`pyTraceWin_wrapper`.

Se lavori con il surrogate, guarda `SurrogateEnv`, `SurrogateBeamSimulator`,
`ModularMLP`, `BeamDataset` e `surrogate/evaluator.py`.

Se vuoi creare dataset offline nuovi da TraceWin, guarda
`env/dataset/tracewin_dataset_builder.py`.

Se lavori su nuovi dati reali per migliorare il surrogate, guarda
`SurrogateDatasetUpdater`.

Se cambi dimensioni, parametri, marker, action bounds o score, guarda
`config/adige.py`.
