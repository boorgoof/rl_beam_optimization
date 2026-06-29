# Schema Generale Del Progetto `beam_optimization`

Questo documento descrive a parole il diagramma globale:

```text
beam_optimization/PROJECT_CLASS_DIAGRAM.drawio
```

Il diagramma riguarda solo il package `beam_optimization`. Non include cartelle
esterne al package.

## Idea Generale

Il progetto ottimizza i parametri della linea ADIGE usando ambienti Gymnasium,
simulazioni TraceWin, surrogate neurali e algoritmi di ottimizzazione/RL.

Il flusso mentale piu importante e:

```text
script o CLI
  -> carica configurazione, dataset e surrogate
      -> crea ambiente
          -> crea agente o baseline
              -> agente sceglie delta-parametri
                  -> ambiente simula
                      -> BeamSimulationResult
                          -> reward / score / output
```

Gli algoritmi non parlano direttamente con TraceWin. Parlano con un ambiente.
L'ambiente poi decide se sotto usare TraceWin reale o il surrogate.

## Blocchi Del Diagramma

Il diagramma globale e organizzato in questi blocchi:

```text
CLI/scripts
configuration
environments and simulators
surrogate model and data
algorithms
files and outputs
```

Ogni blocco rappresenta una responsabilita diversa.

## CLI E Scripts

Il punto di ingresso principale e:

```text
beam_optimization.main
```

Espone il comando:

```bash
python -m beam_optimization <command>
```

I comandi disponibili sono:

```text
check
train
evaluate
benchmark
```

`main.py` non contiene logica scientifica: fa solo dispatch verso i moduli in
`scripts/`.

### `scripts.train`

Allena gli agenti.

Puo creare:

- agenti model-free custom, come `SAC`, `TD3`, `PPO`, `DDPG`, `A2C`,
  `REINFORCE`, `TRPO`;
- `SB3SAC`, cioe il wrapper di Stable Baselines3;
- `SVGAgent`;
- `MBPO` / `MBPOWithModelUpdate`.

Il training standard usa `SurrogateEnv`, perche e molto piu veloce di TraceWin.
Quando serve feedback reale o fine-tuning online del modello, puo usare anche
`TraceWinEnv`.

### `scripts.evaluate`

Carica un ambiente e un agente salvato, poi valuta la policy.

Si occupa di:

- caricare surrogate e dataset;
- costruire l'ambiente;
- caricare l'agente;
- eseguire episodi;
- salvare risultati JSON o render diagnostici.

### `scripts.benchmark`

Confronta piu metodi sullo stesso problema.

Nel diagramma e collegato sia agli agenti RL sia alle baseline classiche:

- `PSOOptimizer`;
- `BayesianOptimizer`;
- `SVGAgent`;
- agenti model-free salvati.

### `scripts.check`

E uno smoke test del progetto.

Controlla che dataset, surrogate, ambiente e agenti principali siano
importabili/istantiabili. Non e il training vero: serve a capire rapidamente se
la pipeline di base e ancora sana.

## Configurazione

La configurazione vive in `config/`.

### `config.adige`

E il file piu centrale del progetto.

Definisce:

- le feature del fascio;
- la dimensione dello stato del fascio;
- i 16 parametri controllabili;
- i marker/stage TraceWin;
- i bound delle azioni;
- la conversione dizionario/vettore parametri;
- la conversione parametri/stage tensor;
- le funzioni di score.

Quasi tutti i blocchi dipendono da `config.adige`, perche devono concordare su:

```text
dimensione osservazioni
dimensione azioni
ordine dei parametri
struttura degli stage
funzione obiettivo
```

### `ParameterSpec`

E la dataclass che descrive un parametro controllabile:

```text
name
key
marker
default
sensitivity
```

`config.adige` usa queste specifiche per costruire vettori parametrici coerenti.

### `config.paths`

Contiene i path di default:

- dataset;
- cartella surrogate;
- output di training;
- output benchmark;
- file TraceWin.

Gli script lo usano per avere default sensati senza hardcodare path in ogni
funzione.

## Ambienti E Simulatori

Gli ambienti sono il ponte tra algoritmi e simulazioni fisiche.

### `BaseBeamEnv`

E la classe base Gymnasium comune.

Gestisce:

- `observation_space`;
- `action_space`;
- `reset()`;
- `step(action)`;
- clipping/applicazione delle azioni;
- reward;
- `best_score`;
- `best_params`;
- conversione del risultato di simulazione in osservazione.

Il reward e:

```text
score_nuovo - score_precedente
```

`BaseBeamEnv` non sa se sotto ci sia TraceWin o il surrogate. Conosce solo un
oggetto che implementa `BeamSimulator`.

### `BeamSimulator`

E il contratto astratto dei simulatori:

```python
reset_context(rng=None)
simulate(params) -> BeamSimulationResult
```

Le due implementazioni principali sono:

- `TraceWinSimulator`;
- `SurrogateBeamSimulator`.

### `BeamSimulationResult`

E il formato comune di ritorno di ogni simulazione.

Contiene:

- parametri usati;
- stati del fascio;
- score;
- successo/fallimento;
- sorgente del dato;
- fascio finale;
- metadata.

Il campo `source` e molto importante:

```text
source="tracewin"    dato fisico reale
source="surrogate"  predizione del modello
```

## Backend TraceWin

### `TraceWinEnv`

E l'ambiente Gymnasium che usa TraceWin reale.

Eredita da `BaseBeamEnv` e monta un `TraceWinSimulator`.

In piu puo renderizzare diagnostiche di phase space leggendo i file prodotti da
TraceWin.

### `TraceWinSimulator`

E il backend fisico reale.

Fa:

1. prepara i parametri;
2. prepara la cartella di calcolo;
3. lancia TraceWin tramite wrapper;
4. legge gli output;
5. estrae gli stati del fascio;
6. calcola lo score;
7. restituisce `BeamSimulationResult(source="tracewin")`.

Se TraceWin fallisce, ritorna comunque un `BeamSimulationResult` strutturato
con `success=False`, cosi gli algoritmi non devono gestire eccezioni sparse.

### `TraceWin`

E il wrapper tecnico dell'eseguibile TraceWin.

Espone metodi come:

```python
run(...)
results()
dst(...)
plt()
```

### `Dst / Plt / file readers`

Sono classi di lettura dei file TraceWin.

Non conoscono RL, reward o policy. Servono solo a leggere file fisici prodotti
dalla simulazione.

## Backend Surrogate

### `SurrogateEnv`

E l'ambiente Gymnasium veloce.

Eredita da `BaseBeamEnv` e monta un `SurrogateBeamSimulator`.

Da fuori ha la stessa forma di `TraceWinEnv`, quindi gli agenti possono essere
allenati sul surrogate e poi valutati nello stesso schema generale.

### `SurrogateBeamSimulator`

E il simulatore neurale.

Contiene:

- uno o piu `ModularMLP`;
- un `BeamDataset`;
- la logica per campionare `beam0`;
- la logica per scegliere il modello attivo dell'ensemble.

Produce:

```text
BeamSimulationResult(source="surrogate")
```

Espone anche `forward_differentiable()`, usato da `SVGAgent` per fare
backpropagation attraverso il modello fisico neurale.

### `run_surrogate_forward`

E una funzione helper che esegue il forward del surrogate senza gradiente e
ricostruisce:

```text
beam_states
final_beam
score
```

### `ModularMLP`

E la rete neurale che approssima TraceWin.

La sua struttura segue gli stage della linea:

```text
beam0 + parametri per stage -> stati successivi del fascio
```

Viene usata da:

- `SurrogateBeamSimulator`;
- `SurrogateDatasetUpdater`;
- `MBPO`;
- `SVGAgent`;
- baseline e script di benchmark.

### `BeamDataset`

Vive in `env/dataset/dataset.py` e mantiene il dataset flat:

```text
X      beam0 + parametri
Y      stati del fascio successivi
scores score finale
```

Lo stesso dataset serve per:

- allenare il surrogate;
- campionare `beam0`;
- generare rollout sintetici MBPO;
- fornire batch al fine-tuning online.

`SurrogateTrainingDataset` resta un alias legacy di `BeamDataset`.
Non esiste piu un `dataset.py` dentro `surrogate/`: gli import devono passare
da `beam_optimization.env.dataset`.

### `TraceWinDatasetBuilder` e `utility.py`

`env/dataset/tracewin_dataset_builder.py` crea dataset offline nuovi usando
TraceWin reale. Non appende a file `.pt` esistenti.

Flusso:

```text
parametri
  -> TraceWinSimulator.simulate(params)
  -> BeamSimulationResult(source="tracewin")
  -> utility.tracewin_result_to_flat_sample(...)
  -> BeamDataset
  -> env/dataset/001/dataset_train.pt
  -> env/dataset/001/dataset_val.pt
  -> env/dataset/001/dataset_test.pt
```

`env/dataset/utility.py` contiene la conversione comune
`BeamSimulationResult -> x/y/score`, usata sia dal builder offline sia
dall'updater online.

`env/dataset/base/dataset_train.pt` e il dataset base usato per campionare
`beam0`. Quando MBPO online salva il merged dataset senza override esplicito,
aggiorna questo stesso file. I dataset offline creati da TraceWin vengono invece
salvati in cartelle numerate sotto `env/dataset`, come `001`, `002`, ...

### `SurrogateDatasetUpdater`

Riceve nuovi risultati TraceWin, li converte in righe flat per `BeamDataset`,
mantiene il dataset online e aggiorna uno o piu surrogate.

La regola chiave e:

```text
aggiorna solo con BeamSimulationResult(source="tracewin")
```

Questo evita di addestrare il modello sulle sue stesse predizioni.

`collector.py` non e piu parte del flusso ufficiale per creare o aggiornare
file `.pt` flat.

### `SurrogateEvaluator`

`surrogate/evaluator.py` valuta tutti i checkpoint `surrogate_*.pt` di una
cartella su un dataset validation/test.

Calcola:

```text
mse_all / rmse_all
mse_final_stage / rmse_final_stage
mse_per_stage / rmse_per_stage
n_samples
```

## Algoritmi

Gli algoritmi vivono in `algorithms/`.

Nel diagramma sono divisi in:

- agenti on-policy;
- agenti off-policy;
- reti neurali condivise;
- buffer e utility;
- algoritmi model-based;
- baseline classiche.

### Agenti On-Policy

Gruppo:

```text
REINFORCE / A2C / PPO / TRPO
```

Usano traiettorie fresche raccolte dall'ambiente e un `EpisodeBuffer`.

Tipicamente usano:

```text
stocActor = GaussianPolicyNetwork
ValueNetwork
EpisodeBuffer
```

Sono on-policy perche aggiornano la policy usando dati raccolti dalla policy
corrente o quasi corrente.

### Agenti Off-Policy

Gruppo:

```text
DDPG / TD3 / SAC
```

Usano `ReplayBuffer`, quindi possono riutilizzare transizioni vecchie.

`DDPG` e `TD3` usano:

```text
detActor = DeterministicPolicyNetwork
critic Q
ReplayBuffer
```

`SAC` usa:

```text
stocActor = GaussianPolicyNetwork
due critic Q
ReplayBuffer
entropy tuning
```

### `SB3SAC`

E un wrapper intorno a `stable_baselines3.SAC`.

Ha API piu alta:

```python
train(env, n_steps)
select_action(obs)
save(path)
load(path, env)
```

Non espone lo stesso ciclo interno degli agenti custom, perche SB3 gestisce da
solo replay buffer e ottimizzazione.

### Reti

Il blocco `Policy networks` contiene:

```text
GaussianPolicyNetwork  = stocActor
DeterministicPolicyNetwork = detActor
```

Il blocco `Value networks` contiene:

```text
ValueNetwork
QNetwork
TwinQNetwork
```

### Buffer E Utility

Il blocco `RL buffers` contiene:

```text
EpisodeBuffer
ReplayBuffer
MixedReplayBuffer
```

Il blocco `algorithm utils` contiene:

```text
NormalNoiseDecayStrategy
trpo_utils
Logger
```

## Algoritmi Model-Based

### `MBPO`

`MBPO` usa un agente interno off-policy, tipicamente il `SAC` custom, e lo
allena su un mix di transizioni reali e sintetiche.

Schema:

```text
transizione reale -> real_buffer
surrogate rollout -> synth_buffer
batch misto -> agent.optimize()
```

Per questo nel diagramma `MBPO` punta a:

- `OffPolicyAgent`;
- `MixedReplayBuffer`;
- `SurrogateBeamSimulator`.

### `MBPOWithModelUpdate`

Estende `MBPO`.

In piu raccoglie risultati reali TraceWin e li usa per fine-tunare il surrogate
online.

Schema:

```text
TraceWinEnv -> BeamSimulationResult(source="tracewin")
  -> online dataset
      -> fine-tuning ensemble surrogate
```

### `SVGAgent`

Usa il surrogate come modello differenziabile.

Invece di imparare un critic, fa passare il gradiente attraverso:

```text
policy -> action -> parametri -> surrogate -> score
```

Restituisce `SVGResult`, che contiene loss, score finale, storico score e norma
del gradiente.

## Baseline

Le baseline non imparano una policy.

Ottimizzano direttamente i parametri:

```text
params -> objective(params) -> score
```

Nel progetto ci sono:

- `PSOOptimizer`;
- `BayesianOptimizer`.

Sono utili come confronto con gli agenti RL.

## File E Output

Il blocco in basso del diagramma mostra gli artefatti principali.

### `flat dataset .pt`

Contiene:

```text
X
Y
scores
metadata
```

Viene letto e scritto da `BeamDataset`; il flusso applicativo di aggiornamento
passa da `SurrogateDatasetUpdater` e puo aggiornare il dataset base usato per
`beam0`, mentre la creazione offline da zero passa da `TraceWinDatasetBuilder`
e usa cartelle numerate.

### `surrogate_*.pt`

Checkpoint dei `ModularMLP`.

Possono essere:

- `models/base`: surrogate originali conservati come riferimento pulito;
- `models/updated`: surrogate di lavoro, usati di default se presenti e
  aggiornati online.

### `agent checkpoints`

Checkpoint degli agenti:

```text
sac_agent.pt
td3_agent.pt
ppo_agent.pt
...
sb3_sac_agent.zip
svg_agent.pt
dyna_agent.pt
```

### `results / renders`

Output di valutazione e benchmark:

- JSON;
- PNG diagnostici;
- metriche;
- eventuali log.

## Flussi Principali

### Training Su Surrogate

```text
scripts.train
  -> carica ModularMLP e BeamDataset
  -> crea SurrogateEnv
  -> crea agente
  -> agente interagisce con env
  -> salva checkpoint agente
```

### Valutazione

```text
scripts.evaluate
  -> carica env
  -> carica agente
  -> run_episode
  -> salva score/render/output
```

### Benchmark

```text
scripts.benchmark
  -> crea ambiente/obiettivo
  -> esegue agenti e baseline
  -> produce tabella e risultati
```

### MBPO

```text
env reale o surrogate
  -> transizioni reali
      -> MBPO real_buffer
surrogate ensemble
  -> rollout sintetici
      -> MBPO synth_buffer
batch misto
  -> SAC/TD3-like optimize
```

### Fine-Tuning Online Del Surrogate

```text
TraceWinEnv
  -> BeamSimulationResult(source="tracewin")
      -> MBPOWithModelUpdate / SurrogateDatasetUpdater
          -> dataset online
              -> update ModularMLP
                  -> salva dataset/checkpoint aggiornati
```

### Creazione Dataset Offline TraceWin

```text
TraceWinDatasetBuilder
  -> genera o riceve parametri
  -> TraceWinSimulator.simulate(params)
  -> utility.tracewin_result_to_flat_sample(result)
  -> BeamDataset
  -> salva env/dataset/001/dataset_train.pt / dataset_val.pt / dataset_test.pt
```

### Evaluation Dei Surrogate

```text
surrogate/evaluator.py
  -> carica dataset_val.pt o dataset_test.pt
  -> carica tutti i surrogate_*.pt in una cartella
  -> calcola MSE/RMSE aggregati, final stage e per stage
  -> opzionalmente salva surrogate_evaluation.json
```

## Come Leggere Le Frecce

Nel diagramma:

- triangolo vuoto = ereditarieta Python;
- triangolo vuoto tratteggiato = implementazione di interfaccia astratta;
- diamante nero = composizione/oggetto posseduto;
- diamante bianco = aggregazione/riferimento condiviso;
- freccia aperta = dipendenza, chiamata o ritorno;
- freccia tratteggiata = dipendenza debole, configurazione o tipo.

Esempi:

```text
TraceWinEnv -> BaseBeamEnv
SurrogateEnv -> BaseBeamEnv
TraceWinSimulator -> BeamSimulator
SurrogateBeamSimulator -> BeamSimulator
BaseBeamEnv -> BeamSimulationResult
MBPO -> SurrogateBeamSimulator
SVGAgent -> SurrogateBeamSimulator
TrainScript -> AgentCheckpoints
```

## Regola Mentale Finale

- `scripts/` decide cosa eseguire.
- `config/` definisce dimensioni, parametri e score.
- `env/` trasforma azioni in simulazioni e reward.
- `surrogate_env/` rende la simulazione veloce e differenziabile.
- `tracewin_env/` produce dati fisici reali.
- `algorithms/` sceglie le azioni o ottimizza direttamente i parametri.
- gli artefatti `.pt`, `.zip`, `.json`, `.png` sono dati, modelli e risultati
  prodotti dalla pipeline.
