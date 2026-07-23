# Schema Della Cartella `algorithms`

Questa cartella contiene gli algoritmi che scelgono come modificare i 18
parametri della linea ADIGE (definiti in `config/adige.py`, `PARAMETERS`) a
partire dall'osservazione prodotta dagli ambienti in `beam_optimization/env`.

L'idea centrale e questa:

```text
osservazione del fascio
  -> policy / ottimizzatore
      -> azione = delta sui parametri
          -> ambiente
              -> nuovo BeamSimulationResult
                  -> reward assoluto limitato nei particle loss,
                     altrimenti score_nuovo / REWARD_SCORE_SCALE
```

Gli algoritmi non devono sapere se sotto l'ambiente c'e TraceWin reale oppure
il surrogate neurale. Vedono solo la normale API Gymnasium:

```python
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(action)
```

## Struttura Logica

```text
beam_optimization/algorithms/
├── model_free/
│   ├── reinforce.py
│   ├── a2c.py
│   ├── ppo.py
│   ├── trpo.py
│   ├── ddpg.py
│   ├── td3.py
│   ├── sac.py
│   └── sb3_sac.py
├── model_based/
│   ├── mbpo.py
│   ├── mbpo_model_update.py
│   └── svg.py
├── baselines/
│   └── bayesian_opt.py
├── networks/
│   ├── policy_nets.py
│   └── value_nets.py
└── utils/
    ├── episode_buffer.py
    ├── replay_buffer.py
    ├── noise.py
    ├── trpo_utils.py
    └── logger.py
```

La cartella e divisa in cinque blocchi:

- `model_free`: algoritmi RL che imparano solo dall'interazione con l'ambiente;
- `model_based`: algoritmi che usano anche il surrogate come modello del mondo;
- `baselines`: ottimizzatori classici senza policy neurale;
- `networks`: reti neurali condivise dagli agenti;
- `utils`: buffer, rumore, logging e funzioni tecniche.

## Concetti Comuni

Tutti gli agenti RL lavorano su spazi continui:

```text
obs_dim = dimensione osservazione ambiente
act_dim = N_PARAMS (18, da config/adige.py)
azione  = variazione dei parametri controllabili
```

Nel progetto esistono due famiglie principali di policy.

`stocActor` e il nome concettuale usato in questo schema per
`GaussianPolicyNetwork`. E una policy stocastica gaussiana:

```text
stocActor(obs) -> distribuzione Normale -> azione campionata
```

Viene usata da algoritmi che devono esplorare tramite campionamento o calcolare
log-probabilita delle azioni:

- `REINFORCE`;
- `A2C`;
- `PPO`;
- `TRPO`;
- `SAC`;
- `SVGAgent`.

`detActor` e il nome concettuale usato in questo schema per
`DeterministicPolicyNetwork`. E una policy deterministica:

```text
detActor(obs) -> azione diretta
```

Viene usata da:

- `DDPG`;
- `TD3`.

La distinzione e importante: negli algoritmi stocastici la rete rappresenta una
distribuzione di azioni, mentre negli algoritmi deterministici rappresenta
direttamente il comando scelto.

## Reti Neurali: `networks`

### `GaussianPolicyNetwork`

Questa e la rete indicata in questo schema come `stocActor`.

Dato uno stato produce:

```text
mean, log_std
```

Poi `full_pass()` campiona un'azione con reparameterization trick, applica
`tanh`, riscalando l'azione nei bound dell'ambiente, e calcola `log_prob`.

Metodi principali:

```python
forward(state) -> mean, log_std
full_pass(state) -> action, log_prob, tanh_action, mean, log_std
select_action(state) -> azione stocastica
select_greedy_action(state) -> azione deterministica sulla media
```

Contiene anche `logalpha`, usato da `SAC` per l'automatic entropy tuning.

### `DeterministicPolicyNetwork`

Questa e la rete indicata in questo schema come `detActor`.

Dato uno stato produce direttamente un'azione continua:

```text
obs -> MLP -> tanh -> rescale nei bound -> action
```

Metodi principali:

```python
forward(state) -> action
select_action(state) -> numpy action
```

### `ValueNetwork`

Stima il valore di uno stato:

```text
V(s) -> scalare
```

Serve agli algoritmi actor-critic on-policy, dove il critic valuta quanto e
buona l'osservazione corrente.

### `QNetwork`

Stima il valore di una coppia stato-azione:

```text
Q(s, a) -> scalare
```

In `SAC` vengono usate due Q-network separate per ridurre la sovrastima.

### `TwinQNetwork`

Contiene due Q-network nello stesso modulo:

```text
Q1(s, a), Q2(s, a)
```

Serve soprattutto a `TD3`, dove la target Q usa il minimo tra i due critic.

## Utility

### `EpisodeBuffer`

Buffer per algoritmi on-policy. Conserva traiettorie temporanee:

```text
states, actions, rewards, next_states, dones, log_probs, values
```

Serve per calcolare returns, advantages e GAE. Di solito viene svuotato dopo
un update.

### `ReplayBuffer`

Buffer circolare per algoritmi off-policy:

```text
state, action, reward, next_state, done
```

Viene usato da `DDPG`, `TD3` e `SAC`. A differenza di `EpisodeBuffer`, i dati
possono essere riutilizzati molte volte.

### `MixedReplayBuffer`

Buffer usato da `MBPO`. Contiene:

```text
real_buffer   transizioni raccolte dall'ambiente reale
synth_buffer  transizioni generate dal surrogate
```

Quando viene campionato un batch, prende una percentuale fissata da
`real_ratio` dal buffer reale e il resto dal buffer sintetico. Se non ci sono
abbastanza dati sintetici, ricade sul solo buffer reale.

### `NormalNoiseDecayStrategy`

Genera rumore gaussiano decrescente per l'esplorazione di agenti deterministici
come `DDPG`.

### `trpo_utils.py`

Contiene le funzioni matematiche di supporto a `TRPO`:

```text
conjugate_gradient
fisher_vector_product
compute_kl_divergence
get_flat_params / set_flat_params
get_flat_grad
line_search
```

Queste funzioni servono per aggiornare la policy rispettando il vincolo di KL.

### `Logger`

Utility semplice per salvare metriche di training in formato tabellare.

## Algoritmi Model-Free

Gli algoritmi model-free non usano direttamente la struttura interna di
TraceWin o del surrogate. Vedono solo transizioni ambiente:

```text
obs, action, reward, next_obs, done
```

### `REINFORCE`

Algoritmo policy-gradient Monte Carlo.

Usa `stocActor` e aggiorna la policy alla fine dell'episodio usando il ritorno
accumulato. Non ha una rete critic separata, quindi e semplice ma puo avere
varianza alta.

### `A2C`

Actor-Critic on-policy.

Usa:

```text
stocActor + ValueNetwork + EpisodeBuffer
```

La policy decide le azioni, il critic stima `V(s)` e l'advantage guida
l'aggiornamento dell'attore.

### `PPO`

Algoritmo on-policy con clipped objective.

Usa:

```text
stocActor + ValueNetwork + EpisodeBuffer
```

Il buffer raccoglie una traiettoria, poi PPO fa piu epoche di ottimizzazione
mantenendo il rapporto tra nuova e vecchia policy entro un intervallo controllato.

### `TRPO`

Algoritmo on-policy con vincolo esplicito sulla KL divergence.

Usa:

```text
stocActor + ValueNetwork + EpisodeBuffer + trpo_utils
```

Invece di usare il clipping come PPO, cerca un passo di aggiornamento che
massimizzi il surrogate objective rispettando un limite di distanza dalla policy
precedente.

### `DDPG`

Algoritmo off-policy deterministico.

Usa:

```text
detActor + QNetwork + target_actor + target_critic + ReplayBuffer
```

L'azione della policy viene perturbata da rumore di esplorazione. Le target
network rendono piu stabile il learning.

### `TD3`

Versione piu robusta di DDPG.

Usa:

```text
detActor + TwinQNetwork + target_actor + target_critic + ReplayBuffer
```

Le tre idee principali sono:

- due critic e target con il minimo tra Q1 e Q2;
- rumore sulla target action;
- update dell'attore meno frequente del critic.

### `SAC`

Soft Actor-Critic custom del progetto.

Usa:

```text
stocActor + QNetwork critic1 + QNetwork critic2 + target critic + ReplayBuffer
```

Ottimizza sia il reward sia l'entropia della policy. Il parametro di entropia
`alpha` viene adattato tramite `policy.logalpha`.

Questa classe espone l'API step-by-step richiesta da `MBPO`:

```python
select_action(...)
store(...)
optimize()
save(...)
load(...)
```

### `SB3SAC`

Wrapper intorno a `stable_baselines3.SAC`.

Espone:

```python
train(env, n_steps)
select_action(obs)
save(path)
load(path, env)
```

E comodo quando si vuole usare l'implementazione robusta di Stable Baselines3,
ma non ha la stessa API interna degli agenti custom: il replay buffer e
l'ottimizzazione sono gestiti dentro SB3.

Per questo non e intercambiabile direttamente con `MBPO`, che si aspetta di
poter sostituire `agent.replay` con un `MixedReplayBuffer` e chiamare
`agent.optimize()` a ogni step.

## Algoritmi Model-Based

Gli algoritmi model-based sfruttano il surrogate neurale non solo come ambiente
veloce, ma come modello del mondo fisico.

### `MBPO`

`MBPO` implementa Model-Based Policy Optimization.

Riceve un agente interno, tipicamente `SAC` custom o un agente con API
compatibile, e un ensemble di `ModularMLP`. Poi:

1. salva le transizioni reali in `MixedReplayBuffer.real_buffer`;
2. usa `SurrogateBeamSimulator` per generare rollout sintetici;
3. salva le transizioni sintetiche in `MixedReplayBuffer.synth_buffer`;
4. allena l'agente interno su batch misti reali/sintetici.

Il surrogate non e un transition model generico `(s, a) -> s'`. Nel progetto e
un simulatore globale:

```text
beam0 + parametri -> tutti gli stati del fascio
```

Per questo i rollout sintetici partono da un `beam0` campionato dal dataset e
da parametri iniziali campionati intorno ai default.

### `MBPOWithModelUpdate`

Estende `MBPO` aggiungendo l'aggiornamento online del surrogate tramite
`SurrogateDatasetUpdater`.

Durante il training con `TraceWinEnv`, se `info["sim_result"]` contiene un
`BeamSimulationResult(source="tracewin")`, il risultato reale viene delegato
all'updater. Ogni `model_train_freq` step, se ci sono abbastanza campioni:

1. l'updater converte il risultato TraceWin in righe `X/Y/scores`;
2. lo aggiunge al dataset online;
3. crea batch misti offline + online;
4. fine-tuna ogni surrogate dell'ensemble;
5. opzionalmente salva il dataset aggiornato;
6. opzionalmente salva i pesi aggiornati dei surrogate.

Questa classe e la variante piu vicina all'idea originale di MBPO, dove il
modello viene migliorato man mano che arrivano nuovi dati reali.

### `SVGAgent`

`SVGAgent` usa il surrogate in modo ancora piu diretto.

Invece di imparare un critic separato, retropropaga il reward attraverso il
surrogate differenziabile:

```text
policy -> action -> parametri -> surrogate -> score -> loss policy
```

Usa `stocActor`, campiona `beam0` dal dataset e ottimizza episodi interi o
semi-troncati attraverso `forward_differentiable()` di `SurrogateBeamSimulator`.

La classe non legge direttamente `config.adige`: dimensione osservazione,
dimensione azione, bounds fisici, chiavi parametriche e parametri di default
sono passati dal caller.

Il risultato delle chiamate di training o valutazione e contenuto in
`SVGResult`, che salva loss, score finale, storico degli score e norma del
gradiente.

## Baseline Senza Policy Neurale

Le baseline non imparano una policy `obs -> action`. Ottimizzano direttamente
il vettore dei parametri usando una funzione obiettivo:

```text
objective(params) -> score
```

### `BayesianOptimizer`

Ottimizzazione bayesiana con modello surrogato probabilistico della funzione
obiettivo.

Prova nuovi parametri bilanciando esplorazione e sfruttamento. Il risultato
viene restituito in `BOResult`.

Riceve dal caller chiavi parametriche, valori default e sensitivity usati per
costruire lo spazio di ricerca.

## Flussi Tipici Di Training

### Training model-free custom

```text
crea env
crea agente
per ogni episodio:
    obs = env.reset()
    per ogni step:
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        agent.optimize()
        obs = next_obs
```

Questo schema vale soprattutto per `SAC`, `TD3` e `DDPG`.

### Training on-policy

```text
crea env
crea agente on-policy
raccogli una traiettoria in EpisodeBuffer
calcola returns / advantages
aggiorna policy e critic
svuota il buffer
```

Questo schema vale per `REINFORCE`, `A2C`, `PPO` e `TRPO`.

### Training MBPO

```text
crea TraceWinEnv oppure SurrogateEnv per le transizioni reali
crea SAC custom
crea MBPO(agent=SAC, surrogates=ensemble, dataset=dataset)
per ogni step reale:
    raccogli transizione reale
    MBPO salva la transizione reale
    MBPO genera rollout sintetici col surrogate
    SAC si allena su MixedReplayBuffer
```

Con `MBPOWithModelUpdate`, quando l'ambiente reale e `TraceWinEnv`, i
`BeamSimulationResult(source="tracewin")` vengono passati a
`SurrogateDatasetUpdater`, che aggiorna dataset online e surrogate.

### Training SVG

```text
crea ensemble surrogate
crea dataset
crea SVGAgent passando obs_dim, act_dim, action_bounds, param_keys e default_params
per ogni episodio:
    campiona beam0
    unroll della policy dentro il surrogate differenziabile
    backprop sul reward cumulativo
```

Qui il surrogate e parte del grafo di calcolo: non serve un replay buffer.

## Mappa Concettuale Delle Dipendenze

Leggendo la struttura della cartella:

- i nodi `model_free` sono gli agenti RL classici;
- i nodi `model_based` sono gli agenti che usano il surrogate come modello;
- i nodi `networks` sono moduli neurali riusati dagli agenti;
- i nodi `utils` sono componenti di supporto condivisi;
- i nodi `baselines` sono ottimizzatori non-RL.

Le frecce vanno lette come "usa", "contiene" o "eredita":

```text
SAC -> stocActor
SAC -> QNetwork
SAC -> ReplayBuffer
MBPO -> SAC/TD3-compatible agent
MBPO -> MixedReplayBuffer
MBPO -> SurrogateBeamSimulator
MBPOWithModelUpdate -> MBPO
SVGAgent -> stocActor
SVGAgent -> DifferentiableSurrogateEnv
```

Questo schema usa i nomi `stocActor` e `detActor` per chiarezza concettuale, ma
tra parentesi mantiene i nomi reali delle classi Python:

```text
stocActor = GaussianPolicyNetwork
detActor  = DeterministicPolicyNetwork
```

## Regole Mentali Per Non Confondersi

- Se l'algoritmo usa `EpisodeBuffer`, e on-policy.
- Se l'algoritmo usa `ReplayBuffer`, e off-policy.
- Se usa `MixedReplayBuffer`, e MBPO.
- Se usa `SurrogateBeamSimulator` per generare dati o gradienti, e model-based.
- Se usa `stocActor`, ragiona su distribuzioni di azioni.
- Se usa `detActor`, produce direttamente l'azione.
- `SB3SAC` e SAC, ma con gestione interna Stable Baselines3.
- `BayesianOptimizer` non impara una policy.
