# 🌍 PREACT – Predictive REality for AI-driven Collective Testing

## Descrizione
**PREACT** è un progetto innovativo che mira a sviluppare un **sistema di simulazione sociale ed economica** basato su **agenti virtuali controllati da modelli di Intelligenza Artificiale**.  

L’obiettivo è creare un **“gemello digitale”** della società (città, Paese o comunità) popolato da migliaia di agenti AI che **imitano il comportamento umano** – cittadini, imprese, istituzioni – nelle loro interazioni quotidiane.  

Prima di introdurre una politica reale (es. riforma fiscale, legge sul lavoro, nuove tariffe nei trasporti), **PREACT** consente di **sperimentarla virtualmente** e osservare gli effetti su:  
- livelli di consumo e produzione  
- distribuzione della ricchezza e disuguaglianze  
- sentiment e opinioni sociali  
- effetti a catena imprevisti nel medio-lungo termine  

Grazie a modelli di IA avanzati, la simulazione può **evolvere autonomamente**, generando scenari non programmati e rivelando dinamiche nascoste, analogamente a come gli algoritmi AI esplorano milioni di possibilità nei giochi complessi.

---

## Obiettivi
- Fornire un **ambiente sicuro** per testare politiche prima della loro applicazione reale.  
- Identificare **conseguenze inattese** e trade-off nascosti nelle decisioni politiche ed economiche.  
- Creare un framework **scalabile** e adattabile a diversi domini (fisco, welfare, sanità, trasporti, energia).  
- Offrire uno strumento utile a **policy maker, ricercatori e think tank**.  
- Contribuire alla **ricerca scientifica in AI, economia computazionale e scienze sociali**.  

---

## Caratteristiche principali
- **Agenti AI multi-ruolo**: cittadini, imprese, istituzioni con obiettivi e comportamenti differenziati.
- **Ambiente modulare**: simulazione di settori specifici o di un’intera società.
- **Scenari personalizzabili**: ogni politica può essere parametrizzata e testata in più varianti.
- **Analisi automatica**: metriche di benessere, equità, sentiment sociale e stabilità economica.
- **Scalabilità cloud**: dalla simulazione di una città fino a interi Paesi.
- **Timeline di eventi globali**: crisi climatiche, shock energetici e flussi migratori possono modificare dinamicamente policy e macro indicatori.
- **Agenti adattivi**: un `AdaptivePolicyAgent` esplora riforme alternative tramite apprendimento per rinforzo tabellare.
- **Esecuzione distribuita**: un runner dedicato permette batch di scenari in parallelo con backend thread, process o Ray.

### Moduli MVP
Il progetto include ora un **motore di simulazione tick-based** allineato alla guida
_building_map.md_. I componenti principali sono:

- `policy_core`: applica aliquote, detrazioni e sussidi per calcolare redditi netti.
- `economy_core`: aggiorna occupazione, consumi e indice dei prezzi in base alla domanda.
- `sentiment_core`: sintetizza il sentiment socio-economico a partire da reddito, lavoro e inflazione.
- `scenario_builder`: genera popolazione, imprese e shock sintetici con parametri riproducibili.
- `simulation_engine`: orchestra i moduli core su un orizzonte mensile e produce KPI richiesti
  (gettito, disoccupazione, Gini, sentiment, winners/losers).
- `simulation_results`: fornisce utility per confronti A/B, esport e aggregazione dei KPI.
- `dashboard`: flusso Streamlit **Config → Run → Results** con slider per policy, timeline A/B e pulsanti di download (CSV, Parquet, HTML, PDF).
- `simulation/templates`: libreria di template territoriali ampliata (città media, area metropolitana, provincia rurale, distretto turistico).
- `simulation/events`: timeline di eventi globali con combinazione di shock economici e aggiustamenti di policy.
- `agents/adaptive.py`: agente Q-learning che ottimizza parametri fiscali in base a funzione di reward definibile.
- `pipeline/distributed.py`: orchestratore per simulazioni parallele con auto-rilevamento del backend disponibile.

Consulta `preact/simulation/` per maggiori dettagli e `tests/test_simulation.py` per esempi
di utilizzo end-to-end.

### Use case principale: sandbox per le politiche pubbliche
Il focus attuale del progetto è offrire a **ministeri, regioni e municipalità** un ambiente di prova completo per
la **valutazione di riforme fiscali e pacchetti di welfare** prima della loro introduzione reale. Il workflow
tipico prevede:

1. **Definizione della riforma**: l’utente configura tasse, trasferimenti e incentivi con il *Policy Builder*.
2. **Calibrazione della popolazione sintetica**: si replica la struttura socio-economica del territorio target
   importando dati reali e definendo comportamenti specifici per cittadini, imprese e PA.
3. **Esecuzione di scenari multipli**: la riforma viene confrontata con status quo e alternative tramite batch di
   simulazioni parametriche.
4. **Analisi multidimensionale**: dashboard e report automatici evidenziano impatti su gettito, equità,
   competitività, sentiment sociale e stabilità macroeconomica.
5. **Supporto alla decisione**: il sistema produce raccomandazioni, controfattuali e spiegazioni trasparenti per
   i decisori, documentando rischi e trade-off.

Questo use case guida le priorità di sviluppo di PREACT in termini di moduli, metriche e requisiti di governance.

### Architettura UX/UI
Per un'anteprima dell'esperienza utente e dell'organizzazione funzionale della piattaforma consulta il documento [Architettura UX/UI di PREACT](docs/ux_architecture.md).

---

## Stack tecnologico (proposto)
- **Python** come linguaggio principale  
- **Framework multi-agente**: [Mesa](https://mesa.readthedocs.io/) o equivalenti  
- **Machine Learning / NLP**: PyTorch o TensorFlow per modelli comportamentali  
- **Database**: PostgreSQL / DuckDB per archiviazione dei dati  
- **Visualizzazione**: Streamlit / Plotly per dashboard interattive  
- **Scalabilità**: supporto a esecuzioni distribuite (Kubernetes, Ray)  

---

## Estendibilità AI
Il progetto è pensato per integrare modelli di IA sempre più sofisticati:  
- **Reinforcement Learning** per esplorare politiche ottimali.  
- **Large Language Models (LLM)** per simulare conversazioni, opinioni e sentiment realistici degli agenti.  
- **Digital Twins personalizzati** basati su dati reali (es. World Bank, GDELT, UNHCR).  
- **Analisi predittiva** per confrontare scenari simulati con dati storici.  

---

## Possibili utilizzi
- **Policy maker**: sperimentare politiche fiscali, sociali o ambientali senza rischi.  
- **Ricerca accademica**: lavori su *computational social science*, *AI policy simulation*, *economia computazionale*.  
- **Organizzazioni internazionali**: valutazione di interventi di sviluppo o transizione energetica.  
- **Think tank e ONG**: analisi di scenari alternativi di governance e cooperazione.  

---

## Roadmap
1. **Fase 1 – Prototipo base**  
   - Simulazione di un ecosistema cittadino con cittadini, imprese e governo locale.  
   - Metriche economiche e sociali di base.  

2. **Fase 2 – Agenti intelligenti**  
   - Integrazione di modelli ML/NLP per comportamenti più realistici.  
   - Introduzione del sentiment sociale e dinamiche di opinione pubblica.  

3. **Fase 3 – Scalabilità**  
   - Simulazioni a livello nazionale.  
   - Dashboard interattiva per testare politiche con input personalizzati.  

4. **Fase 4 – Validazione scientifica**  
   - Confronto dei risultati con dati reali.  
   - Pubblicazioni accademiche sulla metodologia e sui risultati.  

---

## Licenza
Da definire (consigliata: **MIT** o **Apache 2.0** per favorire collaborazione e adozione).

## World Intelligence System

PREACT's product North Star is now a **globe-first World Intelligence System**.

The forecasting engine remains a specialist analytical layer, while the primary product experience becomes an interactive political, historical, geographic and sociological atlas built on a temporal world knowledge graph.

The target experience is:

```text
3D WORLD
  -> select country
  -> current political system and government
  -> deep history and twentieth-century timeline
  -> society/economy
  -> alliances, cooperation, disputes, sanctions and conflicts
  -> inspect evidence and change history
  -> optionally open PREACT forecasts/scenarios as a separate layer
```

World Knowledge is designed to update autonomously from continuously collected sources. New information enters as a candidate claim, is corroborated under a fail-closed policy, and only then may append a new version of factual world state. Superseded facts are closed in valid time rather than overwritten. Material factual changes queue affected narrative sections for regeneration.

The complete product and milestone plan is in [docs/world_intelligence_roadmap.md](docs/world_intelligence_roadmap.md).

The architectural invariant is:

```text
OBSERVED WORLD != INTERPRETATION != FORECAST != COUNTERFACTUAL
```
