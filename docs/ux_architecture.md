# Architettura UX/UI di PREACT

Questo documento definisce la struttura informativa e i componenti principali dell'esperienza utente della piattaforma PREACT. L'architettura è organizzata in aree funzionali che guidano l'utente dalla configurazione degli scenari, all'esecuzione delle simulazioni, fino all'analisi dei risultati e alla governance collaborativa.

## 1. Home / Overview
- **What's running now**
  - Dashboard dei scenari attivi con stato in tempo reale (running, paused, completed).
  - Indicazione dello stato di eventuali simulazioni pianificate e log di alert critici.
- **KPI globali**
  - Metriche aggregate visualizzate in hero cards (benessere, disuguaglianza, inflazione sintetica, sentiment, stabilità sociale).
  - Possibilità di espandere ogni KPI per vedere trend, distribuzioni e benchmark storici.
- **Quick actions**
  - Pulsanti rapidi per creare un nuovo scenario, duplicare uno scenario esistente o importare un template preconfigurato.
  - Accesso a wizard contestuali e a tutorial guidati.
- **Focus fiscale/welfare**
  - Widget dedicati a riforme tributarie o di trasferimento con link diretto a template e analisi preconfigurate.

## 2. Scenario Studio (core del prodotto)
- **Policy Builder**
  - Editor visuale con slider, switch, input numerici, formule personalizzabili e regole "se/allora".
  - Supporto a versioning delle politiche con cronologia, commenti e funzionalità di rollback.
- **Population & Agents**
  - Pannello per configurare parametri demografici (età, reddito, istruzione, geografia).
  - Gestione di classi di agenti (cittadini, imprese, istituzioni) e delle rispettive strategie.
  - Definizione delle distribuzioni iniziali e del seed casuale per replicabilità.
- **Shocks & Events**
  - Libreria di stress test (crisi energetica, alluvione, variazione tassi, campagne di disinformazione).
  - Possibilità di pianificare eventi su timeline e definire trigger condizionali.
- **Data Binding**
  - Interfaccia per collegare dataset reali (GDELT, World Bank, UNHCR) tramite connettori o upload manuale.
  - Mapping dei campi per allineare i dati esterni con i modelli della simulazione.
- **Validation Rules**
  - Definizione di vincoli hard/soft su budget, sostenibilità e limiti normativi.
  - Indicatori visivi che segnalano violazioni o conflitti tra regole configurate.
- **Blueprint use-case**
  - Checklist guidata per le riforme fiscali/welfare che suggerisce parametri obbligatori, dataset consigliati e best practice.

## 3. Simulation Run & Monitor
- **Console in tempo reale**
  - Visualizzazione dello stato della simulazione (step attuale, tick/sec, consumo di risorse, log eventi).
  - Controlli per avviare, mettere in pausa, accelerare o terminare una simulazione.
- **Mappe + Timeline sincronizzate**
  - Mappa interattiva (choropleth, hexbin, markers) collegata a una timeline con zoom e filtri temporali.
  - Possibilità di scrub sulla timeline per visualizzare l'evoluzione degli indicatori.
- **Layer Manager**
  - Gestione di layer tematici (occupazione, prezzi, mobilità, proteste) con opzioni di visibilità e opacità.
  - Funzionalità di salvataggio di viste personalizzate e condivisione con il team.
- **Uncertainty Panel**
  - Visualizzazioni delle incertezze (confidence interval, fan charts, distribuzioni Monte Carlo).
  - Annotazioni su ipotesi e parametri che influenzano la varianza dei risultati.

## 4. Results & Compare
- **Scenario Diff**
  - Confronto side-by-side tra due o più scenari con cursori sincronizzati.
  - Heatmap delle differenze e indicatori di significatività statistica.
- **Explainability**
  - Insight basati su Shapley values e feature importance per collegare le politiche agli outcome.
  - Grafo "policy → outcome" che visualizza le relazioni causali individuate.
- **Narrative Report**
  - Generazione assistita da LLM di un executive summary corredato da figure e note metodologiche.
  - Editor per revisionare il testo e inserire commenti collaborativi.
- **Policy Brief Generator**
  - Template automatici per executive summary destinati a ministeri e municipalità, con raccomandazioni e trade-off principali.
- **Export**
  - Esportazione in PDF/HTML dei report, in CSV/Parquet dei dataset generati e notebook template (Jupyter) per analisi avanzate.

## 5. Agents Lab (ispezione micro)
- **Agent Inspector**
  - Drill-down su singoli agenti con visualizzazione di stato, utility, reti sociali e log delle decisioni.
  - Strumenti per confrontare agenti simili e individuare pattern comportamentali.
- **Behavior Traces**
  - Replays velocizzati delle azioni con evidenza dei fattori che le hanno influenzate.
  - Annotazioni manuali o automatiche per spiegare deviazioni rispetto ai modelli attesi.
- **Counterfactuals**
  - Simulatori locali "what-if" su subset di agenti per valutare alternative tattiche.
  - Confronto diretto tra outcome osservato e outcome controfattuale.

## 6. Library
- Catalogo di template di politiche (IVA, sussidi, trasporti, salario minimo) pronti all'uso.
- Dataset pre-cablati con metadati, provenienza e suggerimenti di utilizzo.
- Modelli di shock/eventi riutilizzabili con possibilità di personalizzazione.

## 7. Governance
- **Projects & Workspaces**
  - Gestione di progetti con permessi granulari (viewer, editor, owner) e gruppi di lavoro.
  - Audit trail delle azioni principali e tracciamento della provenienza dei dati.
- **Scenario Versioning**
  - Meccanismi di branching, tagging, rollback e changelog automatico.
  - Visualizzazioni delle differenze tra versioni con approvazioni e commenti.

## Considerazioni trasversali
- Design responsive per supportare desktop e tablet in contesti di presentazione.
- Localizzazione multilingua con fallback inglese/italiano.
- Accessibilità WCAG 2.1 AA con particolare attenzione a colori, contrasti e navigazione da tastiera.
- Integrazione con sistemi di autenticazione esterni (SSO, OAuth) e log centralizzato degli errori.

Questa architettura funge da blueprint per il design UI e l'implementazione delle funzionalità della piattaforma PREACT, garantendo coerenza e scalabilità nel tempo.
