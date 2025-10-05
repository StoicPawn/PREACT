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
