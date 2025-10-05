# PREACT – MVP (v1) – Building Guide

## Obiettivo dell’MVP
Consentire a un utente **non tecnico** (policy analyst) di:
- configurare **una singola riforma fiscale** semplice su una **città sintetica**,
- lanciarla contro uno **scenario base** (status quo),
- confrontare **8–10 KPI chiave economico-sociali** tramite **dashboard** e **report esportabile**.

> **Nota:** niente IA “pesante” in v1. Solo regole comportamentali deterministiche/stocastiche semplici e un sentiment proxy basilare.  
> Obiettivo: testare il ciclo **modella → esegui → comprendi**.

---

## Scope funzionale (minimo indispensabile)

### 1) Scenario & Policy Builder
- **Parametri fiscali**: 
  - aliquota imposta sul reddito (flat o 2 scaglioni),
  - detrazione base per nucleo,
  - sussidio familiare per figli.
- **Popolazione sintetica** (10k–50k agenti):
  - distribuzione redditi (log-normale),
  - composizione nucleo,
  - occupazione (employed/unemployed),
  - settore (servizi/industria).
- **Imprese sintetiche** (500–2k):
  - produttività,
  - domanda stocastica,
  - occupazione attesa.
- **Amministrazione locale**:
  - budget = gettito – trasferimenti,
  - vincolo: non andare sotto soglia X (warning).

### 2) Motore di simulazione (tick-based)
- **Orizzonte**: 12–24 mesi (tick = 1 mese).
- **Regole base**:
  - reddito netto = reddito lordo – imposte + sussidi,
  - consumi = f(reddito netto; propensione marginale a consumare, risparmio),
  - domanda lavoro imprese = f(domanda beni; costo lavoro),
  - disoccupazione aggiornata con matching semplice,
  - prezzi = indice sintetico (CPI proxy) da domanda-offerta,
  - sentiment proxy (0–100) = funzione di Δreddito disponibile, occupazione, CPI (pesi tarabili).
- **Shock opzionale (1)**:
  - stretta energetica (↑costo imprese) oppure calo domanda estera.

### 3) Output & confronto A/B
- **KPI minimi (10):**
  - gettito, spesa trasferimenti, saldo PA,
  - tasso disoccupazione, tasso occupazione,
  - consumo medio per decile di reddito (equità),
  - Gini (prima/dopo trasferimenti),
  - CPI proxy,
  - sentiment proxy medio & per decile,
  - “winners/losers” (Δ reddito disponibile per cluster).
- **Dashboard (web):**
  - timeline KPI,
  - istogrammi per decili,
  - tabella winners/losers,
  - 2 grafici *Scenario vs Base* side-by-side.
  - ✓ Implementato in `preact/dashboard/app.py` con flusso Config → Run → Results, slider policy e timeline A/B.
- **Export:**
  - CSV/Parquet dei risultati,
  - Report HTML/PDF auto-generato (titolo, setup, grafici, takeaway, caveat).
  - ✓ Report prodotti da `SimulationRepository.export(..., format="html"/"pdf")` con `preact/simulation/reporting.py`.

---

## Fuori scope (v1)
- Reinforcement Learning / LLM negli agenti,
- grafi sociali,
- geografia reale,
- multi-utente/collab,
- mappe,
- Monte Carlo massivo,
- ottimizzatori di policy.  
*(Si pianificano per v2+.)*

---

## Criteri di successo
- **Usabilità:** dalla home a un primo risultato ≤ 2 minuti usando un template.
- **Performance:** 10k–50k agenti × 24 mesi < 30 s su laptop dev (o < 2 min su VM standard).
- **Comprensione:** 80% utenti pilota capisce “chi vince/chi perde” guardando dashboard + report, senza spiegazioni del team.
- **Riproducibilità:** run identiche con lo stesso seed.

---

## Architettura tecnica

### Backend/Engine
- Python 3.11+
- Mesa (multi-agente)
- NumPy / Pandas
- DuckDB (persistenza rapida)
- Moduli:
  - `policy_core`: calcolo imposte/sussidi,
  - `economy_core`: occupazione/consumi/CPI,
  - `sentiment_core`: calcolo sentiment proxy.
- API FastAPI (2–3 endpoint): 
  - `/run`, 
  - `/scenarios`, 
  - `/results/:id`.

### Frontend (MVP)
- Streamlit (rapido prototipo)  
  *(oppure Next.js + Recharts se si vuole look & feel più pro).*
- Pagine:
  - **New Scenario** → **Run** → **Results**.
- Componenti:
  - slider/toggle policy,
  - card KPI,
  - grafici timeline/istogramma,
  - diff A/B,
  - pulsante download.

### Storage
- Config scenario: YAML/JSON
- Risultati: DuckDB + file Parquet/CSV

### Repro & CI
- `pyproject.toml`, `Makefile` con `make run`, `make test`
- Dataset seed + template scenario inclusi nel repo

---

## Dati per calibrazione (minimi & open)
- **Distribuzione redditi**: parametri sintetici (o bins POVCAL World Bank).
- **Propensione al consumo**: ~0.75, differenziata per decile.
- **Occupazione iniziale**: tasso base (8–12%).
- **Prezzi/CPI**: indice base 100, aggiornato da gap domanda-offerta.
- **Sussidi/tasse**: range realistici (aliquota 10–35%, detrazioni 0–2k, sussidio figlio 0–200/m).

> In v1 niente dataset complessi: usare 3–4 parametri macro (reddito mediano, disoccupazione, CPI) per ancorare a un paese “tipo”.

---

## Metriche interne & validazione rapida
- **Coerenza:** saldo PA ≈ gettito − trasferimenti; consumi ≤ reddito netto + risparmio.
- **Stabilità:** senza shock, la serie converge a uno stato stazionario.
- **Sanity checks:**
  - ↑aliquota ⇒ ↑gettito (fino a soglia), ↓consumi, ↓sentiment,
  - ↑sussidio basso reddito ⇒ ↓Gini post-trasferimenti, ↑consumi decili 1–3.

---

## UX minima (3 schermate)
1. **Config:** slider aliquote, detrazioni, sussidi; select *template città*; seed.
2. **Run:** barra progresso, log eventi, stima tempo; pulsante “Compare with Base”.
3. **Results:**  
   - *Executive* (4 KPI grandi + takeaway generati),  
   - *Equità* (Gini + istogrammi decili),  
   - *Macro* (occupazione, CPI, sentiment),  
   - Download + Save Scenario.

---

## Piano operativo (prime 4–6 settimane)
1. Setup engine (Mesa, moduli core, seed & logging).  
2. Implementare regole policy/economia base.  
3. Runner + storage (FastAPI `/run`, DuckDB, Parquet).  
4. Dashboard MVP (Streamlit: Config → Run → Results + export).  
5. Template & demo (3 policy template + dataset seed; script `make demo`).
   - ✓ Libreria template ampliata (`preact/simulation/templates.py`) con città media, regione metropolitana, provincia rurale e hub turistico.
6. User test (task “trova winners/losers”; feedback; fix).  

---

