# Analisi stato vs visione per PREACT

## Aspettative espresse
- Piattaforma di simulazione di eventi globali, utilizzabile da policy maker internazionali.
- Agenti intelligenti con capacità avanzate (LLM/RL) e supporto a scenari complessi multi-policy.
- Interfacce sofisticate per configurare dinamiche non fiscali e policy composite.

## Capacità attuali dell'MVP
- Il motore `SimulationEngine` orchestra solo tre core (policy, economia, sentiment) su scenari fiscali sintetici con tick mensili.【F:preact/simulation/engine.py†L1-L118】
- Il `PolicyCore` applica imposte, detrazioni e sussidi basilari (aliquote a scaglioni, detrazione base, sussidio figli e indennità disoccupazione).【F:preact/simulation/policy.py†L1-L87】
- `EconomyCore` e `SentimentCore` implementano regole deterministiche su occupazione, consumi, CPI e sentiment, senza agenti dotati di ML né grafi sociali.【F:preact/simulation/economy.py†L1-L138】【F:preact/simulation/sentiment.py†L1-L74】
- La guida MVP `building_map.md` delimita esplicitamente il focus a riforme fiscali locali con dashboard Streamlit, escludendo RL, LLM e scenari globali dalla v1.【F:building_map.md†L3-L70】【F:building_map.md†L92-L118】
- L'architettura UX documentata copre policy pubbliche e governance interna, non piattaforme di eventi globali o gestione multi-stakeholder in tempo reale.【F:docs/ux_architecture.md†L1-L92】

## Gap principali
1. **Dominio limitato**: il perimetro corrente riguarda riforme fiscali locali; mancano moduli per crisi globali, eventi climatici, flussi migratori o shock geopolitici.
2. **Intelligenza degli agenti**: l'MVP usa regole fisse. Non sono integrati agenti cognitivi/LLM, come previsto per fasi successive nella roadmap.【F:README.md†L63-L105】【F:README.md†L117-L153】
3. **Scalabilità operativa**: non esistono ancora orchestrazione distribuita, pipeline dati reali o storage enterprise; la roadmap cita Ray/Kubernetes come obiettivo futuro.【F:README.md†L107-L138】
4. **Esperienza utente**: l'interfaccia pianificata è Streamlit con slider; non c'è la piattaforma collaborativa e multi-dominio descritta nell'aspettativa.

## Roadmap consigliata per riallineare
1. **Ridefinire la visione**
   - Aggiornare la roadmap distinguendo MVP fiscale vs piattaforma globale.
   - Identificare casi d'uso prioritari (es. crisi energetica continentale, gestione pandemie) e relativi KPI.
2. **Estendere il dominio simulato**
   - Introdurre moduli aggiuntivi (clima, mobilità, commercio internazionale) con modelli dati dedicati.
   - Creare nuovi builder di eventi/shock con timeline condizionali, ispirandosi alle sezioni "Shocks & Events" della UX blueprint.【F:docs/ux_architecture.md†L19-L44】
3. **Evolvere gli agenti**
   - Integrare meccanismi di RL/LLM per agenti istituzionali e cittadini (vedi sezione "Estendibilità AI" nel README).【F:README.md†L117-L153】
   - Prevedere grafi sociali e interazioni multi-livello per riflettere dinamiche globali.
4. **Architettura scalabile**
   - Implementare pipeline dati in `data_ingestion` per fonti globali (GDELT, World Bank) e storage cloud-native.
   - Aggiungere orchestrazione distribuita e supporto multi-tenant.
5. **Esperienza piattaforma**
   - Passare da Streamlit a frontend modulare (es. React/Next.js) per supportare workflow avanzati descritti in `docs/ux_architecture.md`.
   - Implementare governance, versioning e collaboration real-time.

## Conclusione
Non è stata sbagliata la strada: l'MVP attuale è deliberatamente ristretto per testare il ciclo policy fiscale → simulazione → analisi. Per avvicinarsi alla visione di piattaforma globale occorre:
- espandere gradualmente il dominio degli eventi e l'intelligenza agent-based,
- rafforzare la scalabilità tecnica,
- evolvere UX e governance verso un prodotto enterprise multi-stakeholder.
Questi passi possono essere pianificati sulle fasi future della roadmap esistente, mantenendo continuità con l'architettura attuale ma elevandone l'ambizione.

## Implementazioni a supporto
- **Dominio esteso**: il modulo `simulation/events` gestisce timeline multi-evento con shock combinati e aggiustamenti di policy applicati dinamicamente dal `SimulationEngine`.
- **Agenti intelligenti**: il pacchetto `agents` introduce un `AdaptivePolicyAgent` basato su Q-learning che ottimizza le riforme tramite reward configurabile.
- **Scalabilità**: `pipeline/distributed.py` abilita l'esecuzione batch di scenari su thread, processi o Ray, preparandosi a workload globali.
