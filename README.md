# Imperia Matching TFM

Proyecto TFM independiente basado en Imperia Estate.

El objetivo es desarrollar y evaluar un sistema de recomendacion inmobiliaria explicable que combine:

- scoring estructurado entre clientes/leads y propiedades;
- extraccion automatica de preferencias desde conversaciones o formularios;
- explicaciones accionables para agentes inmobiliarios;
- evaluacion cuantitativa frente a un baseline.

Este repositorio no modifica el CRM productivo. Imperia Estate se usa como caso de uso, referencia funcional y posible fuente de datos anonimizados o sinteticos.

## Alcance Inicial

Tema propuesto:

> Sistema hibrido de recomendacion inmobiliaria explicable integrado en un CRM: matching entre propiedades y clientes mediante scoring estructurado, extraccion de preferencias e IA generativa.

El nucleo del TFM sera el motor de matching. El modulo de leads se usara como entrada para construir o enriquecer las preferencias del cliente.

## Estructura

```text
data/
  raw/          Exportaciones anonimizadas o fuentes originales no procesadas
  processed/    Datos normalizados listos para experimentos
  synthetic/    Datos sinteticos reproducibles
  labels/       Etiquetas humanas o ground truth para evaluacion
notebooks/      Analisis exploratorio y experimentos
src/            Codigo reutilizable del TFM
app/            Demo aislada
reports/        Resultados, graficas y tablas
memoria/        Material para la memoria del TFM
tests/          Tests unitarios
```

## Documentacion de gestion

Antes de desarrollar nuevas funcionalidades, el proyecto queda guiado por estos documentos:

- [Roadmap](docs/ROADMAP.md)
- [Alcance](docs/SCOPE.md)
- [Requisitos](docs/REQUIREMENTS.md)
- [Metodologia](docs/METHODOLOGY.md)
- [Protocolo de evaluacion](docs/EVALUATION_PROTOCOL.md)
- [Gobierno de datos](docs/DATA_GOVERNANCE.md)
- [Fuente de propiedades Idealista](docs/PROPERTY_SOURCE_IDEALISTA.md)
- [Ingesta dinamica de propiedades y clientes](docs/DYNAMIC_INGESTION.md)
- [Guia de prueba para evaluadores](docs/EVALUATOR_GUIDE.md)
- [Plan de entrega y reproducibilidad](docs/DELIVERY_PLAN.md)
- [Registro de riesgos](docs/RISK_REGISTER.md)
- [Registro de decisiones](docs/DECISION_LOG.md)

## Material para la memoria

- [Propuesta inicial](memoria/00-propuesta-tfm.md)
- [Arquitectura experimental](memoria/01-arquitectura-experimental.md)
- [Plan de datos y evaluacion](memoria/02-plan-datos-evaluacion.md)
- [Extraccion de preferencias](memoria/03-extraccion-preferencias.md)
- [Conclusiones preliminares](memoria/04-conclusiones-preliminares.md)

## Primeras Hipotesis

- H1: Un scoring hibrido mejora la calidad del ranking respecto a un scoring puramente manual basado en reglas.
- H2: La extraccion automatica de preferencias desde leads reduce informacion perdida frente a perfiles creados manualmente.
- H3: Un desglose explicable del match aumenta la utilidad del sistema para el agente inmobiliario.

## Evaluacion Prevista

Metricas recomendadas:

- `Precision@K`
- `Recall@K`
- `NDCG@K`
- correlacion con puntuacion experta
- exactitud/F1 en extraccion de preferencias
- cobertura de criterios explicados

## Estado

Scaffold inicial creado con documentacion de proyecto, baseline, datos sinteticos minimos, metricas y tests.

El siguiente paso es ampliar el dataset y definir la guia de etiquetado.

## Ejecucion

Desde la raiz del proyecto:

Modo recomendado para evaluadores:

```bash
python3 run_tfm_demo.py
python3 run_tests.py
```

En Windows:

```powershell
py run_tfm_demo.py
py run_tests.py
```

Modo de desarrollo:

```bash
PYTHONPATH=src python3 scripts/run_baseline.py
PYTHONPATH=src python3 -m unittest discover
```

Importar propiedades desde CSV preparado:

```bash
python3 scripts/import_idealista_csv.py \
  --input data/raw/idealista_magnus_template.csv \
  --output data/processed/properties_from_idealista.json
```

Probar la demo con propiedades importadas:

```bash
python3 run_tfm_demo.py \
  --properties data/processed/properties_from_idealista.json \
  --labels "" \
  --top-k 5
```

Tambien se puede ejecutar el flujo completo:

```bash
make demo-idealista
```

Extraer una propiedad desde PDF guardado con `Cmd + P`:

```bash
python3 scripts/add_property.py \
  --pdf data/inbox/idealista_111260654.pdf \
  --output data/processed/user_properties.json
```

Extraer una propiedad desde texto pegado:

```bash
python3 scripts/add_property.py \
  --input data/inbox/idealista_111260654.txt \
  --output data/processed/user_properties.json
```

El proyecto base no requiere dependencias externas para ejecutar el baseline, la demo offline y los tests iniciales.
