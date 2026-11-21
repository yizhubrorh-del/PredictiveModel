# Política de reentrenamiento

**Modelo:** precio de vehículos usados en Australia (`HistGradientBoostingRegressor`, versiones en `fase-2/models/`)  
**Implementación:** `monitoreo/politica.py` (umbrales en `UMBRALES`) · evaluación con `python monitor.py reporte` y `GET /metrics`

## 1. Objetivo

Decidir de forma objetiva y repetible **cuándo** reentrenar el modelo, **cómo** validar la versión nueva y **cómo** ponerla en producción o revertirla. La política es deliberadamente sencilla: cuatro reglas, tres niveles y un procedimiento fijo.

## 2. Señales monitoreadas

Todas las señales salen del registro que la API guarda en SQLite: cada solicitud, cada predicción y los precios reales que llegan por `POST /feedback`.

| Señal | Qué detecta | Fuente |
|---|---|---|
| **PSI** de `Year`, `Kilometres` y del precio predicho | Cambio en la población de vehículos (deriva de datos) o en la salida del modelo | Predicciones frente a `referencia/perfil_referencia.json` (dataset de entrenamiento, 16.680 filas) |
| % de **marcas no vistas** en entrenamiento | Vehículos que el modelo no conoce | Predicciones frente a las 76 marcas del perfil |
| **Razón de MAE** (producción / prueba) | Degradación del error en AUD | Predicciones con precio real frente a `metrics.json` de la versión |
| **Razón de MAPE** (producción / prueba) | Degradación del error relativo; detecta problemas en autos baratos que el MAE oculta | Ídem |
| **Antigüedad** del modelo activo | Envejecimiento del mercado aunque no se observe deriva | `created_at_utc` en `metadata.json` |
| Errores **5xx** y **latencia p95** | Problemas del servicio (no del modelo) | Registro de solicitudes |

**PSI** (Population Stability Index): se compara la proporción de registros en cada decil de la referencia.

```
PSI = Σ (p_producción − p_referencia) · ln(p_producción / p_referencia)
```

## 3. Reglas y niveles

Cada regla produce **OK**, **VIGILAR** o **REENTRENAR**. La decisión global es el nivel más grave.

| Regla | VIGILAR si… | REENTRENAR si… | Muestra mínima |
|---|---|---|---|
| **R1 – Deriva** | PSI > 0,10 en alguna variable, o > 5 % de marcas desconocidas | PSI > 0,25 en alguna variable | 400 predicciones |
| **R2 – Rendimiento** | razón de MAE o de MAPE > 1,20 | razón de MAE o de MAPE > 1,30 | 400 precios reales del modelo activo |
| **R3 – Antigüedad** | — | modelo activo con más de 180 días | — |
| **R4 – Operación** | errores 5xx > 1 % o latencia p95 de `/predict` > 1.000 ms | — (reentrenar no corrige el servicio) | — |

Si no se alcanza la muestra mínima, la regla informa OK con el motivo "muestra insuficiente". Sin perfil de referencia, R1 informa VIGILAR.

**Qué hacer en cada nivel**

- **OK:** ninguna acción.
- **VIGILAR:** revisar el reporte. Si persiste en **dos evaluaciones semanales seguidas**, se trata como REENTRENAR. Si la causa es R4, se abre una incidencia del servicio.
- **REENTRENAR:** aplicar el procedimiento de la sección 5. `monitor.py reporte` termina con **código 2** para que una tarea programada lo detecte.

## 4. Calibración de los umbrales

Los umbrales de PSI (0,10 y 0,25) son los convencionales. Los mínimos de muestra y las razones de R2 se calibraron con **submuestras aleatorias de la partición de prueba de `v1`**: datos que el modelo no vio y sin deriva real. Lo que se observe ahí es puro ruido de muestreo.

| n | PSI p99 (Year / Km) | Razón MAE p95 · p99 | Razón MAPE p95 · p99 |
|---:|---:|---:|---:|
| 100 | — | 1,39 · 1,59 | 1,23 · 1,40 |
| 200 | 0,11 / 0,11 | 1,25 · 1,38 | 1,16 · 1,26 |
| **400** | **0,05 / 0,05** | **1,17 · 1,26** | **1,11 · 1,16** |
| 800 | — | 1,10 · 1,15 | 1,07 · 1,10 |

Con n = 400, el azar no supera VIGILAR (1,20) en el 95 % de los casos ni REENTRENAR (1,30) en el 99 %. Con muestras de 100 habría muchas falsas alarmas: una razón de 1,39 aparece por azar 1 de cada 20 veces. Por eso los mínimos son 400.

## 5. Procedimiento de reentrenamiento

1. **Confirmar la causa.** Revisar el reporte (`reportes/…/reporte_monitoreo.md`) y descartar errores de datos (columnas vacías, unidades distintas) o del servicio.
2. **Preparar los datos.**
   - `python monitor.py exportar --salida datos_produccion.csv` exporta los vehículos con precio real, con el mismo esquema del dataset.
   - Ordenarlos por fecha y **reservar el 20 % más reciente** como conjunto de evaluación, que **no se usa para entrenar**.
   - Unir el 80 % restante al dataset original para el nuevo entrenamiento.
3. **Entrenar una versión candidata** con un identificador nuevo, p. ej. `v2026-11-21`. Usar `POST /train` (o `train.py`) con `source_commit` para la trazabilidad. Las versiones **nunca se sobrescriben ni se borran**.
4. **Comparar** la candidata con la activa sobre el conjunto reservado:
   `python monitor.py comparar --data evaluacion.csv --versiones v1 v2026-11-21 --activa v1`.
   Se promueve solo si **reduce el MAE al menos 2 %** (`MEJORA_MINIMA_PROMOCION`) y **no empeora el MAPE**.
5. **Promover:** `PUT /models/active {"version": "v2026-11-21"}`. El cambio queda en el historial (`GET /models`) y persiste aunque se reinicie la API.
6. **Vigilar 7 días.** Si R2 empeora frente a la versión anterior, **revertir** con `PUT /models/active {"version": "v1"}`. La versión anterior sigue disponible.

> **Advertencia sobre la comparación (caso real del proyecto).** Al comparar `v1` y `v2` sobre la partición de prueba de `v1`, `comparar` recomienda promover `v2` porque su MAE es 20 % menor (3.932 frente a 4.930 AUD). Esa comparación **no es válida**: `v2` se entrenó con otra semilla y su entrenamiento incluyó gran parte de esas filas. Lo mismo ocurre con los datos simulados del escenario de deriva (27,7 % "mejor"). Por eso el paso 2 exige un conjunto que **ninguna** versión haya visto, y `comparar` lo recuerda en pantalla.

## 6. Frecuencia y responsables

- **Evaluación semanal:** `python monitor.py reporte --horas 168 --salida reportes/AAAA-MM-DD`, programada con el Programador de tareas de Windows o con cron. El reporte se archiva en el repositorio.
- **Revisión continua:** `GET /metrics?horas=24` para consultas puntuales.
- **Reentrenamiento periódico:** al menos cada 180 días (R3), aunque no haya alertas.
- **Decisión de promover o revertir:** la toma un integrante y la revisa el otro, igual que los Pull Requests.

## 7. Limitaciones

- El rendimiento real depende de que lleguen precios de venta por `/feedback`. Sin ellos solo se puede vigilar la deriva.
- El perfil de referencia usa todo el dataset y no exactamente la partición de entrenamiento de cada versión. La diferencia es menor porque la partición es estratificada.
- El PSI se calcula sobre tres variables numéricas y la proporción de marcas desconocidas. Cambios en otras variables categóricas (combustible, carrocería) solo se detectan indirectamente, a través del precio predicho y del error.
- El registro en SQLite es adecuado para un solo servidor. Con varios procesos o servidores habría que pasar a una base compartida.
