# Predicción de precios de vehículos en Australia

Proyecto Integrador – **Modelos y Simulación de Sistemas I**

Llevamos un modelo de Machine Learning desde un notebook hasta un prototipo desplegable: reproducible, contenerizado, expuesto mediante una API REST y con monitoreo básico.

**Problema:** regresión. Se estima el precio (`Price`, AUD) de un vehículo a partir de sus características (marca, modelo, año, kilometraje, motor, combustible, transmisión, tracción, carrocería, etc.).
**Dataset:** [Australian Vehicle Prices – Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices) (16.734 registros, 19 variables).

## Integrantes

| Nombre | CC |
|---|---|
| Faber Josué Palacio Cuyan | 1127793801 |
| Juan José Jaramillo Marín | 1035973625 |
| Juan José Ortiz del Rio | 1007565771 |

## Estructura del repositorio

```
.
├── data/
│   └── raw/australian_vehicle_prices.csv   # dataset original (Kaggle)
├── fase-1/                                  # Modelo predictivo  ✅ Entrega 1
│   ├── fase1_modelo_predictivo.ipynb
│   ├── features.py
│   ├── models/
│   └── README.md
├── fase-2/                                  # Scripts y Docker   (Entrega 2)
├── fase-3/                                  # API REST           (Entrega 3)
├── fase-4/                                  # Monitoreo          (Entrega 3)
├── requirements.txt
└── README.md
```

## Estado del proyecto

| Fase | Descripción | Estado |
|---|---|---|
| 1 | Modelo predictivo | ✅ Completa: [ver README](fase-1/README.md) |
| 2 | Scripts (`train.py`, `predict.py`), Docker y pytest | Pendiente |
| 3 | API REST (`/health`, `/predict`, `/train`) | Pendiente |
| 4 | Monitoreo y política de reentrenamiento | Pendiente |

### Resultado de la Fase 1
`HistGradientBoostingRegressor` sobre `log(Price)`, en un pipeline de scikit-learn que recibe registros crudos.
En test: **MAE 4.930 AUD · R² 0,892 · MAPE 13,2 %** (línea base: MAE 18.478 AUD).

## Inicio rápido

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
cd fase-1
jupyter notebook fase1_modelo_predictivo.ipynb
```

## Flujo de trabajo en Git

- `main`: versiones entregadas (una por entrega).
- `develop`: rama de integración.
- `feature/<descripcion>`: una rama por tarea; se integra a `develop` mediante Pull Request revisado por otro integrante.
