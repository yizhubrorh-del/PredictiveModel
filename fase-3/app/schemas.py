"""Esquemas de entrada y salida de la API (validación con Pydantic).

Los campos de `Vehicle` usan los mismos nombres que las columnas del CSV
original, de modo que una fila del dataset se puede enviar tal cual.
"""
from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
SHA_PATTERN = r"^[0-9a-fA-F]{40}$"
MAX_VEHICLES = 1000

# Texto libre: se recortan espacios; cadenas vacías no se admiten (usar null).
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


def _with_unit(value, unit: str):
    """Permite enviar números (4) además del formato del dataset ("4 Doors")."""
    if isinstance(value, bool):
        raise ValueError("debe ser texto o número, no booleano")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("no puede ser negativo")
        number = int(value) if float(value).is_integer() else value
        return f"{number} {unit}"
    return value


class Vehicle(BaseModel):
    """Un vehículo con las variables crudas del dataset (sin `Price`)."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Brand": "Toyota", "Model": "Corolla", "Year": 2018,
                "UsedOrNew": "USED", "Transmission": "Automatic",
                "Engine": "4 cyl, 1.8 L", "DriveType": "Front", "FuelType": "Unleaded",
                "FuelConsumption": "6.1 L / 100 km", "Kilometres": 85000,
                "ColourExtInt": "White / Black", "Location": "Parramatta, NSW",
                "CylindersinEngine": "4 cyl", "BodyType": "Hatchback",
                "Doors": "5 Doors", "Seats": "5 Seats",
            },
        },
    )

    # Obligatorios: sin ellos la predicción carece de sentido.
    Brand: Text
    Model: Text
    Year: int = Field(ge=1940, description="Año del modelo")

    # Opcionales: si faltan o son null, el pipeline los imputa.
    UsedOrNew: Text | None = None
    Transmission: Text | None = None
    Engine: Text | None = Field(default=None, examples=["4 cyl, 2.2 L"])
    DriveType: Text | None = None
    FuelType: Text | None = None
    FuelConsumption: Text | None = Field(default=None, examples=["8.7 L / 100 km", 8.7])
    Kilometres: float | None = Field(default=None, ge=0, le=5_000_000)
    ColourExtInt: Text | None = Field(default=None, examples=["White / Black"])
    Location: Text | None = Field(default=None, examples=["Caringbah, NSW"])
    CylindersinEngine: Text | None = Field(default=None, examples=["4 cyl", 4])
    BodyType: Text | None = None
    Doors: Text | None = Field(default=None, examples=["4 Doors", 4])
    Seats: Text | None = Field(default=None, examples=["5 Seats", 5])
    Title: Text | None = None
    CarSuv: Text | None = Field(default=None, alias="Car/Suv")

    @field_validator("Year")
    @classmethod
    def year_not_in_future(cls, value: int) -> int:
        limit = date.today().year + 1
        if value > limit:
            raise ValueError(f"el año no puede ser mayor que {limit}")
        return value

    @field_validator("Doors", mode="before")
    @classmethod
    def doors_unit(cls, value):
        return _with_unit(value, "Doors")

    @field_validator("Seats", mode="before")
    @classmethod
    def seats_unit(cls, value):
        return _with_unit(value, "Seats")

    @field_validator("CylindersinEngine", mode="before")
    @classmethod
    def cylinders_unit(cls, value):
        return _with_unit(value, "cyl")

    @field_validator("FuelConsumption", mode="before")
    @classmethod
    def fuel_unit(cls, value):
        return _with_unit(value, "L / 100 km")

    def to_record(self) -> dict:
        """Fila con los nombres de columna originales (p. ej. 'Car/Suv')."""
        return self.model_dump(by_alias=True)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vehicles: list[Vehicle] = Field(min_length=1, max_length=MAX_VEHICLES)


class PredictResponse(BaseModel):
    version: str
    moneda: str = "AUD"
    n: int
    predicciones: list[float]


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(pattern=VERSION_PATTERN, description="Identificador nuevo, p. ej. v3")
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    source_commit: str | None = Field(default=None, pattern=SHA_PATTERN)


class TrainResponse(BaseModel):
    version: str
    metricas: dict
    filas_entrenamiento: int
    filas_prueba: int
    git_commit: str | None
    duracion_s: float


class HealthResponse(BaseModel):
    status: str
    modelo_por_defecto: str
    modelo_cargado: bool
    versiones_disponibles: list[str]
    dataset_disponible: bool
    detalle: str | None = None
