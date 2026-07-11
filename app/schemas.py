"""
app/schemas.py — Contratos Pydantic del microservicio.

El contrato de respuesta refleja `LlmResponseEntity` que espera la app Flutter
(AgroGraph-MAS), para que el cliente móvil consuma la API sin cambios.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ConsultaRequest(BaseModel):
    """
    Entrada del endpoint /consultar. La CNN corre en el dispositivo (TFLite):
    el cliente envía el RESULTADO de la CNN + el texto de síntomas.
    """
    cropName: str = Field(..., description="Cultivo detectado por la CNN (ej. 'Tomate')")
    diseaseName: str = Field(..., description="Enfermedad detectada (ej. 'Tizón tardío')")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confianza de la CNN [0,1]")
    texto: str = Field("", description="Texto de síntomas del usuario (opcional)")
    rol: str = Field("agricultor", description="'agricultor' o 'aprendiz'")
    cultivos: Optional[list[str]] = Field(
        None, description="Cultivos de la parcela para filtrar (None = sin filtro por parcela)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cropName": "Calabaza", "diseaseName": "oídio",
                "confidence": 0.47, "texto": "polvo blanco como ceniza",
                "rol": "agricultor", "cultivos": ["calabaza", "maiz"]
            }
        }
    }


class LlmResponse(BaseModel):
    """Respuesta compatible con LlmResponseEntity de la app Flutter."""
    diagnostico: str
    tratamiento: str
    prevencion: str
    fuentes: list[str]
    confianzaAjustada: float
    estado: str
    explicacion: str
    sintomas: list[str]
    avisos: list[str]
    sinDocumentos: bool
    # Metadatos del servicio (no del LLM)
    inference_id: str
    modo: str


class InferenciaResumen(BaseModel):
    inference_id: str
    created_at: str
    cultivo: str
    enfermedad: str
    confianza: float


class HistorialResponse(BaseModel):
    total: int
    items: list[InferenciaResumen]


# ── Offline: catálogo y descarga de documentos (🔌 puntos 1 y 2 del plan) ─────

class CatalogItem(BaseModel):
    id: str
    crop_name: str
    disease_name: str
    title: str
    source: str
    size_bytes: int
    version: str


class CatalogResponse(BaseModel):
    documents: list[CatalogItem]


class ChunkModel(BaseModel):
    id: str
    index: int
    text: str
    embedding: list[float]   # 384-d (MiniLM-L12)


class DocumentDownloadResponse(BaseModel):
    id: str
    content: str
    size_bytes: int
    embedding: list[float]   # embedding global (media de los chunks), 384-d
    chunks: list[ChunkModel]


# ── Mapa epidemiológico REAL (campañas fitosanitarias SENASICA) ──────────────

class EstadoResumen(BaseModel):
    estado: str
    campanias: int
    superficie_ha: float
    productores: int
    campania_dominante: str
    cultivo_dominante: str


class MapaCampaniasResponse(BaseModel):
    total_campanias: int
    estados: list[EstadoResumen]
