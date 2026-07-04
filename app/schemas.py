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


# ── Clustering fitosanitario (no supervisado) ────────────────────────────────

class FeatureVectorRequest(BaseModel):
    """AgendaFeatureVector (del plan) que el cliente envía para asignarle cluster."""
    diagnosisId: Optional[str] = None
    cropName: str = ""
    diseaseName: str = ""
    zona: str = ""
    cnnConfidence: float = 0.0
    topKEntropy: float = 0.0
    llmConfianzaAjustada: float = 0.0
    llmEstado: float = 1.0
    llmSinDocumentos: float = 0.0
    sintomasCount: float = 0.0
    avisosCount: float = 0.0
    diagnosticoLength: float = 0.0
    completionRate: float = 0.0
    daysSinceCreation: float = 0.0
    isComplete: float = 0.0
    hasRelapse: float = 0.0
    diagnosisMonth: int = 1


class ClusterResponse(BaseModel):
    cluster_id: int
    cluster_label: str
    zona: str


class ZonaResumen(BaseModel):
    zona: str
    cluster_dominante: str
    casos: int
    total: int


class MapaResponse(BaseModel):
    total: int
    zonas: list[ZonaResumen]
