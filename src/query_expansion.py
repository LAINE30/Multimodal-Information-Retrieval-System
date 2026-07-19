"""
Módulo de Expansión de Consultas (Query Expansion).

Implementa expansión automática de consultas usando Gemini como LLM
para generar múltiples reformulaciones semánticas de la consulta original.
Esto mejora el Recall del sistema de recuperación al cubrir más variantes
lingüísticas y semánticas que el usuario puede no haber expresado.
"""
import os
import json
import google.generativeai as genai
from typing import List


class QueryExpander:
    """
    Genera múltiples reformulaciones de una consulta de usuario usando Gemini.

    Estrategia: LLM-based Pseudo-Relevance Expansion.
    El modelo recibe la consulta original y produce N variantes que:
    - Usan sinónimos o términos alternativos
    - Reformulan la intención desde ángulos distintos
    - Cubren vocabulario técnico y coloquial
    Esto mejora el Recall del Bi-Encoder (CLIP) al exponer la búsqueda
    a más regiones del espacio vectorial.
    """

    EXPANSION_PROMPT = """Eres un asistente experto en búsqueda de productos de e-commerce.
Tu tarea es generar {n} reformulaciones alternativas de la siguiente consulta de búsqueda.

Consulta original: "{query}"

Reglas:
1. Cada reformulación debe capturar la misma intención, pero con palabras distintas.
2. Varía el vocabulario: usa sinónimos, términos técnicos, coloquiales o en otro idioma si es relevante.
3. Mantén las reformulaciones cortas (máximo 15 palabras cada una).
4. Devuelve ÚNICAMENTE un JSON válido con la siguiente estructura, sin texto adicional:
{{"expansions": ["reformulación 1", "reformulación 2", "reformulación 3"]}}
"""

    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-2.5-flash")

    def expand(self, query: str, n: int = 3) -> List[str]:
        """
        Expande una consulta original generando n reformulaciones adicionales.

        Args:
            query: La consulta original del usuario.
            n: Número de reformulaciones a generar. Por defecto 3.

        Returns:
            Lista que incluye la consulta original + n reformulaciones.
            Ej: ["guitarra acústica principiantes",
                 "guitarra clásica para aprender",
                 "instrumento de cuerdas para novatos",
                 "acoustic guitar starter beginner"]
            En caso de error, retorna solo [query] para no bloquear el flujo.
        """
        try:
            prompt = self.EXPANSION_PROMPT.format(query=query, n=n)
            response = self._model.generate_content(prompt)
            
            # Extraer el JSON de la respuesta (Gemini a veces añade ```json ... ```)
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            
            data = json.loads(raw_text.strip())
            expansions = data.get("expansions", [])
            
            # Retornar original + expansiones (sin duplicados)
            all_queries = [query] + [e for e in expansions if e.lower() != query.lower()]
            return all_queries

        except Exception as e:
            print(f"[QueryExpander] Error generando expansiones: {e}. Usando consulta original.")
            return [query]
