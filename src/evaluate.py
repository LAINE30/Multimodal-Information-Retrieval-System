"""
Script para ejecutar la evaluación completa del sistema de recuperación.
Carga las consultas de evaluación (qrels), ejecuta cada query contra el
MultimodalRetriever, y calcula las métricas Precision@k, Recall@k y NDCG@k.
Genera un reporte en consola y guarda los resultados en un archivo JSON.
"""
import json
import os
import sys

# Asegurar que el directorio raíz esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import MultimodalRetriever
from src.evaluation_metrics import precision_at_k, recall_at_k, ndcg_at_k

QRELS_PATH = "data/evaluation/qrels.json"
RESULTS_PATH = "data/evaluation/evaluation_results.json"
# Valores de k a evaluar (el profesor pide Precision@k, Recall@k, NDCG@k)
K_VALUES = [1, 3, 5, 10]


def load_qrels(path: str) -> list:
    """Carga las queries de evaluación desde el archivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def run_evaluation():
    """
    Ejecuta la evaluación completa:
    1. Carga las qrels (queries + documentos relevantes conocidos).
    2. Inicializa el MultimodalRetriever (CLIP + ChromaDB).
    3. Para cada query, ejecuta la búsqueda y calcula las métricas.
    4. Calcula promedios globales (MAP-like).
    5. Imprime el reporte y guarda los resultados.
    """
    # 1. Cargar queries de evaluación
    queries = load_qrels(QRELS_PATH)
    print(f"Cargadas {len(queries)} consultas de evaluación desde {QRELS_PATH}\n")

    # 2. Inicializar el retriever
    print("Inicializando MultimodalRetriever (CLIP + ChromaDB)...")
    retriever = MultimodalRetriever()
    print("¡Listo!\n")

    # 3. Evaluar cada query
    all_results = []
    # Acumuladores para promedios por cada k
    avg_metrics = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}

    max_k = max(K_VALUES)

    print("=" * 80)
    print(f"{'QUERY':<50} | {'P@3':>6} | {'R@3':>6} | {'NDCG@3':>7}")
    print("=" * 80)

    for q in queries:
        query_id = q["query_id"]
        query_text = q["query_text"]
        relevant_ids = set(q["relevant_doc_ids"])

        # Si no hay documentos relevantes anotados, saltamos métricas de recall/ndcg
        # pero igual ejecutamos la query para ver qué devuelve
        if len(relevant_ids) == 0:
            print(f"  {query_id}: {query_text:<44} | (sin qrels anotados, omitida)")
            continue

        # Ejecutar la búsqueda con el top_k más alto que necesitemos
        results = retriever.retrieve(query_text, top_k=max_k)
        retrieved_ids = [r["id"] for r in results]

        # Calcular métricas para cada valor de k
        query_metrics = {"query_id": query_id, "query_text": query_text, "relevant_count": len(relevant_ids)}
        
        for k in K_VALUES:
            p = precision_at_k(retrieved_ids, relevant_ids, k)
            r = recall_at_k(retrieved_ids, relevant_ids, k)
            n = ndcg_at_k(retrieved_ids, relevant_ids, k)
            
            query_metrics[f"precision@{k}"] = round(p, 4)
            query_metrics[f"recall@{k}"] = round(r, 4)
            query_metrics[f"ndcg@{k}"] = round(n, 4)

            avg_metrics[k]["precision"].append(p)
            avg_metrics[k]["recall"].append(r)
            avg_metrics[k]["ndcg"].append(n)

        # Imprimir resumen de la query (usando k=3 como referencia principal)
        p3 = query_metrics.get("precision@3", 0)
        r3 = query_metrics.get("recall@3", 0)
        n3 = query_metrics.get("ndcg@3", 0)
        print(f"  {query_id}: {query_text[:44]:<44} | {p3:>6.4f} | {r3:>6.4f} | {n3:>7.4f}")
        
        # Guardar los IDs recuperados para referencia
        query_metrics["retrieved_ids"] = retrieved_ids[:max_k]
        all_results.append(query_metrics)

    # 4. Calcular promedios globales
    print("\n" + "=" * 80)
    print("PROMEDIOS GLOBALES (Mean Average)")
    print("=" * 80)

    summary = {}
    for k in K_VALUES:
        if len(avg_metrics[k]["precision"]) > 0:
            mean_p = sum(avg_metrics[k]["precision"]) / len(avg_metrics[k]["precision"])
            mean_r = sum(avg_metrics[k]["recall"]) / len(avg_metrics[k]["recall"])
            mean_n = sum(avg_metrics[k]["ndcg"]) / len(avg_metrics[k]["ndcg"])
        else:
            mean_p = mean_r = mean_n = 0.0

        summary[f"mean_precision@{k}"] = round(mean_p, 4)
        summary[f"mean_recall@{k}"] = round(mean_r, 4)
        summary[f"mean_ndcg@{k}"] = round(mean_n, 4)

        print(f"  k={k:>2}: Precision={mean_p:.4f} | Recall={mean_r:.4f} | NDCG={mean_n:.4f}")

    print("=" * 80)

    # 5. Guardar resultados
    output = {
        "summary": summary,
        "per_query_results": all_results,
        "k_values_evaluated": K_VALUES,
        "total_queries_evaluated": len(all_results)
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResultados guardados en: {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
