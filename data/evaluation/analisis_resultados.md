# Análisis de Resultados de Evaluación

Este documento detalla los resultados obtenidos al evaluar el sistema de Recuperación de Información Multimodal utilizando el conjunto de consultas anotadas manualmente (`qrels.json`) y explica el significado de cada métrica obtenida.

## 1. Resultados Globales (Promedios)

Los resultados se calcularon evaluando el sistema base (CLIP Bi-Encoder sin re-ranking) para observar su capacidad cruda de recuperación a distintos niveles de profundidad ($k$).

| Profundidad (k) | Precision@k | Recall@k | NDCG@k |
| :---: | :---: | :---: | :---: |
| **k = 1** | 0.6000 | 0.3392 | 0.6000 |
| **k = 3** | 0.3167 | 0.4883 | 0.5211 |
| **k = 5** | 0.2700 | 0.6650 | 0.5937 |
| **k = 10** | 0.1550 | 0.7750 | 0.6361 |

---

## 2. Significado e Interpretación de las Métricas

### Precision@k (Precisión en el Top-K)
**¿Qué mide?** De los $k$ documentos recuperados por el sistema, ¿qué porcentaje es realmente relevante?

- **Precision@1 (0.6000):** Significa que el 60% de las veces, el **primer** resultado que arroja el sistema es correcto. Es un número muy positivo para el sistema base.
- **Precision@10 (0.1550):** Cae drásticamente a medida que aumenta $k$. Esto es predecible y matemático: la cantidad de documentos relevantes por consulta en nuestro *qrels* es pequeña (usualmente 1 o 2). Si solo hay 1 producto relevante en la tienda y evaluamos en $k=10$, la precisión máxima teórica es del 10% (0.10).

### Recall@k (Exhaustividad en el Top-K)
**¿Qué mide?** Del total de documentos que sabemos que son relevantes, ¿qué porcentaje logró encontrar el sistema dentro de los primeros $k$ resultados?

- **Recall@1 (0.3392):** Solo un 34% de los documentos relevantes totales aparecen en la primera posición absoluta.
- **Recall@10 (0.7750):** Este es un **resultado excelente**. Significa que el sistema logra recuperar el **77.5% de los productos relevantes** dentro de los primeros 10 resultados. 
- **Interpretación:** Esto demuestra que CLIP es excepcional como motor de búsqueda crudo (alto recall). Es muy rápido encontrando lo que el usuario quiere y colocándolo "en algún lugar" del Top-10.

### NDCG@k (Normalized Discounted Cumulative Gain)
**¿Qué mide?** Evalúa la calidad del **orden (ranking)** de los resultados. Penaliza al sistema si los documentos más relevantes aparecen en posiciones muy bajas (por ejemplo, en la posición 10 en lugar de la 1). Toma valores entre 0 (pésimo) y 1 (perfecto).

- **NDCG@10 (0.6361):** Un valor de 0.63 indica que el ranking general es decente, pero que los productos relevantes a menudo están "enterrados" en las posiciones 3, 4, 5, etc., en lugar de estar anclados en el top 1 o 2.
- **Interpretación:** Como CLIP es un Bi-Encoder que compara vectores por distancia en un espacio muy comprimido (512 dimensiones), no tiene la capacidad de leer los matices lingüísticos finos para ordenar a la perfección.

---

## 3. Conclusiones y Decisiones de Arquitectura

El análisis de estos datos cuenta una historia muy clara:
El sistema tiene un **alto Recall@10** (encuentra los productos en la base de datos) pero una **baja Precision@3 y un NDCG@3 moderado** (no siempre los ordena bien al principio).

**La solución arquitectónica:**
Estos resultados justifican matemáticamente la implementación del **Cross-Encoder para Re-ranking**. 
En la versión avanzada del pipeline, le pedimos a CLIP que recupere 20 candidatos (aprovechando su alto Recall). Luego, el Cross-Encoder (un modelo pesado que sí lee los matices finos) re-ordena estos 20 candidatos en tiempo real, logrando que los productos correctos suban al Top-3, mejorando dramáticamente el NDCG percibido por el usuario.

### Comportamiento en Casos Atípicos (Puntaje 0)
Consultas como `"zelda nintendo game"` o `"PS5 stand with cooling fan"` obtuvieron un score de 0. Esto ocurre porque son consultas de "vocabulario cerrado" (marcas muy específicas o nombres propios). CLIP a veces sufre mapeando jerga muy técnica o marcas registradas si la imagen no es un claro reflejo del texto.
Para solucionar estos casos rebeldes, el sistema implementa la **Query Expansion**. Al pedirle a Gemini que reformule "zelda nintendo game" en sinónimos, aumentamos brutalmente las chances de atrapar el producto correcto en el primer intento.
