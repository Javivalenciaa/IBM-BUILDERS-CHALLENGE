graph LR
    A[index.html] -->|HTTP Requests| B[main.py - FastAPI]
    B -->|Llama funciones| C[audio_optimizer.py]
    C -->|Retorna audio optimizado| B
    B -->|Respuesta JSON| A
