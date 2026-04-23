# CP2 — Monitor de Atenção/Fadiga (OpenCV + MediaPipe)

Este projeto usa a webcam para estimar **atenção/fadiga** em tempo real. A ideia é simples: detectar o rosto, extrair alguns sinais fáceis de explicar e transformar isso em um **nível** (verde/amarelo/vermelho) e um **score** (0–100).

## O que tem aqui

- **`attention_monitor.py`**: aplicação principal (webcam + detecção + regras + interface).
- **`facemesh_conexoes.py`**: lista de conexões da malha facial (só para o arquivo principal não ficar enorme).
- **`attention_monitor_lab.ipynb`**: notebook “de laboratório” para testar as partes aos poucos.

## Como funciona (resumo)

- **Entrada**: webcam pelo OpenCV (`VideoCapture`).
- **Detecção**: MediaPipe **Face Landmarker** (arquivo `face_landmarker.task`).
- **Métricas**:
  - **EAR**: abertura dos olhos (ajuda a indicar olho fechado por tempo).
  - **MAR**: abertura da boca (ajuda a indicar bocejo).
  - **Pose** (pitch/yaw/roll): direção aproximada da cabeça.
- **Decisão por tempo**: em vez de “um frame”, a condição precisa durar alguns **segundos** para mudar o nível.
- **Calibração (`c`)**: salva um baseline de pose olhando para a tela para reduzir erro por posição natural da cabeça.

## Rodando no Windows (PowerShell)

1) Criar ambiente virtual:

```powershell
python -m venv venv
```

2) Ativar:

```powershell
.\venv\Scripts\Activate.ps1
```

3) Instalar dependências:

```powershell
pip install -r requirements.txt
```

4) Executar:

```powershell
python attention_monitor.py --debug
```

Teclas:
- **`c`**: calibra (~1,5s olhando para a tela)
- **`q`**: sai

Opções úteis:
- **`--no-mesh`**: não desenha a malha
- **`--no-audio`**: desliga o alarme
- **`--debug`**: imprime métricas no terminal (bom para apresentação)

## Roteiro rápido de demo (o que mostrar)

- **1)** Rodar com `--debug` e explicar que o terminal mostra as métricas e o nível.
- **2)** Apertar **`c`** olhando para a tela (calibração).
- **3)** Olhar para o lado por alguns segundos para cair no **amarelo**.
- **4)** Simular cansaço (piscar longo / fechar os olhos por mais tempo) para ir ao **vermelho**.
- **5)** Mostrar que `--no-mesh` e `--no-audio` mudam a interface sem mexer na lógica.

## Divisão da apresentação (Pedro, Lucas, Rafael)

- **Lucas (captura + MediaPipe)**: webcam, BGR→RGB, `mp.Image`, modo VIDEO e `detect_for_video`, arquivo `.task`.
- **Rafael (métricas + regras)**: EAR/MAR/pose, calibração, timers por segundos, níveis e score.
- **Pedro (interface + demo)**: HUD, malha opcional, argumentos `--no-mesh/--no-audio/--debug`, como conduzir a demonstração.
