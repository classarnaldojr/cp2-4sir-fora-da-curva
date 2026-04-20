# Grupo TB — Checkpoint 2

## Visão geral

- **`attention_monitor.py`** — monitor de **atenção / fadiga** em tempo real (webcam): OpenCV + MediaPipe Face Landmarker, métricas (EAR, MAR, pose), níveis **verde / amarelo / vermelho**, score 0–100, calibração com **`c`**, malha opcional e alarme no vermelho.
- **`facemesh_conexoes.py`** — apenas as **arestas** da malha usadas para desenhar o rosto (para o ficheiro principal não ficar gigante).
- **`attention_monitor_lab.ipynb`** — notebook passo a passo (imports → modelo → um frame → métricas → overlay). Para demo estável em tempo real, use o `.py` localmente.

Recomenda-se executar a webcam **no computador** (ficheiro `.py`), não no Colab.

---

## Apresentação em 3 partes (3 alunos)

Cada pessoa cobre um “pedaço” do pipeline: **entrada → raciocínio → saída**. No final, façam **uma demo só** (por exemplo `python attention_monitor.py --debug`).

### Aluno 1 — Captura e MediaPipe (o “olho” do sistema)

- Problema em uma frase: monitorar sinais de atenção/fadiga em vídeo em tempo real.
- **Webcam**: `VideoCapture`, frame em BGR (OpenCV).
- **Pré-processamento**: BGR → RGB, `mp.Image`, modo **VIDEO** e **timestamp em ms** sempre a aumentar.
- **Modelo**: ficheiro `face_landmarker.task` (descarregado na primeira execução se não existir).
- **Saída do modelo**: landmarks do rosto; opcionalmente blendshapes (ex.: piscar).

**Perguntas típicas para este aluno:** como o vídeo entra? Por que RGB para o MediaPipe? O que é o `.task`?

---

### Aluno 2 — Métricas e regras (o “raciocínio”)

- **EAR** (olho aberto/fechado), **MAR** (boca/bocejo), **pose** da cabeça (pitch/yaw).
- **Calibração** com tecla **`c`**: guarda uma referência (“baseline”) de pitch/yaw olhando para o ecrã.
- **Heurística de confiança**: quando os dados dos olhos parecem pouco fiáveis (ex.: oclusão).
- **Decisão por tempo**: não basta um frame — acumulam-se **segundos** em cada condição para reduzir falsos alarmes.
- **Níveis**: verde (atento), amarelo (distração), vermelho (fadiga) e **score** numérico.

**Perguntas típicas para este aluno:** o que é EAR/MAR? Por que calibrar? Por que usar timers em segundos?

---

### Aluno 3 — Interface e demo (o “resultado visível”)

- **HUD**: barra de score, texto com métricas e instruções (`c` / `q`).
- **Malha** sobre o rosto (opcional: `--no-mesh`).
- **Alarme** no nível vermelho (opcional: `--no-audio`).
- **`--debug`**: imprime no terminal (~1×/s) métricas e nível — útil para a banca ver números sem só olhar para o vídeo.
- **Limitações** (breve): iluminação, ângulo, oclusão; ideias de melhoria (log CSV, ajuste fino de limiares).

**Perguntas típicas para este aluno:** o que o utilizador vê? Como demonstram que o sistema reage?

---

## Instalação e execução

1. **Criar a virtualenv**

   ```bash
   python3 -m venv venv
   ```

2. **Ativar o ambiente**

   - macOS / Linux:

     ```bash
     source venv/bin/activate
     ```

   - Windows (PowerShell):

     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

3. **Instalar dependências**

   ```bash
   pip install -r requirements.txt
   ```

4. **Correr o monitor** (na primeira execução descarrega `face_landmarker.task` se necessário)

   ```bash
   python attention_monitor.py
   ```

   - **`c`** — calibrar (~1,5 s olhando para o ecrã)  
   - **`q`** — sair  

   Opções úteis: `--no-mesh`, `--no-audio`, `--debug`.
