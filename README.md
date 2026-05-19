# Pipeline de Visão Computacional — TCC Wind Inspection

Este repositório contém a infraestrutura automatizada para geração de dados sintéticos, orquestração de cenários experimentais e treinamento de modelos de detecção de objetos (YOLO11) aplicados à inspeção visual de pás eólicas.

---

## 💡 Ideia Geral do Projeto

O objetivo central deste ecossistema é quantificar e avaliar o impacto da inclusão de **dados sintéticos** no treinamento de redes neurais profundas, buscando mitigar o *Sim2Real gap* (a diferença de desempenho do modelo entre o ambiente simulado e o real). 

O pipeline resolve o problema da escassez de dados reais de falhas criando um ecossistema controlado que:
1. Pega objetos isolados e gera milhares de imagens sintéticas realistas com inserção de ruído.
2. Corrige e unifica as classes de dados heterogêneos (`dirt` e `damage`).
3. Divide o ecossistema em 4 cenários experimentais usando **Validação Cruzada (5-Fold Cross-Validation)**.
4. Treina e valida os modelos de forma automatizada e isolada por cenário.

---

## 🛠️ Descrição dos Scripts

### 1. Geração de Dados — `01_augmentation.py` (Synthetic Data Factory)
Responsável pela criação em massa de imagens sintéticas robustas através da biblioteca *Albumentations*.
* **Recorte e Colagem Dinâmica:** Extrai objetos de interesse de imagens doadoras (`raw_sintetico`) e os renderiza randomicamente sobre fundos variados (`backgrounds`).
* **Estratégia Sim2Real (Blur/ISO):** Aplica transformações geométricas no objeto e, na composição final, injeta distorções térmicas e ópticas como `GaussianBlur`, `GaussNoise` e `ISONoise` para simular capturas de campo reais.
* **Volume de Produção:** Gera automaticamente **1.220 imagens sintéticas** com rótulos exportados diretamente no formato YOLO.
* **Injeção de Negativos:** Copia **50 imagens de fundo puro** (sem anotações) para o dataset para treinar a rede contra falsos positivos.

### 2. Organização do Dataset — `02_dataset_builder_kfold.py` (Arquiteto de Cenários)
Responsável por estruturar as imagens em partições robustas de testes e criar a matriz de cenários do TCC.
* **Validação Cruzada (5-Folds):** Divide os dados em 5 partições simétricas, isolando um conjunto de teste global fixo e híbrido (contendo 5% das imagens reais e sintéticas separadas na raiz).
* **Remapeamento de Classes:** Corrige o ID das anotações sintéticas (muda de `0` para `1`), alinhando o dataset com a lógica multiclase real: ID `0` para `dirt` (sujeira) e ID `1` para `damage` (dano).
* **Criação Automática dos 4 Cenários:**
  * `scenario_01_baseline`: Base de treino composta por 100% de dados Reais.
  * `scenario_02_mix_15`: Matriz híbrida contendo 15% de dados Sintéticos adicionados.
  * `scenario_03_mix_30`: Matriz híbrida contendo 30% de dados Sintéticos adicionados.
  * `scenario_04_synth_only`: Base de treino composta por 100% de dados Sintéticos.
* **Metadados:** Consolida arquivos `data.yaml` dinâmicos e independentes para cada um dos folds de cada cenário.

### 3. Execução dos Testes — `03_1` a `03_4` (`03_x_run_scenario_*.py`)
Conjunto de 4 scripts com arquitetura e hiperparâmetros idênticos, diferenciando-se apenas pelo cenário (`SCENARIO_TO_TRAIN`) que executam. Isso garante o isolamento total dos experimentos:
* **`03_1_run_scenario_01_baseline.py`:** Treinamento do Cenário 01 (Apenas Real).
* **`03_2_run_scenario_02_mix_15.py`:** Treinamento do Cenário 02 (Mix 15%).
* **`03_3_run_scenario_03_mix_30.py`:** Treinamento do Cenário 03 (Mix 30%).
* **`03_4_run_scenario_04_synth_only.py`:** Treinamento do Cenário 04 (Apenas Sintético).

**Características de Hardware e Motor de Treino (YOLO11):**
* **Arquitetura Base:** Instancia o modelo **YOLO11s** (`yolo11s.pt`) limpo do zero a cada início de fold.
* **Hiperparâmetros de Estabilidade:** Otimizador `AdamW` (LR inicial de `0.0005`), `batch=8` para estabilidade em placas de 12GB de VRAM, precisão mista automática (`amp=True`) e *Early Stopping* com paciência de 20 épocas.
* **Augmentation Nativo:** Aplicação de `mosaic=1.0` e `mixup=0.1`, programado para desativar os mosaicos nas últimas 10 épocas (`close_mosaic=10`) para refinar as caixas delimitadoras.
* **Resiliência a Falhas:** Mecanismo de *Resume Check* que identifica arquivos `best.pt` pré-existentes e pula automaticamente folds já computados, além de realizar a coleta de lixo e limpeza de cache da GPU (`cuda.empty_cache()`).

---

## 🚀 Como Executar o Pipeline

Os scripts seguem uma ordem cronológica de dependência. Execute-os na sequência abaixo:

1. **Gere os dados sintéticos combinados:**
   ```bash
   python scripts/02_dataset_builder_kfold.py
   
1. **Construa a árvore de diretórios e os folds de treino:**
   ```bash
   python scripts/01_augmentation.py
   
1. **Execute o cenário de treinamento desejado (Exemplo: Baseline):**
   ```bash
   python scripts/03_1_run_scenario_01_baseline.py
