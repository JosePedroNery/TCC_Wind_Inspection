import os
import gc
import torch
from pathlib import Path
from ultralytics import YOLO

# ==============================================================================
# 🛠️ CONFIGURAÇÃO DO CENÁRIO (MUDE AQUI NOS OUTROS ARQUIVOS)
# ==============================================================================
# Opções possíveis:
# "scenario_01_baseline"
# "scenario_02_mix_15"
# "scenario_03_mix_30"
# "scenario_04_synth_only"

SCENARIO_TO_TRAIN = "scenario_03_mix_30"  # <--- SÓ MUDE ISSO!

# ==============================================================================
# ⚙️ HIPERPARÂMETROS (SEUS PARÂMETROS PERSONALIZADOS)
# ==============================================================================
HYPERPARAMS = {
    # --- Configuração Base ---
    "epochs": 100,
    "batch": 8,               # 8 é seguro para 12GB VRAM com modelo 's' + Augments
    "imgsz": 640,
    "patience": 20,           # Early Stopping
    "seed": 42,
    "deterministic": True,
    "verbose": True,
    "amp": True,              # Mixed Precision (Essencial na RTX 3060)

    # --- Otimizador (Foco em Estabilidade) ---
    "optimizer": "AdamW",
    "lr0": 0.0005,            # Learning Rate inicial menor para AdamW
    "lrf": 0.1,               # Decay final
    "cos_lr": True,           # Cosine Scheduler
    "weight_decay": 0.0007,   # Regularização
    "warmup_epochs": 3.0,

    # --- Perda (Loss) ---
    "box": 7.5,               
    "cls": 0.5,
    "dfl": 1.5,
    "label_smoothing": 0.05,  

    # --- Augmentations (O Segredo do TCC) ---
    "hsv_h": 0.015,           # Variação de cor
    "hsv_s": 0.7,             # Variação de saturação
    "hsv_v": 0.4,             # Variação de brilho
    "translate": 0.1,         # Translação
    "scale": 0.5,             # Zoom in/out
    "fliplr": 0.5,            # Espelhamento horizontal
    "mosaic": 1.0,            # Mosaico (Forte!)
    "mixup": 0.1,             # Mixup (Leve)
    "close_mosaic": 10,       # Desliga mosaico nas últimas 10 épocas
}

MODEL_WEIGHTS = "yolo11s.pt"  # Usando a versão 11s como pediu (mais moderna)

# ==============================================================================
# 🚀 MOTOR DE TREINO (NÃO PRECISA MEXER)
# ==============================================================================

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
FOLDS_DIR = BASE_DIR / "data/processed_folds"
RUNS_DIR = BASE_DIR / "runs/tcc_kfold" # Organizado em subpasta tcc_kfold

def train_scenario():
    print(f"\n🔥 INICIANDO TREINO DEDICADO: {SCENARIO_TO_TRAIN}")
    print(f"   > GPU: {torch.cuda.get_device_name(0)}")
    print(f"   > Modelo: {MODEL_WEIGHTS}")
    
    scenario_path = FOLDS_DIR / SCENARIO_TO_TRAIN
    if not scenario_path.exists():
        print(f"❌ ERRO: Pasta '{SCENARIO_TO_TRAIN}' não encontrada em {FOLDS_DIR}")
        return

    # Garante a ordem dos folds (0, 1, 2, 3, 4)
    folds = sorted([f for f in scenario_path.iterdir() if "fold_" in f.name])
    
    for fold_dir in folds:
        fold_name = fold_dir.name # ex: "fold_0"
        
        # Define saída: runs/tcc_kfold/scenario_X/fold_Y
        project_path = RUNS_DIR / SCENARIO_TO_TRAIN
        run_name = fold_name
        
        # Resume Check
        if (project_path / run_name / "weights" / "best.pt").exists():
            print(f"   ✅ {fold_name} já concluído. Pulando...")
            continue
            
        print(f"\n   ▶️  Executando {fold_name}...")
        
        try:
            # 1. Instancia modelo limpo
            model = YOLO(MODEL_WEIGHTS)
            
            # 2. Treina
            results = model.train(
                data=str(fold_dir / "data.yaml"),
                project=str(project_path),
                name=run_name,
                exist_ok=True,
                **HYPERPARAMS
            )
            
            # 3. Limpeza de VRAM
            del model
            del results
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ ERRO NO {fold_name}: {e}")
            continue

    print(f"\n🏆 FIM DO CENÁRIO {SCENARIO_TO_TRAIN}!")

if __name__ == "__main__":
    train_scenario()