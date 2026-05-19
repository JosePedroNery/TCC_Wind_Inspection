import os
import gc
import torch
from pathlib import Path
from ultralytics import YOLO

# ==============================================================================
# 🛠️ CONFIGURAÇÃO DO CENÁRIO
# ==============================================================================
SCENARIO_TO_TRAIN = "scenario_04_synth_only" 

# ==============================================================================
# ⚙️ HIPERPARÂMETROS
# ==============================================================================
HYPERPARAMS = {
    "epochs": 100,
    "batch": 8,
    "imgsz": 640,
    "patience": 20,
    "seed": 42,
    "deterministic": True,
    "verbose": True,
    "amp": True,
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "lrf": 0.1,
    "cos_lr": True,
    "weight_decay": 0.0007,
    "warmup_epochs": 3.0,
    "box": 7.5, "cls": 0.5, "dfl": 1.5, "label_smoothing": 0.05,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "translate": 0.1, "scale": 0.5, "fliplr": 0.5,
    "mosaic": 1.0, "mixup": 0.1, "close_mosaic": 10,
}

MODEL_WEIGHTS = "yolo11s.pt"

# ==============================================================================
# 🚀 MOTOR DE TREINO INTELIGENTE (COM RESUME)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
FOLDS_DIR = BASE_DIR / "data/processed_folds"
RUNS_DIR = BASE_DIR / "runs/tcc_kfold" # Atenção ao caminho correto

def train_scenario():
    print(f"\n🔥 RECUPERANDO TREINO: {SCENARIO_TO_TRAIN}")
    print(f"   > GPU: {torch.cuda.get_device_name(0)}")
    
    scenario_path = FOLDS_DIR / SCENARIO_TO_TRAIN
    if not scenario_path.exists():
        print(f"❌ ERRO: Pasta não encontrada.")
        return

    folds = sorted([f for f in scenario_path.iterdir() if "fold_" in f.name])
    
    for fold_dir in folds:
        fold_name = fold_dir.name
        project_path = RUNS_DIR / SCENARIO_TO_TRAIN
        run_name = fold_name
        
        # Caminhos Críticos
        weights_dir = project_path / run_name / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        yaml_path = fold_dir / "data.yaml"

        # CASO 1: JÁ TERMINOU
        if best_pt.exists():
            print(f"   ✅ {fold_name} já concluído totalmente. Pulando...")
            continue
            
        print(f"\n   ▶️  Verificando {fold_name}...")
        
        try:
            # CASO 2: MORREU NO MEIO (RESUME)
            if last_pt.exists():
                print(f"      ⚠️ Encontrado treino interrompido! Resumindo de 'last.pt'...")
                # Para resumir, carregamos o PESO, não o modelo base
                model = YOLO(last_pt)
                
                # No resume, não passamos hiperparâmetros de novo (já estão salvos no .pt)
                model.train(resume=True)
            
            # CASO 3: COMEÇAR DO ZERO
            else:
                print(f"      ✨ Iniciando do Zero...")
                model = YOLO(MODEL_WEIGHTS)
                model.train(
                    data=str(yaml_path),
                    project=str(project_path),
                    name=run_name,
                    exist_ok=True,
                    **HYPERPARAMS
                )
            
            # Limpeza
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ ERRO NO {fold_name}: {e}")
            print("      Dica: Se o erro for de arquivo corrompido, apague a pasta desse fold e comece do zero.")
            continue

    print(f"\n🏆 FIM DO CENÁRIO {SCENARIO_TO_TRAIN}!")

if __name__ == "__main__":
    train_scenario()