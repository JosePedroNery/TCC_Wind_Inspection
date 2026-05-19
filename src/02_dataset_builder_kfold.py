import os
import shutil
import random
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
SEED = 42
K_FOLDS = 5
TEST_SPLIT_RATIO = 0.05

# --- DEFINIÇÃO DE CLASSES (IMPORTANTE) ---
# A ordem aqui define os IDs: 0=dirt, 1=damage
CLASS_NAMES = ['dirt', 'damage'] 

# Mapeamento para correção do Sintético
# O sintético vem com ID 0 (pois só tem 1 classe). Precisamos jogar para ID 1 (damage).
SYNTH_REMAP_MAP = {0: 1} 

# --- CAMINHOS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

REAL_SOURCE = DATA_DIR / "raw_real"
SYNTH_SOURCE = DATA_DIR / "synthetic_generated_pool"
PROCESSED_DIR = DATA_DIR / "processed_folds"

# --- FUNÇÕES AUXILIARES ---

def get_image_label_pairs(source_dir):
    pairs = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(list(source_dir.rglob(ext)))
    
    images = sorted(images)

    for img_path in images:
        label_path = None
        possibles = [
            img_path.with_suffix('.txt'),
            img_path.parents[1] / "labels" / f"{img_path.stem}.txt",
            img_path.parent / "labels" / f"{img_path.stem}.txt"
        ]
        
        for p in possibles:
            if p.exists():
                label_path = p
                break
        
        if label_path and label_path.stat().st_size > 0:
            pairs.append((img_path, label_path))
            
    return pairs

def copy_and_process_data(pairs, destination_root, split_name, remap_labels=None):
    """
    Copia imagens e labels.
    Se 'remap_labels' for fornecido (ex: {0:1}), edita os labels durante a cópia.
    """
    dest_img_dir = destination_root / split_name / 'images'
    dest_lbl_dir = destination_root / split_name / 'labels'
    
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    desc = f"   > Processando {split_name}"
    if not pairs: return

    for img_src, lbl_src in tqdm(pairs, desc=desc, leave=False, unit="files"):
        # 1. Copia Imagem (Sempre igual)
        shutil.copy(img_src, dest_img_dir / img_src.name)
        
        # 2. Processa Label
        dst_lbl_path = dest_lbl_dir / lbl_src.name
        
        if remap_labels:
            # LER, EDITAR, SALVAR (Para o Sintético)
            try:
                with open(lbl_src, 'r') as f_in, open(dst_lbl_path, 'w') as f_out:
                    lines = f_in.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # CORREÇÃO AQUI: int(float(...)) aceita '0.0' e '0'
                            cls_id = int(float(parts[0]))
                            
                            # Aplica o remapping se necessário (0 -> 1)
                            new_cls_id = remap_labels.get(cls_id, cls_id)
                            
                            # Reconstrói a linha
                            new_line = f"{new_cls_id} " + " ".join(parts[1:]) + "\n"
                            f_out.write(new_line)
            except ValueError as e:
                print(f"⚠️ Erro ao ler label {lbl_src}: {e}")
                continue
        else:
            # CÓPIA DIRETA (Para o Real)
            shutil.copy(lbl_src, dst_lbl_path)

def create_yaml(save_path, train_path, val_path, test_path):
    # O YAML agora reflete a realidade COMPLEXA (2 classes)
    content = {
        'train': str(train_path.resolve()),
        'val': str(val_path.resolve()),
        'test': str(test_path.resolve()), 
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    with open(save_path / "data.yaml", 'w') as f:
        yaml.dump(content, f, default_flow_style=None)

# --- EXECUÇÃO PRINCIPAL ---

def main():
    print("🏗️  INICIANDO ARQUITETO K-FOLD (Correção de Classes 0=Dirt, 1=Damage)...")
    random.seed(SEED)
    np.random.seed(SEED)
    
    # 1. Carregar Dados
    print("   > Indexando Dados...")
    all_real = get_image_label_pairs(REAL_SOURCE)
    all_synth = get_image_label_pairs(SYNTH_SOURCE)
    random.shuffle(all_synth)

    # 2. Splits (Teste Fixo Híbrido)
    train_pool_real, test_set_real = train_test_split(all_real, test_size=TEST_SPLIT_RATIO, random_state=SEED)
    train_pool_synth, test_set_synth = train_test_split(all_synth, test_size=TEST_SPLIT_RATIO, random_state=SEED)
    
    GLOBAL_TEST_SET = test_set_real + test_set_synth 
    
    print(f"     ✅ Real (Treino): {len(train_pool_real)} | Teste: {len(test_set_real)}")
    print(f"     ✅ Synth (Treino): {len(train_pool_synth)} | Teste: {len(test_set_synth)}")

    # 3. Definição dos Cenários
    n_real = len(train_pool_real)
    n_synth_15 = int(np.ceil(n_real / 0.85) - n_real)
    n_synth_30 = int(np.ceil(n_real / 0.70) - n_real)
    
    if n_synth_30 > len(train_pool_synth):
        n_synth_30 = len(train_pool_synth)

    # Dicionário de cenários
    # Cada tupla é: (dataset_list, precisa_remapear_boolean)
    scenarios = {
        "scenario_01_baseline": {
            "parts": [ (train_pool_real, False) ], 
            "description": "100% Real"
        },
        "scenario_02_mix_15": {
            "parts": [ 
                (train_pool_real, False), 
                (train_pool_synth[:n_synth_15], True) 
            ],
            "description": "Mix 15%"
        },
        "scenario_03_mix_30": {
            "parts": [ 
                (train_pool_real, False),
                (train_pool_synth[:n_synth_30], True)
            ],
            "description": "Mix 30%"
        },
        "scenario_04_synth_only": {
            "parts": [ (train_pool_synth[:n_synth_30], True) ],
            "description": "100% Sintético"
        }
    }
    
    if PROCESSED_DIR.exists(): shutil.rmtree(PROCESSED_DIR)
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    
    print(f"\n🚀 Gerando Folds com correção de classes...")
    
    for name, data in scenarios.items():
        print(f"\n📂 CENÁRIO: {name}")
        
        # Unifica pools mantendo flag de remap
        full_dataset_objs = []
        for subset, should_remap in data['parts']:
            for pair in subset:
                full_dataset_objs.append({'pair': pair, 'remap': should_remap})
        
        full_dataset_objs = np.array(full_dataset_objs) 
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset_objs)):
            fold_dir = PROCESSED_DIR / name / f"fold_{fold_idx}"
            
            # --- PROCESSAMENTO DO TREINO/VAL ---
            for split, idxs in [('train', train_idx), ('val', val_idx)]:
                subset_objs = full_dataset_objs[idxs]
                
                to_remap = [obj['pair'] for obj in subset_objs if obj['remap']]
                no_remap = [obj['pair'] for obj in subset_objs if not obj['remap']]
                
                # Copia Real (Sem remap)
                copy_and_process_data(no_remap, fold_dir, split, remap_labels=None)
                # Copia Sintético (Com remap 0->1)
                copy_and_process_data(to_remap, fold_dir, split, remap_labels=SYNTH_REMAP_MAP)

            # --- PROCESSAMENTO DO TESTE (Global) ---
            copy_and_process_data(test_set_real, fold_dir, 'test', remap_labels=None)
            copy_and_process_data(test_set_synth, fold_dir, 'test', remap_labels=SYNTH_REMAP_MAP)
            
            create_yaml(
                fold_dir, 
                fold_dir / 'train' / 'images',
                fold_dir / 'val' / 'images',
                fold_dir / 'test' / 'images'
            )
            print(f"   ✅ Fold {fold_idx} OK")

    print(f"\n✨ ESTRUTURA FINALIZADA!")
    print(f"   Classes Config: {CLASS_NAMES}")
    print(f"   Synth Remap: {SYNTH_REMAP_MAP}")

if __name__ == "__main__":
    main()