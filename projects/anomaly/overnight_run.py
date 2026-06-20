# ============================================================================
#  CAMELYON17 — пълен пайплайн за през нощта (един cell)
#  Сваля по 1 слайд → извлича патчове → ТРИЕ слайда (пести диск) →
#  чекпойнт на всеки слайд в Drive (resume-safe) → тренира CNN/VGG/U-Net →
#  запазва всичко в Drive.
# ============================================================================
import os
import sys
import time
import json
import gc
import subprocess
import traceback

# ---- 0. зависимости ----
subprocess.run('apt-get -qq install -y openslide-tools', shell=True)
subprocess.run('pip -q install openslide-python awscli torchmetrics scikit-learn', shell=True)

from google.colab import drive
drive.mount('/content/drive')

OUT = '/content/drive/MyDrive/camelyon_run'      # всичко се пази тук (оцелява при срив)
CHUNKS = os.path.join(OUT, 'chunks')
os.makedirs(CHUNKS, exist_ok=True)

# ---- 1. код от репото ----
REPO_URL = 'https://github.com/plamena-ilieva/camelyon-anomaly.git'
REPO_DIR = '/content/camelyon-anomaly'
if os.path.exists(REPO_DIR):
    subprocess.run(['git', '-C', REPO_DIR, 'pull'])
else:
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR])
sys.path.insert(0, REPO_DIR)
for _m in [k for k in list(sys.modules) if k.split('.')[0] == 'projects']:
    del sys.modules[_m]

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from projects.anomaly import data, eda, models, train


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ---- 2. КОНФИГУРАЦИЯ (мащаб) ----
BUCKET, PREFIX = 's3://camelyon-dataset', 'CAMELYON17'
N_TUMOR_PATIENTS = 15      # различни пациенти с тумор
N_NORMAL_PATIENTS = 15     # различни нормални пациенти
N_PER_SLIDE = 1500         # патчове от всеки слайд
TILE_SIZE, LEVEL = 96, 2
EPOCHS, LR = 25, 5e-4
SEED = 23


def patient(n):
    return '_'.join(os.path.basename(n).split('_')[:2])


def s3_ls(p):
    out = subprocess.run(['aws', 's3', 'ls', '--no-sign-request', f'{BUCKET}/{p}'],
                         capture_output=True, text=True).stdout
    return [l.split()[-1] for l in out.splitlines() if l.split()]


def aws_cp(src, dst):
    subprocess.run(['aws', 's3', 'cp', '--no-sign-request', src, dst], check=True)


# ---- 3. избор на слайдове (различни пациенти, без leakage) ----
imgs = sorted(f for f in s3_ls(f'{PREFIX}/images/') if f.endswith('.tif'))
ann = {os.path.splitext(a)[0] for a in s3_ls(f'{PREFIX}/annotations/') if a.endswith('.xml')}
masks = {m[:-len('_mask.tif')] for m in s3_ls(f'{PREFIX}/masks/') if m.endswith('.tif')}

tumor, tpat = [], set()
for im in imgs:
    if os.path.splitext(im)[0] in ann and patient(im) not in tpat and len(tumor) < N_TUMOR_PATIENTS:
        tumor.append(im)
        tpat.add(patient(im))
normal, npat = [], set()
for im in imgs:
    b = os.path.splitext(im)[0]
    if (b not in ann and b not in masks and patient(im) not in tpat
            and patient(im) not in npat and len(normal) < N_NORMAL_PATIENTS):
        normal.append(im)
        npat.add(patient(im))
log('избрани:', len(tumor), 'tumor /', len(normal), 'normal слайда')

# ---- 4. извличане: сваляне → извличане → ТРИЕНЕ → чекпойнт чанк ----
RAW = '/content/raw'
os.makedirs(RAW, exist_ok=True)
jobs = [(im, 'tumor') for im in tumor] + [(im, 'normal') for im in normal]

for k, (im, kind) in enumerate(jobs, 1):
    chunk = os.path.join(CHUNKS, im + '.npz')
    if os.path.exists(chunk):
        log(f'[{k}/{len(jobs)}] {im} — вече готов, пропускам')
        continue
    tif = os.path.join(RAW, im)
    try:
        log(f'[{k}/{len(jobs)}] {kind} {im} — сваляне')
        aws_cp(f'{BUCKET}/{PREFIX}/images/{im}', tif)
        xmlp = None
        if kind == 'tumor':
            xmlp = os.path.join(RAW, os.path.splitext(im)[0] + '.xml')
            aws_cp(f'{BUCKET}/{PREFIX}/annotations/{os.path.splitext(im)[0]}.xml', xmlp)
        t0 = time.time()
        tiles = data.extract_tiles_balanced(
            tif, tile_size=TILE_SIZE, level=LEVEL, annotation_path=xmlp,
            n_tumor=(N_PER_SLIDE if kind == 'tumor' else 0),
            n_normal=(0 if kind == 'tumor' else N_PER_SLIDE), verbose=True)
        if tiles:
            patches = np.stack([t for t, _ in tiles])
            labels = np.array([l for _, l in tiles])
            np.savez_compressed(chunk, patches=patches, labels=labels, patient=patient(im))
        log(f'   +{len(tiles)} патча за {time.time() - t0:.0f}s -> чекпойнт записан')
    except Exception as e:
        log('   ГРЕШКА при', im, '->', repr(e))
        traceback.print_exc()
    finally:
        for f in os.listdir(RAW):  # ВАЖНО: освободи диска веднага
            os.remove(os.path.join(RAW, f))
        gc.collect()

# ---- 5. зареди всички чанкове ----
P, L, G = [], [], []
for fn in sorted(os.listdir(CHUNKS)):
    if not fn.endswith('.npz'):
        continue
    d = np.load(os.path.join(CHUNKS, fn), allow_pickle=True)
    P.extend(list(d['patches']))
    L.extend(list(d['labels']))
    G.extend([str(d['patient'])] * len(d['labels']))
labels = np.array(L)
groups = np.array(G)
log('ОБЩО патчове:', len(P), '| класове:', eda.class_distribution(labels),
    '| пациенти:', len(set(G)))

# ---- 6. EDA отчет + фигури ----
report = eda.summarize(P, labels)
with open(os.path.join(OUT, 'eda.json'), 'w') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
eda.plot_class_distribution(labels, os.path.join(OUT, 'class_distribution.png'))
eda.plot_sample_grid(P, list(labels), os.path.join(OUT, 'sample_grid.png'))
log('EDA:', report)

# ---- 7. patient-level split + DataLoader-и ----
train_idx, val_idx = data.grouped_train_val_split(labels, groups, val_fraction=0.25, seed=SEED)
train_ds = Subset(data.PatchDataset(P, labels, transform=data.default_transform(train=True)), train_idx)
val_ds = Subset(data.PatchDataset(P, labels, transform=data.default_transform(train=False)), val_idx)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
log('train патчове:', len(train_idx), '| val патчове:', len(val_idx))
log('val пациенти:', sorted(set(groups[val_idx].tolist())))

# ---- 8. трениране на класификаторите ----
results, trained = {}, {}
specs = [('SimpleCNN', lambda: models.SimpleCNN(num_classes=2)),
         ('VGG11', lambda: models.VGG(config='VGG11', num_classes=2)),
         ('VGG16', lambda: models.VGG(config='VGG16', num_classes=2))]
for name, build in specs:
    log(f'трениране {name} ...')
    m = build()
    h = train.fit(m, train_loader, val_loader, num_epochs=EPOCHS, lr=LR)
    results[name] = train.evaluate(m, val_loader)
    trained[name] = m
    torch.save({'arch': name, 'state_dict': m.state_dict()}, os.path.join(OUT, f'{name}.pt'))
    log(f'   {name}: {results[name]} | best epoch {h.get("best_epoch")}')


def save_outputs():
    """Запазва най-добрия класификатор + резултатите (извиква се рано и накрая)."""
    best = max(('SimpleCNN', 'VGG11', 'VGG16'), key=lambda n: results[n]['cohen_kappa'])
    torch.save({'arch': best, 'state_dict': trained[best].state_dict()},
               os.path.join(OUT, 'model.pt'))
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return best


# запази веднага щом класификаторите са готови (U-Net е бонус -> да не губим това)
best_name = save_outputs()
log('Класификаторите са запазени. Засега най-добър:', best_name, '|', results[best_name])

# ---- 9. U-Net автоенкодер ----
log('трениране U-Net (автоенкодер) ...')
recon_tf = data.transforms.Compose([data.transforms.ToTensor()])
normal_train = [P[i] for i in train_idx if labels[i] == data.NORMAL]
normal_loader = DataLoader(data.PatchDataset(normal_train, [0] * len(normal_train), transform=recon_tf),
                           batch_size=64, shuffle=True, num_workers=2)
device = train.get_device()
unet = models.UNetAutoencoder(in_channels=3, out_channels=3).to(device)
opt = torch.optim.AdamW(unet.parameters(), lr=1e-3)
crit = nn.MSELoss()
for epoch in range(EPOCHS):
    unet.train()
    run = 0.0
    for images, _ in normal_loader:
        images = images.to(device)
        opt.zero_grad()
        loss = crit(unet(images), images)
        loss.backward()
        opt.step()
        run += loss.item()
    if epoch % 5 == 0:
        log(f'   U-Net epoch {epoch}: recon MSE {run / len(normal_loader):.4f}')
torch.save({'arch': 'UNetAutoencoder', 'state_dict': unet.state_dict()}, os.path.join(OUT, 'UNet.pt'))

# U-Net AUC върху val (БАТЧОВО -> без CUDA OOM; в try, защото е бонус стъпка)
val_t = [P[i] for i in val_idx if labels[i] == data.TUMOR][:2000]
val_n = [P[i] for i in val_idx if labels[i] == data.NORMAL][:2000]


def rerr(pl, bs=128):
    out = []
    for i in range(0, len(pl), bs):
        batch = torch.stack([recon_tf(p.astype('uint8')) for p in pl[i:i + bs]])
        out.append(train.reconstruction_error(unet, batch))
    return torch.cat(out)


try:
    if val_t and val_n:
        et, en = rerr(val_t), rerr(val_n)
        yt = np.concatenate([np.ones(len(et)), np.zeros(len(en))])
        sc = np.concatenate([et.numpy(), en.numpy()])
        results['UNet_recon_AUC'] = float(roc_auc_score(yt, sc))
        log(f'   U-Net recon AUC: {results["UNet_recon_AUC"]:.3f}')
except Exception as e:
    log('   U-Net AUC пропуснат (грешка):', repr(e))

# ---- 10. финален запис ----
best_name = save_outputs()
log('ГОТОВО. Най-добър:', best_name, '|', results[best_name])
log('Всичко е в', OUT)
