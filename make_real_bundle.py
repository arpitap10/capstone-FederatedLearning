"""Build the UI bundle from YOUR project (run in Colab inside capstone_project/src, with cueaudit installed).

    %cd /content/drive/MyDrive/capstone_project/src
    !pip -q install -e /content/drive/MyDrive/capstone_project/cueaudit[ui,report,bench]
    !python /content/drive/MyDrive/capstone_project/cueaudit/scripts/make_real_bundle.py \
        --fed main --global-tag main__fedavg__seed0 --out bundle.json --reference-pattern '*reliability*'
    !cueaudit ui --data bundle.json --share

What it does (nothing is re-implemented here):
  1. each hospital trains ONE local epoch from the trained global model using YOUR local_train.train_client;
  2. cueaudit.audit(...) audits that local model on the hospital's private validation loader (YOUR loaders);
  3. the reference + dose-response table come from runs/*/client_signals.csv via cueaudit;
  4. hero image = the first positive validation image of each hospital.
NOT tested in the sandbox where cueaudit was written (no GPU, no RSNA data): check the printed audit table first.
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed", default="main")
    ap.add_argument("--global-tag", default="main__fedavg__seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bundle.json")
    ap.add_argument("--reference-pattern", default="*reliability*")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    a = ap.parse_args()

    sys.path.insert(0, os.getcwd())                      # project src/
    import torch
    import config
    from config import TrainConfig
    from model import build_model, load_state
    from hospital_dataset import HospitalDataset, make_loader
    from local_train import train_client
    from utils import set_seed, get_device, load_json
    import cueaudit
    from cueaudit import AuditConfig, Reference
    from cueaudit.bundle import hospital_entry, make_bundle, save_bundle
    from cueaudit import benchmark as B
    from cueaudit.data import from_loader

    set_seed(a.seed)
    dev = get_device()
    tcfg = TrainConfig(); tcfg.local_epochs = 1; tcfg.pretrained = False; tcfg.early_stop_patience = 0
    gp = os.path.join(config.RUNS_DIR, a.global_tag, "global_model.pt")
    gstate = torch.load(gp, map_location="cpu")
    msd = build_model(pretrained=False).state_dict()
    if "classifier.weight" in gstate and "classifier.1.weight" in msd:       # head key naming differs between saves
        gstate["classifier.1.weight"] = gstate.pop("classifier.weight"); gstate["classifier.1.bias"] = gstate.pop("classifier.bias")

    man = load_json(os.path.join(config.FEDERATIONS_DIR, a.fed, "hospital_manifest.json"))["hospitals"]
    ref = Reference.from_runs(config.RUNS_DIR, a.reference_pattern, n_eff_nominal=54)
    bench = B.benchmark(config.RUNS_DIR, methods=["reliability"])["table"]
    cut = os.path.join(getattr(config, "METADATA_DIR", ""), "cue_cutpoints.json")
    cut = cut if os.path.exists(cut) else None
    if cut is None:
        print("! cue_cutpoints.json not found: image cues use each validation set's own median")

    hs = []
    for hid in sorted(man):
        res = train_client(gstate, hid, tcfg, dev, federation=a.fed)
        net = build_model(pretrained=False, dropout=0.0).to(dev); load_state(net, res["state"]); net.eval()
        loader = make_loader(HospitalDataset(hid, "val", federation=a.fed), tcfg.batch_size, shuffle=False, num_workers=tcfg.num_workers)
        val = from_loader(loader)
        rep = cueaudit.audit(net, val, cues=["view", "sex", "age_band", "frame_intensity", "brightness"], name=hid, reference=ref,
                             cutpoints=cut, device=str(dev), cfg=AuditConfig(n_bootstrap=a.n_bootstrap))
        print(f"{hid}: r={rep.r:.3f} leak={rep.signals['leak']:.3f} gap={rep.signals['gap']:.3f} ctx={rep.signals['ctx']:.3f} "
              f"verdict={rep.verdict.label if rep.verdict else '-'}")
        hero = val.images[int(np.argmax(val.labels == 1))]
        m = man[hid]
        hs.append(hospital_entry(hid, hid.split("_", 1)[-1], rep, hero, planted=bool(m.get("is_shortcut")),
                                 planted_cue=m.get("cue") or m.get("shortcut_cue"), n_train=res["n_samples"]))
    bundle = make_bundle(hs, ref, bench, synthetic=False, title="cueaudit", federation=a.fed,
                         note=f"local epoch from {a.global_tag}, seed {a.seed}")
    save_bundle(bundle, a.out)
    ranked = sorted(hs, key=lambda h: h["report"]["r"])
    n_pl = sum(1 for h in hs if h["planted"])
    print(f"\nlowest-r hospitals: {[h['id'] for h in ranked[:n_pl]]}  planted: {[h['id'] for h in hs if h['planted']]}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
