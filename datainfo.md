data: /nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT
additional data: /nexus/posix0/MBR-neuralsystems/alim/regdata/processed
dino clone repo : /u/almik/feb25/dinov3
matcha clone reop : /u/almik/feb25/matcha
matcha pretrained weights:/u/almik/feb25/matcha/weights/matcha_pretrained.pth
dino pretrained weights:/u/almik/feb25/dino_weights
PYTHONPATH=/u/almik/feb25 python -m pipeline.scripts.run_pipeline \
    --pair 0 --split train --feature dinov3 --matcher nn --device cuda \
    --no-intensity-refine
