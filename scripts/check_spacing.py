#!/usr/bin/env python3
"""Check spacing and affine differences between fixed and moving volumes."""
import nibabel as nib
import numpy as np
import json
from pathlib import Path

data_root = Path('/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT')
with open(data_root / 'ThoraxCBCT_dataset.json') as f:
    ds = json.load(f)

pair = ds['training_paired_images'][0]
fixed_path = data_root / pair['fixed'].lstrip('./')
moving_path = data_root / pair['moving'].lstrip('./')

print('=== Fixed:', fixed_path.name, '===')
f_img = nib.load(str(fixed_path))
print('Shape:', f_img.shape)
print('Affine:')
print(f_img.affine)
print('Voxel sizes:', f_img.header.get_zooms())

print()
print('=== Moving:', moving_path.name, '===')
m_img = nib.load(str(moving_path))
print('Shape:', m_img.shape)
print('Affine:')
print(m_img.affine)
print('Voxel sizes:', m_img.header.get_zooms())

# Physical extent (in mm)
f_extent = np.array(f_img.shape) * np.array(f_img.header.get_zooms())
m_extent = np.array(m_img.shape) * np.array(m_img.header.get_zooms())
print()
print('=== Physical extent (mm) ===')
print('Fixed: ', f_extent)
print('Moving:', m_extent)
print('Difference:', f_extent - m_extent)

print()
print('=== Affine difference ===')
print(f_img.affine - m_img.affine)

# Physical origin
f_origin = f_img.affine[:3, 3]
m_origin = m_img.affine[:3, 3]
print()
print('=== Origins ===')
print('Fixed origin: ', f_origin)
print('Moving origin:', m_origin)
print('Origin shift: ', m_origin - f_origin, 'mm')
