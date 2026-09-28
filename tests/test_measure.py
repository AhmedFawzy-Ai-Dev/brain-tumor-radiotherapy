"""The measurement engine is pure geometry, so we can check it against phantoms
of exactly known size."""
import numpy as np

from brats_report.measure import measure_all, measure_region, region_mask
from brats_report.synthetic import ellipsoid, make_label

AX = ("R", "A", "S")  # identity affine orientation


def _ellipsoid_label(shape=(96, 96, 64), center=(48, 48, 32), semi=(14, 10, 8)):
    lab = np.zeros(shape, np.uint8)
    lab[ellipsoid(shape, center, semi)] = 1
    return lab


def test_volume_matches_analytic():
    lab = _ellipsoid_label(semi=(14, 10, 8))
    m = measure_region(lab, "WT", (1.0, 1.0, 1.0), AX)
    expected = (4 / 3) * np.pi * 14 * 10 * 8 / 1000  # cm3
    assert abs(m.volume_cm3 - expected) / expected < 0.05


def test_extents_map_to_anatomy():
    lab = _ellipsoid_label(semi=(14, 10, 8))
    m = measure_region(lab, "WT", (1.0, 1.0, 1.0), AX)
    assert m.extent_mm["LR"] == 29.0   # axis0 -> R
    assert m.extent_mm["AP"] == 21.0   # axis1 -> A
    assert m.extent_mm["SI"] == 17.0   # axis2 -> S


def test_max_diameter_and_recist():
    lab = _ellipsoid_label(semi=(14, 10, 8))
    m = measure_region(lab, "WT", (1.0, 1.0, 1.0), AX)
    assert abs(m.max_diameter_mm - 28.0) <= 1.0
    assert abs(m.recist_long_mm - 28.0) <= 1.0
    assert m.recist_short_mm <= m.recist_long_mm


def test_anisotropic_spacing_scales_extent():
    lab = _ellipsoid_label(semi=(14, 10, 8))
    m = measure_region(lab, "WT", (1.0, 1.0, 3.0), AX)  # 3 mm slices
    assert m.extent_mm["SI"] == 17 * 3.0
    # volume scales by the extra voxel size in z
    m1 = measure_region(lab, "WT", (1.0, 1.0, 1.0), AX)
    assert abs(m.volume_cm3 - 3 * m1.volume_cm3) / m.volume_cm3 < 0.02


def test_region_nesting():
    lab = make_label(shape=(96, 96, 64), center=(48, 48, 32), semi_axes=(14, 10, 8))
    wt = region_mask(lab, "WT").sum()
    tc = region_mask(lab, "TC").sum()
    et = region_mask(lab, "ET").sum()
    assert wt > tc > et > 0


def test_absent_region_reported():
    lab = np.zeros((32, 32, 32), np.uint8)  # empty
    res = measure_all(lab, (1.0, 1.0, 1.0), AX)
    assert res["WT"].present is False
    assert res["WT"].volume_cm3 == 0.0


def test_stray_island_does_not_stretch_diameters():
    # a 28 mm lesion plus one far-away voxel: the island adds to the volume, but
    # must not turn the diameters into the distance between the two
    lab = _ellipsoid_label(semi=(14, 10, 8))
    clean = measure_region(lab, "WT", (1.0, 1.0, 1.0), AX)
    lab[90, 90, 32] = 1
    m = measure_region(lab, "WT", (1.0, 1.0, 1.0), AX)
    assert m.components == 2 and clean.components == 1
    assert m.voxels == clean.voxels + 1
    assert m.recist_long_mm == clean.recist_long_mm
    assert m.max_diameter_mm == clean.max_diameter_mm
    assert m.extent_mm == clean.extent_mm


def test_second_large_region_is_flagged_not_hidden():
    from brats_report.report import _findings_bilingual

    speck = _ellipsoid_label(center=(30, 48, 32), semi=(10, 10, 8))
    speck[90, 90, 32] = 1
    two = _ellipsoid_label(center=(30, 48, 32), semi=(10, 10, 8))
    two[ellipsoid(two.shape, (75, 48, 32), (8, 8, 6))] = 1       # ~1.6 cm3, far away
    m_speck = measure_all(speck, (1.0, 1.0, 1.0), AX)
    m_two = measure_all(two, (1.0, 1.0, 1.0), AX)
    assert m_speck["WT"].large_pieces == 1
    assert m_two["WT"].large_pieces == 2 and m_two["WT"].components == 2
    notes = [en for en, _ in _findings_bilingual(m_two) if en.startswith("Note:")]
    assert len(notes) == 1 and "2 separate regions" in notes[0]
    assert not [en for en, _ in _findings_bilingual(m_speck) if en.startswith("Note:")]


def test_location_side():
    # centroid pushed toward +axis0 which is 'R' -> should read "right"
    lab = _ellipsoid_label(center=(70, 48, 32), semi=(8, 8, 8))
    m = measure_region(lab, "WT", (1.0, 1.0, 1.0), AX)
    assert "right" in m.location
