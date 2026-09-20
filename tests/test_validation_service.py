import cv2
import numpy as np
import pytest
from services.validation_service import audit_splits, invalidate_validation, summarize_test, check_acceptance
from services.training_assistant import recommend_threshold_with_anomalies
from services.validation_service import artifact_identity, calibrate_pixel_gate, summarize_pixel_test


def test_split_audit_rejects_decoded_duplicates(tmp_path):
    for index, split in enumerate(('train', 'calibration', 'test')):
        folder = tmp_path / split / 'good'
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / 'sample.png'), np.full((24, 24, 3), index * 40, np.uint8))
    assert audit_splits(tmp_path)['counts'] == {'train': 1, 'calibration': 1, 'test': 1}
    cv2.imwrite(str(tmp_path / 'test/good/copy.png'), np.zeros((24, 24, 3), np.uint8))
    with pytest.raises(ValueError, match='Duplicate'):
        audit_splits(tmp_path)


def test_invalidation_removes_old_acceptance():
    recipe = {'validated': True, 'calibration': {}, 'acceptance_report': {}}
    invalidate_validation(recipe)
    assert not recipe['validated']
    assert 'calibration' not in recipe
    with pytest.raises(ValueError, match='independent'):
        check_acceptance(recipe)


def test_small_calibration_set_does_not_allow_more_than_one_percent():
    threshold, trace = recommend_threshold_with_anomalies([.1, .2, .8], [.3, .4, .9])
    assert trace['expected_false_alarms'] == 0
    assert threshold >= .8


def test_test_report_uses_frozen_threshold():
    assert summarize_test([('a', .1), ('b', .7)], .5)['rate'] == .5
    with pytest.raises(ValueError):
        summarize_test([('a', float('nan'))], .5)


def test_acceptance_expires_when_model_or_threshold_changes(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('patchcore.pt', 'memory_bank.pt', 'metadata.json'):
        (model / name).write_bytes(b'test artifact')
    data = tmp_path / 'data'
    for index, split in enumerate(('train', 'calibration', 'test')):
        folder = data / split / 'good'
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / 'sample.png'), np.full((24, 24, 3), index * 40, np.uint8))
    recipe = {'model': {'trained': True, 'path': str(model)},
              'prepared_dataset_path': str(data), 'anomaly_threshold': .5,
              'calibration': {'proposed': .5}}
    recipe['acceptance_report'] = {
        'model_identity': artifact_identity(recipe), 'dataset_audit': audit_splits(data),
        'threshold': .5, 'good': {'rate': 0.0}, 'synthetic': {'rate': 1.0},
        'pixel_gate': {'good': {'rate': 0.0}, 'synthetic': {'rate': 1.0}}}
    assert check_acceptance(recipe)
    recipe['anomaly_threshold'] = .6
    with pytest.raises(ValueError, match='threshold'):
        check_acceptance(recipe)
    recipe['anomaly_threshold'] = .5
    (model / 'memory_bank.pt').write_bytes(b'changed model')
    with pytest.raises(ValueError, match='model'):
        check_acceptance(recipe)


def test_pixel_gate_calibration_avoids_good_map_and_detects_local_defect():
    clean = np.zeros((32, 32), np.float32)
    defect = clean.copy()
    defect[10:15, 10:15] = 80.0
    gate = calibrate_pixel_gate(
        [('good', 0.1, clean)], [('defect', 0.1, defect)], 0.6
    )
    assert gate['expected_false_alarms'] == 0
    assert gate['synthetic_detection_rate'] == 1.0
    metrics = summarize_pixel_test([('defect', 0.1, defect)], 0.6, gate)
    assert metrics == {'count': 1, 'triggered': 1, 'rate': 1.0}
