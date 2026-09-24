import numpy as np
import pytest

pytest.importorskip('PySide6')

from main import InspectionWorker
from services.verdict_service import evaluate_verdict, InspectionError


GOOD_MAP = np.zeros((16, 16), np.float32)
RECIPE = {'recipe_name': 'R', 'anomaly_threshold': 0.6,
          'top_mark_template': 'top.jpg', 'model': {'path': 'model'}}
POINTS = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], np.float32)
FRAME = np.zeros((100, 100, 3), np.uint8)


@pytest.mark.parametrize('score,amap,code', [
    (None, GOOD_MAP, 'INVALID_SCORE'),
    (float('nan'), GOOD_MAP, 'INVALID_SCORE'),
    (0.1, None, 'INVALID_ANOMALY_MAP'),
    (0.1, np.empty((0, 0)), 'INVALID_ANOMALY_MAP'),
    (0.1, np.full((4, 4), np.nan), 'INVALID_ANOMALY_MAP'),
])
def test_invalid_prediction_is_error_not_defect(score, amap, code):
    with pytest.raises(InspectionError) as exc:
        evaluate_verdict(score, amap, 0.6, {})
    assert exc.value.code == code


def test_good_and_anomaly_keep_quality_verdict():
    assert evaluate_verdict(0.1, GOOD_MAP, 0.6, {})[0] is False
    assert evaluate_verdict(0.7, GOOD_MAP, 0.6, {})[0] is True


@pytest.fixture
def worker(monkeypatch):
    instance = InspectionWorker(detection_model=None)
    instance.current_recipe = dict(RECIPE)
    instance.patchcore_service.model = object()
    monkeypatch.setattr(instance, 'cached_orientation', lambda *args: (False, 0, ''))
    monkeypatch.setattr(instance, 'check_laser_mark', lambda roi: ('MATCH', ''))
    return instance


def test_missing_model_is_error(worker):
    worker.patchcore_service.model = None
    assert worker.inspect(FRAME, POINTS)[0:2] == ('ERROR', 0.0)


def test_missing_model_stays_error_when_no_box_is_detected(worker, monkeypatch):
    statuses = []
    worker.status_update.connect(statuses.append)
    worker.patchcore_service.model = None
    monkeypatch.setattr(worker, 'find_window', lambda: type('W', (), {'_hWnd': None})())
    monkeypatch.setattr(worker, 'grab_frame', lambda w: (FRAME, (0, 0, 100, 100)))
    monkeypatch.setattr(worker, 'detect', lambda frame: (None, 30.0))
    monkeypatch.setattr(worker, 'parse_boxes', lambda result: [])
    worker.tick()
    assert statuses[-1]['inspection_result'] == 'ERROR'


@pytest.mark.parametrize('points', [np.zeros((4, 2), np.float32),
                                     np.full((4, 2), np.nan, np.float32)])
def test_invalid_roi_is_error(worker, points):
    assert worker.inspect(FRAME, points)[0] == 'ERROR'


def test_missing_threshold_is_error(worker):
    worker.current_recipe.pop('anomaly_threshold')
    assert worker.inspect(FRAME, POINTS)[0] == 'ERROR'


def test_preprocessing_size_is_validated(worker):
    worker.target_width = 0
    assert worker.inspect(FRAME, POINTS)[0] == 'ERROR'


def test_valid_live_good_and_anomaly(worker, monkeypatch):
    monkeypatch.setattr(worker.patchcore_service, 'predict_full', lambda roi: (0.1, GOOD_MAP))
    assert worker.inspect(FRAME, POINTS, render_heatmap=False)[0] == 'PASS'
    monkeypatch.setattr(worker.patchcore_service, 'predict_full', lambda roi: (0.7, GOOD_MAP))
    assert worker.inspect(FRAME, POINTS, render_heatmap=False)[0] == 'FAIL'


@pytest.mark.parametrize('prediction', [
    (0.1, None), (0.1, np.empty((0, 0))),
    (0.1, np.full((4, 4), np.nan)), (float('nan'), GOOD_MAP),
])
def test_invalid_live_prediction_is_error(worker, monkeypatch, prediction):
    monkeypatch.setattr(worker.patchcore_service, 'predict_full', lambda roi: prediction)
    assert worker.inspect(FRAME, POINTS, render_heatmap=False)[0] == 'ERROR'


def test_inference_and_decision_exceptions_are_error(worker, monkeypatch):
    def broken(*args):
        raise RuntimeError('broken')
    monkeypatch.setattr(worker.patchcore_service, 'predict_full', broken)
    assert worker.inspect(FRAME, POINTS)[0] == 'ERROR'
    monkeypatch.setattr(worker.patchcore_service, 'predict_full', lambda roi: (0.1, GOOD_MAP))
    monkeypatch.setattr('main.evaluate_verdict', broken)
    assert worker.inspect(FRAME, POINTS)[0] == 'ERROR'


def test_missing_top_mark_template_is_inspection_error(worker, monkeypatch):
    monkeypatch.setattr(worker, 'cached_orientation', lambda *a: (False, 0, 'template missing'))
    assert worker.inspect(FRAME, POINTS)[0] == 'ERROR'


def test_marking_unavailable_is_error_but_mismatch_is_fail(worker, monkeypatch):
    monkeypatch.setattr(worker.patchcore_service, 'predict_full', lambda roi: (0.1, GOOD_MAP))
    monkeypatch.setattr(worker, 'check_laser_mark', lambda roi: ('ERROR', 'template unreadable'))
    assert worker.inspect(FRAME, POINTS)[0] == 'ERROR'
    monkeypatch.setattr(worker, 'check_laser_mark', lambda roi: ('FAIL', 'mark mismatch'))
    assert worker.inspect(FRAME, POINTS)[0] == 'FAIL'


@pytest.mark.parametrize('match_score', [float('nan'), -999.0])
def test_unusable_orientation_match_cannot_pass(worker, monkeypatch, match_score):
    monkeypatch.setattr(worker.patchcore_service, 'predict_full', lambda roi: (0.1, GOOD_MAP))
    monkeypatch.setattr(worker, 'cached_orientation',
                        lambda roi, points, key: worker.check_orientation(roi))
    monkeypatch.setattr(worker.top_mark_service, 'detect_orientation',
                        lambda roi, path: (0, match_score, 0.0))
    assert worker.inspect(FRAME, POINTS)[0] == 'ERROR'


def test_stale_pass_is_not_reported_as_current_pass(worker, monkeypatch):
    statuses = []
    worker.status_update.connect(statuses.append)
    worker.last_boxes = [dict(points=POINTS, conf=0.99, label='R', result='PASS', score=0.1)]
    worker.last_detect_time = __import__('time').time()
    monkeypatch.setattr(worker, 'find_window', lambda: type('W', (), {'_hWnd': None})())
    monkeypatch.setattr(worker, 'grab_frame', lambda w: (FRAME, (0, 0, 100, 100)))
    monkeypatch.setattr(worker, 'detect', lambda frame: (None, 30.0))
    monkeypatch.setattr(worker, 'parse_boxes', lambda result: [])
    worker.tick()
    assert statuses[-1]['inspection_result'] == 'ERROR'


def test_unknown_box_result_cannot_aggregate_to_pass(worker, monkeypatch):
    statuses = []
    worker.status_update.connect(statuses.append)
    monkeypatch.setattr(worker, 'find_window', lambda: type('W', (), {'_hWnd': None})())
    monkeypatch.setattr(worker, 'grab_frame', lambda w: (FRAME, (0, 0, 100, 100)))
    monkeypatch.setattr(worker, 'detect', lambda frame: (None, 30.0))
    monkeypatch.setattr(worker, 'parse_boxes', lambda result: [dict(points=POINTS, conf=.99, label='R')])
    monkeypatch.setattr(worker, 'inspect', lambda *a, **kw: ('UNEXPECTED', .1, 'bad state', False, None))
    worker.tick()
    assert statuses[-1]['inspection_result'] == 'ERROR'


def test_segment_exception_cannot_leave_prior_pass_visible(worker, monkeypatch):
    statuses = []
    worker.status_update.connect(statuses.append)
    monkeypatch.setattr(worker, 'find_window', lambda: type('W', (), {'_hWnd': None})())
    monkeypatch.setattr(worker, 'grab_frame', lambda w: (FRAME, (0, 0, 100, 100)))
    monkeypatch.setattr(worker, 'detect', lambda frame: (None, 30.0))
    monkeypatch.setattr(worker, 'parse_boxes', lambda result: [dict(points=POINTS, conf=.99, label='R')])
    monkeypatch.setattr(worker, 'inspect', lambda *a, **kw: ('PASS', .1, '', False, None))
    monkeypatch.setattr(worker, '_segment_contours', lambda *a: (_ for _ in ()).throw(RuntimeError('segment failed')))
    worker.tick()
    assert statuses[-1]['inspection_result'] == 'ERROR'
