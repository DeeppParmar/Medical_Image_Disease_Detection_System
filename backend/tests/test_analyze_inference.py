"""
Inference routing and prediction flow tests for POST /api/analyze.
Uses mocks to verify routing, response formatting, and error scenarios
without loading heavy neural network weights or making network calls.
"""

from unittest.mock import MagicMock, patch


def test_analyze_explicit_mura_model_success(client, valid_bone_xray_bytes):
    """POST /api/analyze with explicit model=mura routes to MURA predictor."""
    mock_predictor = MagicMock()
    mock_predictor.predict_for_frontend.return_value = [{
        'disease': 'Fracture',
        'confidence': 88,
        'status': 'warning',
        'description': 'Possible bone fracture detected.',
        'regions': ['radius'],
    }]

    with patch('app.validate_medical_image', return_value={'valid': True}), \
         patch('app.model_checkpoint_available', return_value=True), \
         patch('app.get_predictor', return_value=mock_predictor), \
         patch('app._attach_heatmap', side_effect=lambda res, *args, **kwargs: res):

        data = {
            'image': (valid_bone_xray_bytes, 'wrist_xray.png'),
            'model': 'mura',
            'scan_type': 'bone'
        }
        response = client.post('/api/analyze', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        assert response.headers.get('X-Model-Used') == 'mura'
        res_list = response.get_json()
        assert isinstance(res_list, list)
        assert len(res_list) == 1
        assert res_list[0]['disease'] == 'Fracture'
        assert res_list[0]['confidence'] == 88


def test_analyze_explicit_tb_model_success(client, valid_chest_xray_bytes):
    """POST /api/analyze with explicit model=tuberculosis routes to TB predictor."""
    mock_predictor = MagicMock()
    mock_predictor.predict_for_frontend.return_value = [{
        'disease': 'Tuberculosis',
        'confidence': 92,
        'status': 'critical',
        'description': 'Signs consistent with active pulmonary tuberculosis.',
        'regions': ['apical']
    }]

    with patch('app.validate_medical_image', return_value={'valid': True}), \
         patch('app.model_checkpoint_available', return_value=True), \
         patch('app.get_predictor', return_value=mock_predictor), \
         patch('app._attach_heatmap', side_effect=lambda res, *args, **kwargs: res):

        data = {
            'image': (valid_chest_xray_bytes, 'chest_tb.png'),
            'model': 'tuberculosis',
            'scan_type': 'chest'
        }
        response = client.post('/api/analyze', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        assert response.headers.get('X-Model-Used') == 'tuberculosis'
        res_list = response.get_json()
        assert res_list[0]['disease'] == 'Tuberculosis'


def test_analyze_explicit_chexnet_model_success(client, valid_chest_xray_bytes):
    """POST /api/analyze with explicit model=chexnet returns multi-disease predictions."""
    mock_predictor = MagicMock()
    mock_predictor.predict_for_frontend.return_value = [
        {'disease': 'Pneumonia', 'confidence': 85, 'status': 'critical', 'description': 'Infiltration'},
        {'disease': 'Infiltration', 'confidence': 60, 'status': 'warning', 'description': 'Lung consolidation'}
    ]

    with patch('app.validate_medical_image', return_value={'valid': True}), \
         patch('app.model_checkpoint_available', return_value=True), \
         patch('app.get_predictor', return_value=mock_predictor), \
         patch('app._attach_heatmap', side_effect=lambda res, *args, **kwargs: res):

        data = {
            'image': (valid_chest_xray_bytes, 'chest_pneumonia.png'),
            'model': 'chexnet',
            'scan_type': 'chest'
        }
        response = client.post('/api/analyze', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        assert response.headers.get('X-Model-Used') == 'chexnet'
        res_list = response.get_json()
        assert len(res_list) == 2
        assert res_list[0]['disease'] == 'Pneumonia'


def test_analyze_model_unavailable_returns_503(client, valid_chest_xray_bytes):
    """POST /api/analyze returns 503 when requested model checkpoint is unavailable."""
    with patch('app.validate_medical_image', return_value={'valid': True}), \
         patch('app.model_checkpoint_available', return_value=False):

        data = {
            'image': (valid_chest_xray_bytes, 'chest.png'),
            'model': 'chexnet',
            'scan_type': 'chest'
        }
        response = client.post('/api/analyze', data=data, content_type='multipart/form-data')

        assert response.status_code == 503
        data = response.get_json()
        assert data.get('error') == 'model_not_loaded'


def test_analyze_model_failure_returns_500(client, valid_chest_xray_bytes):
    """POST /api/analyze returns 500 when predictor raises an unhandled exception."""
    mock_predictor = MagicMock()
    mock_predictor.predict_for_frontend.side_effect = RuntimeError("GPU memory exhausted")

    with patch('app.validate_medical_image', return_value={'valid': True}), \
         patch('app.model_checkpoint_available', return_value=True), \
         patch('app.get_predictor', return_value=mock_predictor):

        data = {
            'image': (valid_chest_xray_bytes, 'chest.png'),
            'model': 'chexnet',
            'scan_type': 'chest'
        }
        response = client.post('/api/analyze', data=data, content_type='multipart/form-data')

        assert response.status_code == 500
        data = response.get_json()
        assert data.get('error') == 'processing_failed'
