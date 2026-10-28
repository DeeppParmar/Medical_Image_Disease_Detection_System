"""
Tests for legacy and direct model prediction endpoints:
- /predict/<model_name>
- /predict/all
"""

import io
from unittest.mock import MagicMock, patch


def test_predict_unknown_model_returns_404(client):
    """POST /predict/<model> for an unknown model should return 404."""
    response = client.post('/predict/nonexistent_model')
    assert response.status_code == 404
    data = response.get_json()
    assert 'error' in data


def test_predict_missing_file_returns_400(client):
    """POST /predict/<model> without file payload should return 400."""
    with patch('app.model_checkpoint_available', return_value=True):
        response = client.post('/predict/chexnet')
        assert response.status_code == 400
        data = response.get_json()
        assert data.get('error') == 'No file provided'


def test_predict_empty_filename_returns_400(client):
    """POST /predict/<model> with empty filename should return 400."""
    with patch('app.model_checkpoint_available', return_value=True):
        data = {'file': (io.BytesIO(b"dummy"), '')}
        response = client.post('/predict/chexnet', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        assert response.get_json().get('error') == 'No file selected'


def test_predict_unsupported_extension_returns_400(client):
    """POST /predict/<model> with disallowed extension should return 400."""
    with patch('app.model_checkpoint_available', return_value=True):
        data = {'file': (io.BytesIO(b"sample text content"), 'document.pdf')}
        response = client.post('/predict/chexnet', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        assert response.get_json().get('error') == 'File type not allowed'


def test_predict_unavailable_model_returns_503(client):
    """POST /predict/<model> when checkpoint is unavailable should return 503."""
    with patch('app.model_checkpoint_available', return_value=False):
        response = client.post('/predict/rsna')
        assert response.status_code == 503
        assert response.headers.get('X-Model-Used') == 'rsna'


def test_predict_success_with_mocked_predictor(client, valid_chest_xray_bytes):
    """POST /predict/<model> with valid image and mocked predictor returns 200."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {
        'findings': [{'disease': 'Infiltration', 'confidence': 78}],
        'top_prediction': 'Infiltration'
    }

    with patch('app.model_checkpoint_available', return_value=True), \
         patch('app.get_predictor', return_value=mock_predictor):
        data = {
            'file': (valid_chest_xray_bytes, 'chest_scan.png')
        }
        response = client.post('/predict/chexnet', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        res_data = response.get_json()
        assert res_data.get('success') is True
        assert res_data.get('model') == 'chexnet'
        assert response.headers.get('X-Model-Used') == 'chexnet'
        assert 'result' in res_data


def test_predict_predictor_exception_returns_500(client, valid_chest_xray_bytes):
    """POST /predict/<model> handles predictor failure gracefully with 500 status."""
    mock_predictor = MagicMock()
    mock_predictor.predict.side_effect = RuntimeError("Inference engine failure")

    with patch('app.model_checkpoint_available', return_value=True), \
         patch('app.get_predictor', return_value=mock_predictor):
        data = {'file': (valid_chest_xray_bytes, 'chest_scan.png')}
        response = client.post('/predict/chexnet', data=data, content_type='multipart/form-data')
        assert response.status_code == 500
        res_data = response.get_json()
        assert res_data.get('success') is False
        assert 'error' in res_data


def test_predict_all_missing_file_returns_400(client):
    """POST /predict/all without file should return 400."""
    response = client.post('/predict/all')
    assert response.status_code == 400
    assert response.get_json().get('error') == 'No file provided'


def test_predict_all_unsupported_file_returns_400(client):
    """POST /predict/all with invalid file type should return 400."""
    data = {'file': (io.BytesIO(b"data"), 'notes.docx')}
    response = client.post('/predict/all', data=data, content_type='multipart/form-data')
    assert response.status_code == 400


def test_predict_all_success_with_mocks(client, valid_chest_xray_bytes):
    """POST /predict/all runs mock inference across models and returns aggregated results."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {'status': 'processed'}

    with patch('app.model_checkpoint_available', return_value=True), \
         patch('app.get_predictor', return_value=mock_predictor):
        data = {'file': (valid_chest_xray_bytes, 'scan.png')}
        response = client.post('/predict/all', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        res_data = response.get_json()
        assert res_data.get('success') is True
        assert 'results' in res_data
