"""
Validation and error handling tests for POST /api/analyze endpoint.
"""

import io
from unittest.mock import patch


def test_analyze_missing_file_returns_400(client):
    """POST /api/analyze without file or image in request returns 400."""
    response = client.post('/api/analyze')
    assert response.status_code == 400
    data = response.get_json()
    assert data.get('error') == 'unsupported_scan'
    assert 'No image file provided' in data.get('message', '')


def test_analyze_empty_filename_returns_400(client):
    """POST /api/analyze with empty filename returns 400."""
    data = {'image': (io.BytesIO(b"content"), '')}
    response = client.post('/api/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    data = response.get_json()
    assert data.get('error') == 'unsupported_scan'
    assert 'No file selected' in data.get('message', '')


def test_analyze_unsupported_file_extension_returns_415(client):
    """POST /api/analyze with disallowed extension returns 415."""
    data = {'image': (io.BytesIO(b"plain text document"), 'scan_notes.txt')}
    response = client.post('/api/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 415
    data = response.get_json()
    assert data.get('error') == 'unsupported_scan'
    assert 'File type not allowed' in data.get('message', '')


def test_analyze_corrupted_image_returns_415(client, corrupted_image_bytes):
    """POST /api/analyze with corrupted non-decodable image returns 415."""
    data = {'image': (corrupted_image_bytes, 'corrupted_scan.png')}
    response = client.post('/api/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 415
    data = response.get_json()
    assert data.get('error') == 'unsupported_scan'
    assert 'Cannot decode image' in data.get('message', '')


def test_analyze_low_resolution_image_returns_422(client, tiny_image_bytes):
    """POST /api/analyze with resolution below 32x32 returns 422."""
    data = {'image': (tiny_image_bytes, 'too_small.png')}
    response = client.post('/api/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 422
    data = response.get_json()
    assert data.get('error') == 'resolution_too_low'
    assert 'Minimum 32x32 required' in data.get('message', '')


def test_analyze_non_medical_image_rejected_422(client, non_medical_image_bytes):
    """POST /api/analyze with a non-medical image returns 422 rejecting non-medical content."""
    # When medical validator returns valid=False
    mock_val_result = {
        'valid': False,
        'reason': 'non_medical_image',
        'sub_reason': 'High color variance and non-medical texture',
        'message': 'Uploaded image does not appear to be a valid medical scan.',
        'confidence': 0.12,
        'heuristic_score': 0.15,
        'cnn_probability': 0.05
    }
    with patch('app.validate_medical_image', return_value=mock_val_result):
        data = {'image': (non_medical_image_bytes, 'vacation_photo.jpg')}
        response = client.post('/api/analyze', data=data, content_type='multipart/form-data')
        assert response.status_code == 422
        data = response.get_json()
        assert data.get('error') == 'not_medical_image'
        assert data.get('status') == 'rejected'
        assert 'heuristic_score' in data
