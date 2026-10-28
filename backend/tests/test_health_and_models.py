"""
Tests for health check and model listing endpoints.
"""

def test_health_check_endpoint(client):
    """GET /health should return 200 with healthy status message."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data.get('status') == 'healthy'
    assert 'message' in data


def test_models_list_endpoint(client):
    """GET /models should return list of supported models with availability metadata."""
    response = client.get('/models')
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert 'models' in data
    assert isinstance(data['models'], list)

    model_names = [m.get('name') for m in data['models']]
    for expected in ['chexnet', 'mura', 'tuberculosis', 'rsna', 'unet']:
        assert expected in model_names

    for model_item in data['models']:
        assert 'name' in model_item
        assert 'available' in model_item
        assert isinstance(model_item['available'], bool)


def test_nonexistent_endpoint_returns_404(client):
    """Non-existent endpoints should return standard 404 status."""
    response = client.get('/api/does_not_exist')
    assert response.status_code == 404
