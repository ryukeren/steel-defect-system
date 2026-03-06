from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_analytics_endpoint():
    response = client.get("/analytics")
    assert response.status_code == 200


def test_predict_endpoint():
    files = {"file": ("test.jpg", b"fake_image_data", "image/jpeg")}
    response = client.post("/predict", files=files)
    
    # API should respond (even if image is fake)
    assert response.status_code in [200, 400]