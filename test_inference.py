import joblib
import os

MODEL_PATH = os.path.join("models", "sentiment_model.pkl")

def test_inference():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

    print("Cargando modelo...")
    model = joblib.load(MODEL_PATH)

    sample_texts = [
        "This movie was absolutely amazing!",
        "I really hated the ending.",
        "It was fine, nothing special."
    ]

    predictions = model.predict(sample_texts)
    print("\n=== Resultados de prueba ===")
    for text, pred in zip(sample_texts, predictions):
        print(f"'{text}' → {pred}")

if __name__ == "__main__":
    test_inference()
