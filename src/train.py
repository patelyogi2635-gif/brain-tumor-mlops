import mlflow
import mlflow.tensorflow
from src.data_pipeline import load_data
from src.model import build_model

train_dir = "data/Training"
test_dir = "data/Testing"

mlflow.set_experiment("Brain_Tumor_Classification")

with mlflow.start_run(run_name="Brain_Tumor_Classification"):
    train_ds, test_ds = load_data(train_dir, test_dir)
    model = build_model()
    history = model.fit(train_ds, epochs=10, validation_data=test_ds)

    loss, accuracy = model.evaluate(test_ds)

    mlflow.log_metric("loss", loss)
    mlflow.tensorflow.log_model(model, "model")

    mlflow.log_param("image_size", 300)
    mlflow.log_param("augmentation", "flip+rotation+zoom+contrast")
    mlflow.log_param("base_model", "EfficientNetB3")

    model.save("models/brain_tumor_model.h5")
    print("Model saved successfully!")
