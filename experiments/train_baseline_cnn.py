import torch
import numpy as np
from src.param_config import SDSSDataConfig
from src.sdss_dataloader import SDSSDataModule
from src.models.classical_cnn import SpectraClassifier
from src.training.trainer import SDSSPerformanceTrainer
from src.training.metrics import SDSSMetricTracker
from src.param_config import TrainingConfig

def main():
    # 1. Setup Configuration & Data
    # Path is pulled from SDSSDataConfig: dataset/ML_SDSS_CLEANED_DATA.parquet
    data_config = SDSSDataConfig()
    training_config = TrainingConfig()
    data_module = SDSSDataModule(data_config)
    data_module.prepare_data()

    # Get loaders using the sampler for class balance
    train_loader = data_module.get_loader(data_module.train_ds, use_sampler=True)
    val_loader = data_module.get_loader(data_module.val_ds)
    test_loader = data_module.get_loader(data_module.test_ds)

    print(f"Training on {data_module.num_classes} classes: {data_module.classes}")

    # 2. Initialize Architecture
    # aux_features=6 corresponds to Z, U, G, R, I, Z in param_config.py
    model = SpectraClassifier(
        num_classes=data_module.num_classes,
        aux_features=len(data_config.scalar_cols),
        dropout=training_config.dropout
    )

    # 3. Initialize Trainer & Metrics
    # Performance focus: Handles CUDA/MPS device selection automatically
    trainer = SDSSPerformanceTrainer(model, data_config)
    tracker = SDSSMetricTracker(results_dir=training_config.results_dir_baseline)

    # 4. Run Training
    # OneCycleLR logic is embedded in the trainer for rapid convergence
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.epochs,
        lr=training_config.lr,
        weight_decay=training_config.weight_decay
    )

    # 5. Final Evaluation (Find the "Hard Classes")
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load("best_baseline_model.pt"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(trainer.device)
            aux = batch['scalars'].to(trainer.device)
            labels = batch['label'].to(trainer.device)

            logits = model(flux, aux)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Generate Insights
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    tracker.plot_history(trainer.history)
    tracker.plot_confusion_matrix(y_true, y_pred, data_module.classes)
    tracker.plot_per_class_accuracy(y_true, y_pred, data_module.classes)

    print("\nBaseline Complete! Results saved to ./results_baseline")

if __name__ == "__main__":
    main()