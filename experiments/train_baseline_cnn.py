import torch
import numpy as np
from src.param_config import SDSSDataConfig
from src.sdss_dataloader import SDSSDataModule
from src.models.classical_cnn import SpectraClassifier
from src.training.trainer import SDSSPerformanceTrainer, FocalLoss
from src.training.metrics import SDSSMetricTracker
from src.param_config import TrainingConfig
import os

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
    # aux_features=0 maybe add later if needed
    model = SpectraClassifier(
        num_classes=data_module.num_classes,
        aux_features=0,
        dropout=training_config.dropout
    )

    # 3. Initialize Trainer & Metrics
    # Performance focus: Handles CUDA/MPS device selection automatically
    trainer = SDSSPerformanceTrainer(model, data_config, run_name='Baseline_CNN')
    plot_folder = trainer.plots_dir
    tracker = SDSSMetricTracker(results_dir=plot_folder)

    # 4. Run Training
    # OneCycleLR logic is embedded in the trainer for rapid convergence
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.epochs,
        lr=training_config.lr,
        weight_decay=training_config.weight_decay
    )

    # 5. Final Evaluation
    print("\n--- Evaluating on Test Set ---")
    best_model_path = os.path.join(trainer.checkpoint_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=trainer.device))
    model.eval()

    all_preds = []
    all_labels = []
    test_loss = 0.0
    criterion = FocalLoss(gamma=2.0)

    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(trainer.device)
            aux = batch['scalars'].to(trainer.device)
            labels = batch['label'].to(trainer.device)

            logits = model(flux, aux)
            loss = criterion(logits, labels)
            test_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Generate Insights
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    test_acc = (y_true == y_pred).mean()
    test_loss = test_loss / len(y_true)

    trainer.logger.info("========== FINAL TEST RESULTS ==========")
    trainer.logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    trainer.logger.info("========================================")

    tracker.plot_history(trainer.history)
    tracker.plot_confusion_matrix(y_true, y_pred, data_module.classes)
    tracker.plot_per_class_accuracy(y_true, y_pred, data_module.classes)

    tracker.analyze_confusion_matrix(y_true, y_pred, data_module.classes, split_name='baseline_cnn')

    print("\nBaseline Complete! Results saved to ./results_baseline")

if __name__ == "__main__":
    main()