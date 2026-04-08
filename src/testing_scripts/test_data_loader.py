import torch
import numpy as np
import matplotlib.pyplot as plt
from src.param_config import SDSSDataConfig
from src.sdss_dataloader import SDSSDataModule


def check_pipeline():
    # 1. Initialize Config and Module
    config = SDSSDataConfig(batch_size=32, num_workers=0)  # Set workers=0 for easier debugging
    data_module = SDSSDataModule(config)

    print("--- 1. Data Preparation ---")
    data_module.prepare_data()

    # 2. Get Loaders
    train_loader = data_module.get_loader(data_module.train_ds, use_sampler=True)
    val_loader = data_module.get_loader(data_module.val_ds)

    print(f"Total Samples: {len(data_module.train_ds) + len(data_module.val_ds) + len(data_module.test_ds)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Number of classes: {data_module.num_classes}")

    # 3. Inspect a Batch
    print("\n--- 2. Batch Inspection ---")
    batch = next(iter(train_loader))

    flux = batch['flux']
    labels = batch['label']
    scalars = batch['scalars']

    print(f"Flux shape (Batch, Channel, Length): {flux.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Scalars shape: {scalars.shape}")

    # 4. Verify Sampler (Balance Check)
    unique, counts = np.unique(labels.numpy(), return_counts=True)
    print("\n--- 3. Class Balance in Training Batch ---")
    for u, c in zip(unique, counts):
        class_name = data_module.classes[u]
        print(f"   {class_name:<25}: {c} samples")

    # 5. Verify Augmentation (Visual/Numerical Check)
    print("\n--- 4. Augmentation Verification ---")
    # We pull the SAME index twice to see if augmentation makes them different
    idx = 0
    # Accessing internal dataset directly to bypass sampler for a moment
    sample_1 = data_module.train_ds[idx]['flux']
    sample_2 = data_module.train_ds[idx]['flux']

    diff = torch.abs(sample_1 - sample_2).sum().item()
    if diff > 0:
        print(f"✅ Augmentation Active: Cumulative pixel difference is {diff:.4f}")
    else:
        print("❌ Warning: Samples are identical. Check if use_augmentation=True")

    # 6. Quick Plot (Optional Visualization)
    plt.figure(figsize=(12, 4))
    plt.plot(sample_1[0].numpy(), label='Augmented Version A', alpha=0.8)
    plt.plot(sample_2[0].numpy(), label='Augmented Version B', alpha=0.6)
    plt.title(f"Augmentation Check: {data_module.classes[data_module.train_ds[idx]['label']]}")
    plt.legend()
    plt.xlabel("Pixel (Log-Lambda)")
    plt.ylabel("Flux")
    plt.show()


    # # plot 5 others to see if they are different
    # for i in range(1, 6):
    #     # use random indices to check different samples
    #     idx = np.random.randint(0, len(data_module.train_ds))
    #     s1 = data_module.train_ds[idx]['flux']
    #     s2 = data_module.train_ds[idx]['flux']
    #
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(s1[0].numpy(), label='Augmented Version A', alpha=0.8)
    #     plt.plot(s2[0].numpy(), label='Augmented Version B', alpha=0.6)
    #     plt.title(f"Augmentation Check: {data_module.classes[data_module.train_ds[idx]['label']]}")
    #     plt.legend()
    #     plt.xlabel("Pixel (Log-Lambda)")
    #     plt.ylabel("Flux")
    #     plt.show()


if __name__ == "__main__":
    try:
        check_pipeline()
        print("\n✨ DATALOADER CHECK PASSED!")
    except Exception as e:
        print(f"\n💥 CHECK FAILED: {str(e)}")