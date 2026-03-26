if __name__ == '__main__':
    import json
    import torch
    import torch.nn as nn
    import torchmetrics
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from core_twice.data_module import DataModule
    from core_twice.twice_da import twice_da_tiny
    from core_twice.model_compilation import ModelCompilation
    from core_twice.callbacks import LossMetricTracker

    with open('core_twice/configs/twice_config.json') as f:
        cfg = json.load(f)['caltech']

    dataset = cfg['dataset']
    dataset_path = cfg['dataset_path']
    image_size = cfg['image_size']
    batch_size = cfg['batch_size']
    num_classes = cfg['num_classes']
    device = cfg['device']
    task = cfg['task']
    label_smoothing = cfg['label_smoothing']
    learning_rate = cfg['learning_rate']
    gradient_clipping = cfg['gradient_clipping']
    accumulate_grad_batches = cfg['accumulate_grad_batches']
    epochs = cfg['epochs']

    data_module = DataModule(dataset=dataset,
                             dataset_path=dataset_path,
                             image_size=image_size,
                             batch_size=batch_size,
                             num_classes=num_classes)

    network = twice_da_tiny(num_classes)
    metrics = torchmetrics.MetricCollection({'accuracy': torchmetrics.Accuracy(task=task, num_classes=num_classes).to(device)})
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW

    model = ModelCompilation(model=network,
                             metrics=metrics,
                             loss_function=loss_function,
                             optimizer=optimizer,
                             learning_rate=learning_rate,
                             accelerator=device,
                             data_module=data_module)

    checkpoint_callback = ModelCheckpoint(filename='model-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}', monitor="val_loss")
    loss_metric_tracker_callback = LossMetricTracker()

    trainer = pl.Trainer(callbacks=[checkpoint_callback, loss_metric_tracker_callback],
                         precision='32',
                         accelerator=device,
                         devices="auto",
                         max_epochs=epochs,
                         gradient_clip_val=gradient_clipping,
                         accumulate_grad_batches=accumulate_grad_batches
                         )

    #flops, macs, params = calculate_flops(model=network, input_shape=(1, 3, 224, 224), output_as_string=True, output_precision=4, print_detailed=False)

    trainer.fit(model, datamodule=data_module)
    torch.save(loss_metric_tracker_callback.collection, 'train_results.pth')
