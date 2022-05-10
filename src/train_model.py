import datetime
import logging
import os
import time
from types import SimpleNamespace

import hydra
import mlflow
import torch
import torchvision

import cookie_mask_rcnn as cmr
from cookie_mask_rcnn.modeling import presets, utils
from cookie_mask_rcnn.modeling.coco_utils import get_coco, get_coco_kp
from cookie_mask_rcnn.modeling.engine import evaluate, train_one_epoch
from cookie_mask_rcnn.modeling.group_by_aspect_ratio import (
    GroupedBatchSampler,
    create_aspect_ratio_groups,
)

# torchvision training script args
CONF = SimpleNamespace(
    **dict(
        aspect_ratio_group_factor=3,
        batch_size=2,
        data_augmentation="hflip",
        dataset="coco",
        device="cuda",
        dist_url="env://",
        epochs=26,
        lr=0.02,
        lr_gamma=0.1,
        lr_scheduler="multisteplr",
        lr_step_size=8,
        lr_steps=[16, 22],
        model="maskrcnn_resnet50_fpn",
        momentum=0.9,
        output_dir=".",
        pretrained=False,
        print_freq=20,
        resume="",
        rpn_score_thresh=None,
        start_epoch=0,
        sync_bn=False,
        test_only=False,
        trainable_backbone_layers=None,
        weight_decay=0.0001,
        workers=4,
        world_size=1,
    )
)


def get_dataset(name, image_set, transform, data_path):
    paths = {"coco": (data_path, get_coco, 91), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, data_augmentation):
    return (
        presets.DetectionPresetTrain(data_augmentation)
        if train
        else presets.DetectionPresetEval()
    )


@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    """This main function does the following:
    - load logging config
    - initialise experiment tracking (MLflow)
    - loads training, validation and test data
    - initialises model layers and compile
    - trains, evaluates, and then exports the model
    """

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    cmr.general_utils.setup_logging(logger_config_path)

    mlflow_init_status, mlflow_run = cmr.general_utils.mlflow_init(
        args,
        setup_mlflow=args["train"]["setup_mlflow"],
        autolog=args["train"]["mlflow_autolog"],
    )
    cmr.general_utils.mlflow_log(mlflow_init_status, "log_params", params=args["train"])

    if "POLYAXON_RUN_UUID" in os.environ:
        cmr.general_utils.mlflow_log(
            mlflow_init_status,
            "log_param",
            key="polyaxon_run_uuid",
            value=os.environ["POLYAXON_RUN_UUID"],
        )

    utils.init_distributed_mode(CONF)
    device = torch.device(CONF.device)

    # datasets = cmr.modeling.data_loaders.load_datasets(
    #     hydra.utils.get_original_cwd(), args
    # )
    logger.info("Loading data")
    dataset, num_classes = get_dataset(
        CONF.dataset,
        "train",
        get_transform(True, CONF.data_augmentation),
        args["train"]["data_path"],
    )
    dataset_test, _ = get_dataset(
        CONF.dataset,
        "val",
        get_transform(False, CONF.data_augmentation),
        args["train"]["data_path"],
    )

    logger.info("Creating data loaders")
    if CONF.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if CONF.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(
            dataset, k=CONF.aspect_ratio_group_factor
        )
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, CONF.batch_size
        )
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, CONF.batch_size, drop_last=True
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=CONF.workers,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=CONF.workers,
        collate_fn=utils.collate_fn,
    )

    logger.info("Creating model")
    kwargs = {"trainable_backbone_layers": CONF.trainable_backbone_layers}
    if "rcnn" in CONF.model:
        if CONF.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = CONF.rpn_score_thresh
    model = torchvision.models.detection.__dict__[CONF.model](
        num_classes=num_classes, pretrained=CONF.pretrained, **kwargs
    )
    model.to(device)
    if CONF.distributed and CONF.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if CONF.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[CONF.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=CONF.lr, momentum=CONF.momentum, weight_decay=CONF.weight_decay
    )

    CONF.lr_scheduler = CONF.lr_scheduler.lower()
    if CONF.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=CONF.lr_steps, gamma=CONF.lr_gamma
        )
    elif CONF.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONF.epochs
        )
    else:
        raise RuntimeError(
            "Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
            "are supported.".format(CONF.lr_scheduler)
        )

    if CONF.resume:
        checkpoint = torch.load(CONF.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        CONF.start_epoch = checkpoint["epoch"] + 1

    if CONF.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(CONF.start_epoch, CONF.epochs):
        if CONF.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, CONF.print_freq)
        lr_scheduler.step()
        # Save model checkpoints
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": args,
            "epoch": epoch,
        }
        utils.save_on_master(
            checkpoint,
            os.path.join(
                hydra.utils.get_original_cwd(), "models", "model_{}.pth".format(epoch)
            ),
        )
        utils.save_on_master(
            checkpoint,
            os.path.join(hydra.utils.get_original_cwd(), "models", "checkpoint.pth"),
        )

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # model = cmr.modeling.models.seq_model(args)

    # logger.info("Training the model...")
    # model.fit(
    #     datasets["train"],
    #     epochs=args["train"]["epochs"],
    #     validation_data=datasets["val"],
    # )

    # logger.info("Evaluating the model...")
    # test_loss, test_acc = model.evaluate(datasets["test"])

    # logger.info("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

    # logger.info("Exporting the model...")
    # cmr.modeling.utils.export_model(model)

    if mlflow_init_status:
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Artifact URI: {}".format(artifact_uri))
        cmr.general_utils.mlflow_log(
            mlflow_init_status, "log_params", params={"artifact_uri": artifact_uri}
        )
        logger.info(
            f"Model training with MLflow run ID {mlflow_run.info.run_id} has completed."
        )
        mlflow.end_run()
    else:
        logger.info("Model training has completed.")


if __name__ == "__main__":
    main()
