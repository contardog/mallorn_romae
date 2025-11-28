from romae.model import RoMAEForClassification, RoMAEForClassificationConfig
from romae.trainer import Trainer, TrainerConfig
import torch

from romae_mallorn.dataset import MallornDatasetwLabel
from romae_mallorn.config import MallornConfig


def finetune(args):
    #print("The finetuning part has not been implemented yet")
    # # Let's use the tiny model:

    ## This is using a regular loss
    config = MallornConfig()

    if args.pretrained_model is not None:
        print(f"Using {args.pretrained_model}")
        config.pretrained_model = args.pretrained_model

    
    ## ? Not sure what happesn otherwise


    if args.lr is not None:
        print("Overridding configured learning rate")
        config.finetune_lr = args.lr
    if args.batch_size is not None:
        print("Overriding configured batch size")
        config.finetune_batch_size = args.batch_size
    if args.epochs is not None:
        print("Overridding configured number of epochs")
        config.finetune_epochs = args.epochs
    if args.finetune_mask_ratio is not None:
        print("Overriding mask ratio")
        config.finetune_mask_ratio = args.finetune_mask_ratio



        
    model = RoMAEForClassification.from_pretrained(
        config.pretrained_model,
        dim_output=config.n_classes
    )
    model.set_loss_fn(
        torch.nn.CrossEntropyLoss(
            weight=torch.tensor(config.class_weights if config.finetune_use_class_weights else None),
            label_smoothing=config.finetune_label_smoothing
        )
    )


    
    trainer_config = TrainerConfig(
        warmup_steps=config.pretrain_warmup_steps,
        checkpoint_dir=args.model_name+"_checkpoint-finetune_", # "checkpoints-finetune-",
        epochs=config.finetune_epochs,
        base_lr=config.finetune_lr,
        eval_every=config.finetune_eval_every,
        save_every=config.finetune_save_every,
        optimizer_args=config.finetune_optimargs,
        batch_size=config.finetune_batch_size,
        project_name= config.project_name + args.model_name,
        entity_name='contardog-university-of-nova-gorica',
        gradient_clip=config.finetune_grad_clip,
        lr_scaling=True
    )
    
    if (args.vega):
        print("Original trainer config num_dataset_workerS")
        print(trainer_config.num_dataset_workers)
        print("Overriding")
        trainer_config.num_dataset_workers=len(os.sched_getaffinity(0)) #*2
        print(trainer_config.num_dataset_workers)

    
    trainer = Trainer(trainer_config)
    with ( ## Mask ratio ?!
        MallornDatasetwLabel(args.test_parquet, mask_ratio=config.finetune_mask_ratio ) as test_dataset,
        MallornDatasetwLabel(args.train_parquet,            mask_ratio=config.finetune_mask_ratio,     
                         gaussian_noise=config.gaussian_noise) as train_dataset
    ):
        trainer.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
        )
