from romae.utils import get_encoder_size
from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig
from romae.trainer import Trainer, TrainerConfig

from romae_mallorn.dataset import MallornDataset
from romae_mallorn.config import MallornConfig
from romae_mallorn.utils import override_encoder_size

def pretrain(args):
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """
    config = MallornConfig()

    ## Make it smaller?! Add some piece of code for this
    if args.model_size is not None:
        print("Overriding configured model size!!!!")
        config.model_size = args.model_size
    
    encoder_args = override_encoder_size(config.model_size)

    ## Forcing the same size of embeddings in decoder but a shallow depth
    decoder_args = {
                    "d_model": encoder_args['d_model'],
                    "nhead": 3,
                    "depth": 2 ## I guess this is arbitrary? We could have it at =1? 
                }
    
    model_config = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config= EncoderConfig(**decoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=2,
        n_pos_dims=2
    )

    print(model_config)
    
    if args.lr is not None:
        print("Overridding configured learning rate")
        config.pretrain_lr = args.lr
    if args.batch_size is not None:
        print("Overriding configured batch size")
        config.pretrain_batch_size = args.batch_size
    if args.epochs is not None:
        print("Overridding configured number of epochs")
        config.pretrain_epochs = args.epochs



    model = RoMAEForPreTraining(model_config)
    trainer_config = TrainerConfig(
        warmup_steps=config.pretrain_warmup_steps,
        checkpoint_dir=args.model_name+"_checkpoint_",
        epochs=config.pretrain_epochs,
        base_lr=config.pretrain_lr,
        eval_every=config.pretrain_eval_every,
        save_every=config.pretrain_save_every,
        optimizer_args=config.pretrain_optimargs,
        batch_size= config.pretrain_batch_size,
        project_name= config.project_name + args.model_name,
        entity_name='contardog-university-of-nova-gorica',
        gradient_clip=config.pretrain_grad_clip,
        lr_scaling=True,
        #max_checkpoints = 20,
    )
    print("Start pretrain")
    
    trainer = Trainer(trainer_config)
    with (
        MallornDataset(args.test_parquet) as test_dataset,
        MallornDataset(args.train_parquet,                
                         gaussian_noise=config.gaussian_noise) as train_dataset
    ):
        trainer.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
        )
