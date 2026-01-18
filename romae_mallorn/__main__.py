import argparse as ap

def run_preprocess(*_, **__):
    from romae_mallorn import preprocess
    preprocess.preprocess()

def run_plot(*_, **__):
    from romae_mallorn import plot
    plot.plot()

def run_evaluate(*_, **__):
    from romae_mallorn import evaluate
    evaluate.evaluate(args)


def run_pretrain(args):
    from romae_mallorn import pretrain
    pretrain.pretrain(args)

def run_pretrain_contrastive(args):
    from romae_mallorn import pretrain_contrastive
    pretrain_contrastive.pretrain_contrastive(args)


def run_finetune(*_, **__):
    from romae_mallorn import finetune
    finetune.finetune(args)


if __name__ == '__main__':
    """Very simple command line interface that takes in some command and runs 
    the corresponding function.
    """
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.set_defaults(func=run_preprocess)

     # Pretrain subparser with additional arguments
    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("--lr", type=float, default=None, 
                         help="Learning rate for training")
    
    pretrain.add_argument("--batch_size", type=int, default=None,
                         help="Batch size for training")

    
    pretrain.add_argument("--model_size", type=str, default=None,
                         help="Model size (see RoMAE and this code utils files)")
    
    pretrain.add_argument("--decoder_size", type=int, default=None,
                         help="Decoder embedding size (int; default use the same as encoder passed in model_size)")
    
    pretrain.add_argument("--epochs", type=int, default=None,
                         help="Number of training epochs")
    
    pretrain.add_argument("--model_name", type=str, required=True,
                         help="Name for the saved model (Required to avoid overwriting my own experiment")
    
    pretrain.add_argument("--train_parquet", type=str, required=True,
                         help="Path to training parquet")
    pretrain.add_argument("--test_parquet", type=str, required=True,
                         help="Path to test parquet")

    
    pretrain.add_argument("--pretrain_mask_ratio", type=float, default=None, 
                         help="Mask ratio for pretraining")
    
    pretrain.add_argument('--vega',  action="store_true",
                    help="Use if run on clusters to avoid wrong computation of worker numbers!!")
                          #action=argparse.BooleanOptionalAction)
    pretrain.add_argument('--no_cls',  action="store_true",
                    help="Use if you want to NOT have the CLS token")
    
    pretrain.set_defaults(func=run_pretrain)

    ## Pretrain contrastive subparser

    pretrain_contrastive = subparsers.add_parser("pretrain_contrastive")
    pretrain_contrastive.add_argument("--lr", type=float, default=None, 
                         help="Learning rate for training")
    
    pretrain_contrastive.add_argument("--batch_size", type=int, default=None,
                         help="Batch size for training")

    
    pretrain_contrastive.add_argument("--model_size", type=str, default=None,
                         help="Model size (see RoMAE and this code utils files)")
    
    pretrain_contrastive.add_argument("--decoder_size", type=int, default=None,
                         help="Decoder embedding size (int; default use the same as encoder passed in model_size)")
    
    pretrain_contrastive.add_argument("--epochs", type=int, default=None,
                         help="Number of training epochs")
    
    pretrain_contrastive.add_argument("--model_name", type=str, required=True,
                         help="Name for the saved model (Required to avoid overwriting my own experiment")
    
    pretrain_contrastive.add_argument("--train_parquet", type=str, required=True,
                         help="Path to training parquet")
    pretrain_contrastive.add_argument("--test_parquet", type=str, required=True,
                         help="Path to test parquet")

    
    pretrain_contrastive.add_argument("--pretrain_mask_ratio", type=float, default=None, 
                         help="Mask ratio for pretrain-contrastive (MAE)")
    
    pretrain_contrastive.add_argument('--vega',  action="store_true",
                    help="Use if run on clusters to avoid wrong computation of worker numbers!!")
                          #action=argparse.BooleanOptionalAction)
    # pretrain_contrastive.add_argument('--no_cls',  action="store_true",
    #                 help="Use if you want to NOT have the CLS token")

    ## Contrastive  arguments
    pretrain_contrastive.add_argument("--temperature", type=float,  default=None,  #default=0.15, 
                         help="Temperature for contrastive loss")
    
    pretrain_contrastive.add_argument("--contrastive_mask_ratio", type=float,  default=None, #default=0.5, 
                         help="Mask ratio for contrastive augmentation (MAE)")

    
    pretrain_contrastive.add_argument("--class_contrast_weight", type=float, default=None,  #default=1.0, 
                         help="Weight for class contrastive loss")
    
    pretrain_contrastive.add_argument("--augm_contrast_weight", type=float,  default=None, #default=1.0, 
                         help="Weight for augmentation contrastive loss")
    
    pretrain_contrastive.add_argument("--recon_weight", type=float,  default=None, #default=1.0, 
                         help="Weight for reconstruction loss")
    
    pretrain_contrastive.add_argument("--projection_dim", type=int,  default=None, #default=1.0, 
                         help="Projection dim for ContrastiveHead")
    pretrain_contrastive.add_argument("--cls_contrastive_dim", type=int,  default=None, #default=1.0, 
                         help="Number of feature to use in CLS")
    
    pretrain_contrastive.add_argument("--no_decode", action="store_true", # default=None, #default=1.0, 
                         help="If there's no decoding")
    
    #temperature: float = 0.15
    #projection_dim: int = 32
    #cls_contrastive_dim: Optional[int] = 32  # Split CLS token if set
    #aug_contrast_weight: float = 1.0
    #class_contrast_weight: float = 1.0
    #recon_weight: float = 0.0
    #mask_ratio_contrastive = 0.5
    
    
    pretrain_contrastive.set_defaults(func=run_pretrain_contrastive)

    
    
    # Finetune subparser
    finetune = subparsers.add_parser("finetune")

    finetune.add_argument("--lr", type=float, default=None, 
                         help="Learning rate for training")
    
    finetune.add_argument("--batch_size", type=int, default=None,
                         help="Batch size for training")
    
    finetune.add_argument("--epochs", type=int, default=None,
                         help="Number of training epochs")
    
    finetune.add_argument("--pretrained_model", type=str, required=True,
                         help="Path to test parquet")
    
    finetune.add_argument("--model_name", type=str, required=True,
                         help="Name for the saved model (Required to avoid overwriting my own experiment")
    
    finetune.add_argument("--train_parquet", type=str, required=True,
                         help="Path to training parquet")
    finetune.add_argument("--test_parquet", type=str, required=True,
                         help="Path to test parquet")

    
    finetune.add_argument("--finetune_mask_ratio", type=float, default=None, 
                         help="Mask ratio for finetuning")
    
    finetune.set_defaults(func=run_finetune)
    
    # Evaluate subparser
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--test_parquet", type=str, required=True,
                         help="Path to parquet to evaluate")

    evaluate.add_argument("--eval_checkpoint", type=str, required=True,
                         help="Path to checkpoint to evaluate")
    
    evaluate.set_defaults(func=run_evaluate)
    
    # Plot subparser
    plot = subparsers.add_parser("plot")
    plot.set_defaults(func=run_plot)
    

    args = parser.parse_args()
    args.func(args)
