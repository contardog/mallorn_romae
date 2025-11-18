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

    
    pretrain.set_defaults(func=run_pretrain)
    
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
