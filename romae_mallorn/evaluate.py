import os

import torch
from romae.model import RoMAEForClassification, RoMAEForClassificationConfig
from romae.utils import load_from_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from romae_mallorn.dataset import MallornDatasetwLabel
from romae_mallorn.config import MallornConfig
from sklearn.metrics import classification_report


def evaluate(args):
    #print("Implement evaluate")
    config = MallornConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on test set")
    
    # if args.pretrained_model is not None:
    #     print(f"Using {args.pretrained_model}")
    #     config.eval_checkpoint = args.pretrained_model

    
    model = load_from_checkpoint(args.eval_checkpoint, RoMAEForClassification,
                                 RoMAEForClassificationConfig).to(device).eval()
    all_preds = []
    all_labels = []
    with ( # Force mask_ratio to 0 ?
        MallornDatasetwLabel(args.test_parquet, mask_ratio=0) as test_dataset,
    ):
        dataloader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            num_workers=os.cpu_count()-1,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {key: val.to(device) for key, val in batch.items()}
                logit, _ = model(**batch)
                preds = torch.argmax(torch.nn.functional.softmax(logit, dim=1), dim=1)
                all_labels.extend(list(batch["label"].cpu().numpy()))
                all_preds.extend(list(preds.cpu().numpy()))
    print(classification_report(
        all_labels,
        all_preds,
        ## Edit the following as I hard-coded it and it's ugly
        labels=list(range(2)),
        target_names= ['non-TDE', 'TDE'], #config.class_names,
        digits=4
    ))