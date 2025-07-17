import math
import torch
from tqdm import tqdm
from collections import defaultdict
from src.utils.utils  import calculate_mean
from torch.utils.data import DataLoader


def evaluate_inidiv_dataset_LM(datasets, data_collator, batch_size, accelerator, model ,task="LM"):
    """
    Evaluate individual lanaguages
    """
    bpc_dictionary = {}
    loss_dictionary = {}
    stats_agg = defaultdict(list)
    model.eval()
    for i in datasets:
        dataset = datasets[i]
        dataloader = DataLoader(dataset,
                                collate_fn=data_collator,
                                batch_size=batch_size,
                                shuffle=False)
        dataloader = accelerator.prepare(dataloader)
        count = 0
        losses = []
        for step, batch in enumerate(tqdm(dataloader, desc=f'evaluating {i} language...')):
            with torch.no_grad():
                seq_loss, stats, aux_loss, _ = model(batch, task=task)
                count += 1

            losses.append(accelerator.gather_for_metrics(seq_loss.repeat(batch_size)))
            for k, v in stats.items():
                stats_agg[f"{i}_{k}"].append(v)

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            eval_bpc = eval_loss / math.log(2)
        except OverflowError:
                eval_bpc = float("inf")
        
        bpc_dictionary[f"{i}_eval_bpc"] = eval_bpc.item()
        bpc_dictionary[f"{i}_eval_loss"] = eval_loss.item()

            

        print(f"Finished evaluating {i} language")
    stats_mean_dict = calculate_mean(stats_agg)
    bpc_dictionary.update(stats_mean_dict)
    print(bpc_dictionary)
    return bpc_dictionary, loss_dictionary

    
