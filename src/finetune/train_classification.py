import argparse
import inspect
import json
import logging
import math
import os
import random
import evaluate
import transformers
import torch
import torch.optim as optim
import yaml
import shutil
import numpy as np
from tqdm import tqdm

from accelerate import (
    load_checkpoint_and_dispatch,
    init_empty_weights,
    Accelerator,
    DistributedDataParallelKwargs,
)

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datetime import datetime
from collections import defaultdict
from transformers import (
    default_data_collator,
    get_scheduler,
    DataCollatorForTokenClassification,
)
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
    DistributedLengthGroupedSampler,
)
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from src.model.fxt import FxTTransformerLM, FxTAverageSingleInputWithPadding
from src.utils.utils import (
    read_json_file,
    init_seed,
    save_args_to_json,
    calculate_mean,
    grad_norm,
)
from src.utils.data_utils import MixtureByteVocab, JointInputcorpus, BYTE_MODEL


logger = get_logger(__name__)

TASK_MAPPING = {
    "conll2003": "token_cls",
    "paws-x": "seq_cls",
    "wikiann": "token_cls",
    "sentiment": "seq_cls",
    "xnli": "seq_cls",
    "sib200": "seq_cls",
    "medical_abstracts": "seq_cls",
    "ili": "seq_cls",
    "code": "seq_cls",
    "legal": "seq_cls",
    "irony": "seq_cls",
    "codemix": "seq_cls",
}
# create NER label mapping
NER_LABELS_TO_IDS = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}
NER_IDS_TO_LABELS = {v: k for k, v in NER_LABELS_TO_IDS.items()}


def transform_ids_to_labels(labels):
    """
    Takes in a numpy array of ids seq_len x batch_size
    and returns a list of list of labels
    """
    return [[NER_IDS_TO_LABELS[l] for l in label if l != -100] for label in labels]


def get_task_type(task_name):
    return TASK_MAPPING.get(task_name, "unknown")


def dynamic_padding_data_collator(features, tokenizer):
    """
    Dynamically pads sequences in the batch to the maximum length of the batch.
    """
    # Extract input_ids and attention_mask
    input_ids = [f["input_ids"] for f in features]
    attention_masks = [f["attention_mask"] for f in features]

    # Dynamically pad to the maximum sequence length in the batch
    batch = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        return_tensors="pt",
    )
    if "label" in features[0]:
        labels = [f["label"] for f in features]
        batch["labels"] = torch.tensor(labels)

    return batch


def load_pretrained_model(model_path, config):

    model_ckpt = os.path.join(model_path, "model.pth")
    if not os.path.exists(model_ckpt):
        model_ckpt = os.path.join(model_path, "model.safetensors")

    # Load Pretrained model
    def get_model_config():
        model_args = inspect.getfullargspec(FxTTransformerLM).args
        assert model_args.index("self") == 0
        model_args = model_args[1:]

        return {arg: config.get(arg) for arg in model_args}

    pretrained_model = FxTTransformerLM(**get_model_config())
    # model checkpoints is a .pth path file use this line else load model checkpoint as a .safetensors file
    if model_ckpt.endswith(".pth"):
        pretrained_model.load_state_dict(torch.load(model_ckpt)["model"])
    else:
        pretrained_model.load_state_dict(load_file(model_ckpt))

    return pretrained_model, config


def evaluate_model(
    model,
    dataloader,
    accelerator,
    metric,
    args,
    phase="valid",
    return_predictions=False,
):
    """
    Evaluate the model on the given dataloader

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        accelerator: Accelerator instance
        metric: Evaluation metric to compute
        args: Script arguments
        phase: Evaluation phase (valid/test)
        return_predictions: Whether to return predictions and targets

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    losses = []
    stats_agg = defaultdict(list)
    samples_seen = 0
    all_predictions = []
    all_targets = []
    compression_rate = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            if args.joint_input:
                loss, logits, stats, _ = model(batch)
            else:
                loss, logits, stats = model(
                    batch["x_ids"], batch["y_ids"], batch["labels"]
                )

        losses.append(accelerator.gather_for_metrics(loss.repeat(args.batch_size)))
        # breakpoint()
        if f"{args.language}_compression_rate" in stats:
            compression_rate.append(stats[f"{args.language}_compression_rate"])

        predictions = logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        if accelerator.num_processes > 1:
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]

        # Save predictions
        if args.task_type == "token_cls":
            # transform labels to ids
            predictions = transform_ids_to_labels(predictions.cpu().tolist())
            references = transform_ids_to_labels(references.cpu().tolist())
        else:
            predictions = predictions.cpu().numpy()
            references = references.cpu().numpy()

        all_predictions.extend(predictions)
        all_targets.extend(references)
        metric.add_batch(predictions=predictions, references=references)

        for k, v in stats.items():
            stats_agg[f"{phase}_{k}"].append(v)
    # Compute boundary statsn
    stats_mean_dict = calculate_mean(stats_agg)

    losses = torch.cat(losses)
    avg_loss = torch.mean(losses)
    eval_metric = metric.compute()
    if args.task_type == "token_cls":
        print(f"Eval metric: {eval_metric}")
        eval_metric = eval_metric["overall_f1"]
        print(f"Printing Token level scores for {args.language}")
        # get the mean and variance of the compression rate
        mean_comp_rate = np.mean(compression_rate)
        mean_comp_rate_var = np.var(compression_rate)
        print(f"Mean compression rate: {mean_comp_rate}")
        print(f"Mean compression rate variance: {mean_comp_rate_var}")

    result = {
        f"{phase}_accuracy": eval_metric,
        f"{phase}_loss": avg_loss.item(),
    }
    result.update(stats_mean_dict)

    if return_predictions:
        return result, all_predictions, all_targets

    return result


def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent_parser])
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser])

    cfg_parser.add_argument("--config", default="default")
    cfg_parser.add_argument("--config_file", default=None)

    config_args, _ = cfg_parser.parse_known_args()

    assert config_args.config is not None and config_args.config_file is not None
    with open(config_args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]["train"]

    # Main args
    general = parser.add_argument_group("general setup")
    general.add_argument(
        "--work_dir", required=True, type=str, help="Directory for the results"
    )

    dataset = parser.add_argument_group("dataset setup")
    dataset.add_argument(
        "--dataset_name", type=str, help="Name of dataset on huggingface"
    )
    dataset.add_argument("--language", type=str, help="Language")
    dataset.add_argument(
        "--joint_input",
        type=bool,
        help="Whether to encode muliple inputs as a single sequence",
    )
    dataset.add_argument(
        "--cache_dir",
        type=str,
        default="~/owos/.cache/",
        help="Directory to cache the dataset and tokenizer",
    )

    model = parser.add_argument_group("model setup")
    model.add_argument("--n_labels", type=int, default=3, help="Number of labels")
    model.add_argument(
        "--pretrained_path",
        type=str,
        help="Path to the pretrained model",
        required=True,
    )
    model.add_argument("--model_type", type=str, help="If model is fixed or routed")

    opt = parser.add_argument_group("optimizer setup")
    opt.add_argument(
        "--optim", default="adam", type=str, choices=["adam"], help="Optimizer to use"
    )
    opt.add_argument("--lr", type=float, help="Initial learning rate")
    opt.add_argument(
        "--scheduler",
        default="cosine",
        type=str,
        choices=["cosine"],
        help="LR scheduler to use",
    )
    opt.add_argument("--clip", type=float, default=0.25, help="Gradient clipping")
    opt.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for adam"
    )
    opt.add_argument("--adam_b1", type=float, default=0.9)
    opt.add_argument("--adam_b2", type=float, default=0.999)
    opt.add_argument("--adam_eps", type=float, default=1e-8)

    training = parser.add_argument_group("training setup")
    training.add_argument(
        "--max_train_steps", type=int, default=None, help="Max number of training steps"
    )
    training.add_argument(
        "--batch_size", type=int, default=32, help="Global batch size"
    )
    training.add_argument("--seed", type=int, default=42, help="Random seed")
    training.add_argument(
        "--seq_len", type=int, default=512, help="Maximum sequence length"
    )
    training.add_argument("--report_to", type=str, default="wandb", help="Wandb")
    training.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )
    training.add_argument(
        "--num_warmup_steps", type=int, default=5000, help="num_warmup_steps"
    )
    training.add_argument("--warmup_ratio", type=int, default=0.1, help="warmup_ratio")
    training.add_argument(
        "--logging_steps", type=int, default=250, help="logging_steps"
    )
    training.add_argument(
        "--checkpointing_steps",
        type=str,
        help="whether to group texts ?",
        default="4000",
    )
    training.add_argument(
        "--with_tracking", type=bool, help="whether to track with wandb ?", default=True
    )
    training.add_argument(
        "--resume_from_checkpoint",
        help="resume_from_checkpoint",
        default=None,
    )
    training.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    training.add_argument(
        "--use_best_model",
        type=str,
        default="false",
        help="Whether to use the best model based on validation loss for test evaluation. If False, uses the final model.",
    )

    parser.set_defaults(**config)

    args, _ = parser.parse_known_args()
    args.use_best_model = True if args.use_best_model.lower() == "true" else False

    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    config = config_file = os.path.join(args.pretrained_path, "config.json")
    config = read_json_file(config_file)
    # Create output directory with timestamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # seet model_dir to be the last 3 paths of the args.pretrained_path
    model_dir = "_".join(args.pretrained_path.split("/")[-3:])
    work_dir = f"{args.work_dir}/{str(model_dir).split('/')[-1]}/{str(args.dataset_name).replace('/', '_')}/epochs_{str(args.num_train_epochs)}/{args.language}/bz{args.batch_size}_seed{args.seed}"

    basename = f"{os.path.basename(work_dir)}_{current_time}"
    new_path = os.path.join(os.path.dirname(work_dir), basename)
    args.output_dir = new_path
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 50)
    print("Start training to {}".format(args.output_dir))

    # Create directory for best model checkpoint
    best_model_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    # Accelerate config
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["project_dir"] = args.output_dir
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the sssconfiguration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    transformers.logging.set_verbosity_error()

    # Load pretrained model
    logger.info("Loading pretrained model ....")
    pretrained_model, pretrained_config = load_pretrained_model(
        args.pretrained_path, config
    )
    pretrained_config["cache_dir"] = args.cache_dir
    boundary_kwargs = {
        "boundaries_type": pretrained_config["boundaries_type"],
        "fixed_sf": pretrained_config["fixed_sf"],
        "tokenizer_path": (
            pretrained_config["tokenizer_path"]
            if pretrained_config["tokenizer_path"]
            else BYTE_MODEL
        ),
        "script_tokens": pretrained_config["script_tokens"],
        "cache_dir": pretrained_config["cache_dir"],
    }
    vocab = MixtureByteVocab(**boundary_kwargs)
    id_to_script = {
        value: key for key, value in pretrained_config["id_to_script"].items()
    }
    language_to_script_id = {
        lang: int(id_to_script[script])
        for lang, script in pretrained_config["language_to_script"].items()
    }
    logger.info(f"language_to_script_id is {language_to_script_id}")

    ###########################################################################
    # Load data
    ###########################################################################
    logger.info("Loading data corpus ....")
    data_corpus = JointInputcorpus(
        language=args.language,
        dataset_name=args.dataset_name,
        tokenizer=vocab.tokenizer,
        max_seq_length=args.seq_len,
        accelerator=accelerator,
        cache_dir=args.cache_dir,
        model_type=args.model_type,
        language_to_script_id=language_to_script_id,
    )

    # Save config file
    save_args_to_json(args, args.output_dir)
    task_type = get_task_type(args.dataset_name)
    args.task_type = task_type
    if task_type == "token_cls":
        # Use DataCollatorForTokenClassification for token classification tasks

        data_collator = DataCollatorForTokenClassification(
            tokenizer=vocab.tokenizer,
            label_pad_token_id=-100,  # -100 is the default for ignoring labels in token classification
            padding="longest",
        )
    else:
        data_collator = lambda x: dynamic_padding_data_collator(
            x, vocab.tokenizer
        )  # default_data_collator

    train_dataloader = DataLoader(
        data_corpus.train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        data_corpus.validation_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )
    test_dataloader = DataLoader(
        data_corpus.test_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )
    # Initialize Classification model
    logger.info("Initializing model ....")
    model = FxTAverageSingleInputWithPadding(
        data_corpus.num_labels, pretrained_model, task=task_type
    )
    logger.info(model)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_b1, args.adam_b2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "scheduler"
        ]  # .value
        accelerator.init_trackers(
            project_name="gradient-based-tokenization",
            config=experiment_config,
            init_kwargs={"wandb": {"entity": "owos", "name": basename}},
        )
        # pass

    # Get the metric function
    if args.dataset_name == "xnli":
        metric = evaluate.load(
            "xnli", cache_dir="cache", experiment_id=f"{basename}_xnli"
        )
    elif args.dataset_name == "paws-x":
        metric = evaluate.load(
            "accuracy", cache_dir="cache", experiment_id=f"{basename}_acc"
        )
    elif args.task_type == "token_cls":
        metric = evaluate.load(
            "seqeval", cache_dir="cache", experiment_id=f"{basename}_seqeval"
        )
    else:
        metric = evaluate.load(
            "accuracy", cache_dir="cache", experiment_id=f"{basename}_acc"
        )

    # Evaluate the model before training on test set only
    logger.info("Evaluating model on test set before training")
    initial_test_metrics, initial_test_predictions, initial_test_targets = (
        evaluate_model(
            model,
            test_dataloader,
            accelerator,
            metric,
            args,
            phase="test",
            return_predictions=True,
        )
    )
    logger.info(f"Initial test metrics: {initial_test_metrics}")

    # Store all initial test metrics for later comparison
    initial_results = {}

    # Add all test metrics with pre_training_ prefix
    for key, value in initial_test_metrics.items():
        initial_results[f"pre_training_{key}"] = value

    # Save initial test predictions
    initial_predict_file = os.path.join(
        args.output_dir, "pre_training_predict_results.txt"
    )
    with open(initial_predict_file, "w") as writer:
        logger.info("***** Pre-training predict results *****")
        writer.write("index\tprediction\treference\n")
        for index, (pred, targ) in enumerate(
            zip(initial_test_predictions, initial_test_targets)
        ):
            writer.write(f"{index}\t{pred}\t{targ}\n")

    logger.info("Pre-training predict results saved at {}".format(initial_predict_file))

    # Train!
    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(data_corpus.train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    completed_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        starting_epoch = int(args.resume_from_checkpoint.split("_")[-1])
        completed_steps = int(
            (args.max_train_steps / args.num_train_epochs) * starting_epoch
        )
        logger.info(
            f"Resuming from epoch: {starting_epoch} and step: {completed_steps}"
        )
        print(f"Skipping first {completed_steps} batches")
        active_dataloader = accelerator.skip_first_batches(
            train_dataloader, completed_steps
        )
    else:
        logger.info("No checkpoint found. Starting from scratch.")
        resume_from_checkpoint = None

    # Variables to track best model
    best_eval_loss = float("inf")
    best_epoch = -1

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            if args.joint_input:
                classification_loss, _, stats, boundary_loss = model(batch)
                boundary_loss = boundary_loss[0]
                loss = classification_loss + boundary_loss
            else:
                loss, _, stats = model(batch["x_ids"], batch["y_ids"], batch["labels"])

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

            # Gradient Clipping
            accelerator.clip_grad_norm_(
                model.parameters(),
                args.clip,
            )

            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                if step % args.logging_steps == 0:
                    grad_norm_ = grad_norm(model)
                current_lr = lr_scheduler.get_last_lr()[0]

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if step % args.logging_steps == 0:
                if args.with_tracking:
                    # accelerator.log({"train_loss": loss}, step=completed_steps,)
                    accelerator.log(
                        {
                            "train_cls_loss": classification_loss,
                            "train_loss": loss,
                            "train_boundary_loss": boundary_loss,
                            "learning_rate": current_lr,
                            "grad_norm": grad_norm_,
                        },
                        step=completed_steps,
                    )
                    accelerator.log(
                        stats,
                        step=completed_steps,
                    )
                    logger.info(f"stats are {stats}")

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    # accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        ##########################################
        # Evaluate on validation set
        ##########################################
        logger.info(f"Evaluating validation set for epoch {epoch}")
        eval_metrics = evaluate_model(
            model, eval_dataloader, accelerator, metric, args, phase="valid"
        )
        eval_loss = eval_metrics["valid_loss"]
        eval_metric = eval_metrics["valid_accuracy"]

        logger.info(f"epoch {epoch}: valid {eval_metric} valid loss {eval_loss}")

        # Check if this is the best model so far based on validation loss
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            logger.info(
                f"New best model found at epoch {epoch} with validation loss: {eval_loss}"
            )

            # Save the best model checkpoint
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(
                    {
                        "model": unwrapped_model.state_dict(),
                        "optimizer": optimizer.optimizer.state_dict(),
                        "epoch": epoch,
                        "step": completed_steps,
                        "eval_loss": best_eval_loss,
                        "eval_metric": eval_metric,
                    },
                    os.path.join(best_model_dir, f"{best_epoch}_model.pth"),
                )
                logger.info(f"Saved best model checkpoint to {best_model_dir}")
        metrics_dict = {
            "valid_accuracy": eval_metric,
            "train_loss": total_loss.item() / len(train_dataloader),
            "eval_loss": eval_loss,
            "epoch": epoch,
            "step": completed_steps,
            "best_eval_loss": best_eval_loss,
            "best_epoch": best_epoch,
            "best_model_path": os.path.join(best_model_dir, f"{best_epoch}_model.pth"),
        }

        if args.with_tracking:
            accelerator.log(
                metrics_dict,
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    ##########################################
    # Load the best model for test evaluation
    ##########################################

    accelerator.wait_for_everyone()

    if args.use_best_model:
        # Load the best model checkpoint
        logger.info(f"Loading best model from epoch {best_epoch} for test evaluation")
        best_model_path = os.path.join(best_model_dir, f"{best_epoch}_model.pth")
        if os.path.exists(best_model_path):

            # Initialize a new model instance
            best_model = FxTAverageSingleInputWithPadding(
                data_corpus.num_labels, pretrained_model
            )
            # Load state dict
            best_checkpoint = torch.load(best_model_path)
            best_model.load_state_dict(best_checkpoint["model"])

            # Prepare the model with accelerator
            best_model = accelerator.prepare(best_model)

            # Replace current model with best model for evaluation
            model = best_model
            logger.info(
                f"Successfully loaded best model from epoch {best_checkpoint['epoch']} "
                f"with validation loss {best_checkpoint['eval_loss']}"
            )
        else:
            logger.warning(
                f"Best model checkpoint not found at {best_model_path}. "
                f"Using the final model for evaluation."
            )

    ##########################################
    # Evaluate on the test set
    ##########################################
    logger.info("Evaluating test set")
    if args.dataset_name == "xnli":
        test_metric = evaluate.load(
            "xnli", cache_dir="cache", experiment_id=f"{basename}_xnli"
        )
    elif args.dataset_name == "paws-x":
        test_metric = evaluate.load(
            "accuracy", cache_dir="cache", experiment_id=f"{basename}_acc"
        )
    elif args.task_type == "token_cls":
        test_metric = evaluate.load(
            "seqeval", cache_dir="cache", experiment_id=f"{basename}_seqeval"
        )
    else:
        test_metric = evaluate.load(
            "accuracy", cache_dir="cache", experiment_id=f"{basename}_acc"
        )

    test_metrics, all_predictions, all_targets = evaluate_model(
        model,
        test_dataloader,
        accelerator,
        test_metric,
        args,
        phase="test",
        return_predictions=True,
    )
    final_test_loss = test_metrics["test_loss"]
    test_metric = test_metrics["test_accuracy"]

    logger.info(f"epoch {epoch}: test {test_metric} test loss {final_test_loss}")

    test_metrics_dict = {
        "test_accuracy": test_metric,
        "test_loss": final_test_loss,
        "epoch": epoch,
        "step": completed_steps,
    }

    if args.with_tracking:
        accelerator.log(
            test_metrics_dict,
            step=completed_steps,
        )

    if args.with_tracking:
        accelerator.end_training()

    final_metrics_dict = {
        "test_accuracy": test_metric,
        "valid_accuracy": eval_metric,
        "train_loss": total_loss.item() / len(train_dataloader),
        "valid_loss": eval_loss,
        "test_loss": final_test_loss,
    }

    # Save Test predictions
    output_predict_file = os.path.join(args.output_dir, "predict_results.txt")
    with open(output_predict_file, "w") as writer:
        logger.info("***** Predict results *****")
        writer.write("index\tprediction\treference\n")
        for index, (pred, targ) in enumerate(zip(all_predictions, all_targets)):
            writer.write(f"{index}\t{pred}\t{targ}\n")

    logger.info("Predict results saved at {}".format(output_predict_file))

    # Save final comparison results showing before/after training metrics for test set only
    comparison_file = os.path.join(args.output_dir, "training_comparison.txt")
    with open(comparison_file, "w") as writer:
        writer.write(
            "***** Model Performance Comparison Before and After Training *****\n\n"
        )

        # Write all test metrics only
        writer.write("Test Metrics:\n")
        for key, value in initial_test_metrics.items():
            if isinstance(value, (int, float)):
                writer.write(f"  Pre-training {key}: {value}\n")

        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                writer.write(f"  Post-training {key}: {value}\n")

        # Calculate improvements for all numeric metrics
        writer.write("\nImprovement Summary:\n")
        for key in initial_test_metrics.keys():
            if (
                key in test_metrics
                and isinstance(initial_test_metrics[key], (int, float))
                and isinstance(test_metrics[key], (int, float))
            ):
                improvement = test_metrics[key] - initial_test_metrics[key]
                writer.write(f"  {key} improvement: {improvement:.4f}\n")

    logger.info(f"Training comparison results saved at {comparison_file}")

    # Include all initial metrics in the final metrics dictionary with pre_training_ prefix
    for key, value in initial_results.items():
        final_metrics_dict[key] = value

    # Calculate and include improvements for all metrics
    for key in initial_test_metrics.keys():
        if (
            key in test_metrics
            and isinstance(initial_test_metrics[key], (int, float))
            and isinstance(test_metrics[key], (int, float))
        ):
            improvement = test_metrics[key] - initial_test_metrics[key]
            final_metrics_dict[f"{key}_improvement"] = improvement

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(
            {
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.optimizer.state_dict(),  # optimizer is an AcceleratedOptimizer object
            },
            os.path.join(args.output_dir, "model.pth"),
        )

        # save results into a json file
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(final_metrics_dict, f)


if __name__ == "__main__":
    main()
