import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from helpers import (
    prepare_dataset_nli,
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
    QuestionAnsweringTrainer,
    compute_accuracy,
)
import os
import json

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument(
        "--model",
        type=str,
        default="google/electra-small-discriminator",
        help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""",
    )
    argp.add_argument(
        "--dataset",
        type=str,
        choices=["snli", "augsnli"],
        required=True,
        help="""Pick a dataset""",
    )
    argp.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""",
    )
    argp.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit the number of examples to train on.",
    )
    argp.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit the number of examples to evaluate on.",
    )

    training_args, args = argp.parse_args_into_dataclasses()

    # Load either snli or augsnli data
    if args.dataset == "snli":
        dataset = datasets.load_dataset("snli")
    elif args.dataset == "augsnli":
        data_base_path = "/scratch/rahul/nlp/augsnli_1.0/"
        dataset = datasets.load_dataset("csv", data_files={"train": data_base_path+"train.tsv", "validation": data_base_path+"dev.tsv", "test": data_base_path+"test.tsv"}, sep='\t', usecols=["Unnamed: 0", "index", "captionID", "sentence1", "sentence2", "gold_label"])
        dataset = dataset.rename_column("gold_label", "label")
        dataset = dataset.rename_column("sentence1", "premise")
        dataset = dataset.rename_column("sentence2", "hypothesis")
        dataset = dataset.rename_column("Unnamed: 0", "ct_idx")
        dataset = dataset.rename_column("index", "og_idx")
        dataset = dataset.rename_column("captionID", "trans")
        def label_to_int(ex):
            if ex["label"] == "entailment":
                ex["label"] = 0
            elif ex["label"] == "neutral":
                ex["label"] = 1
            elif ex["label"] == "contradiction":
                ex["label"] = 2
            else:
                ex["label"] = -1
            return ex
        dataset = dataset.map(label_to_int)
        def rename_trans(ex):
            if ex["trans"].startswith("*"):
                ex["trans"] = "original"
            return ex
        dataset = dataset.map(rename_trans)

    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {"num_labels": 3}

    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    prepare_train_dataset = prepare_eval_dataset = lambda exs: prepare_dataset_nli(
        exs, tokenizer, args.max_length
    )

    print(
        "Preprocessing data... (this takes a little bit, should only happen once per dataset)"
    )
    # remove SNLI examples with missing label, premise, or hypothesis
    dataset = dataset.filter(lambda ex: ex["label"] != -1)
    dataset = dataset.filter(lambda ex: ex["premise"] is not None)
    dataset = dataset.filter(lambda ex: ex["hypothesis"] is not None)

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset["train"]
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names,
        )
    elif training_args.do_eval:
        eval_dataset = dataset["test"]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names,
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = compute_accuracy

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.
        print("Evaluation results:")
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(
            os.path.join(training_args.output_dir, "eval_metrics.json"),
            encoding="utf-8",
            mode="w",
        ) as f:
            json.dump(results, f)

        with open(
            os.path.join(training_args.output_dir, "eval_predictions.jsonl"),
            encoding="utf-8",
            mode="w",
        ) as f:
            for i, example in enumerate(eval_dataset):
                example_with_prediction = dict(example)
                example_with_prediction["predicted_scores"] = (
                    eval_predictions.predictions[i].tolist()
                )
                example_with_prediction["predicted_label"] = int(
                    eval_predictions.predictions[i].argmax()
                )
                f.write(json.dumps(example_with_prediction))
                f.write("\n")


if __name__ == "__main__":
    main()
