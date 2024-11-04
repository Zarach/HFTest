# Load Model from HF Model Hub mentioning the branch name in revision field
import math
import os

import numpy as np
import pandas as pd
import torch
from clearml import Dataset
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
from transformers import Trainer, EarlyStoppingCallback, AdamW, TrainingArguments
from tsfm_public import TinyTimeMixerForPrediction, TimeSeriesPreprocessor, TrackingCallback, count_parameters

TTM_MODEL_REVISION = "main"

df = pd.read_csv("GEMHouse/2a2357676efbe89fc69e96509c5e9e6527551d90a8d940715472a3a28ed8c8f.csv")
df.columns = [0, 1]

column_specifiers = {
        "timestamp_column": 0,
        #"id_columns": id_columns,
        "target_columns": [1],
        "control_columns": [],
    }

tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=512,
    prediction_length=96,
    scaling=True,
    encode_categorical=False,
    scaler_type="standard",
)

split_config = {
                "train": [0, 12 * 30 * 24],
                "valid": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24],
                "test": [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
            }

train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
    df, split_config, fewshot_fraction=1.0, fewshot_location="first"
)

def fewshot_finetune_eval(
        dataset_name,
        batch_size,
        learning_rate=0.001,
        context_length=512,
        forecast_length=24,
        fewshot_percent=20,
        freeze_backbone=True,
        num_epochs=250,
        save_dir="plots",
        prediction_filter_length=None
):
    out_dir = os.path.join(save_dir, dataset_name)

    print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)


    # change head dropout to 0.7 for ett datasets
    if "ett" in dataset_name:
        if prediction_filter_length is None:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm/TTM", revision=TTM_MODEL_REVISION, head_dropout=0.7
            )
        elif prediction_filter_length <= forecast_length:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm/TTM", revision=TTM_MODEL_REVISION, head_dropout=0.7,
                prediction_filter_length=prediction_filter_length
            )
        else:
            raise ValueError(f"`prediction_filter_length` should be <= `forecast_length")
    else:
        if prediction_filter_length is None:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm/TTM", revision=TTM_MODEL_REVISION,
            )
        elif prediction_filter_length <= forecast_length:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm/TTM", revision=TTM_MODEL_REVISION, prediction_filter_length=prediction_filter_length
            )
        else:
            raise ValueError(f"`prediction_filter_length` should be <= `forecast_length")
    if freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(finetune_forecast_model),
        )

        # Freeze the backbone of the model
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(finetune_forecast_model),
        )

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()

    # Evaluation
    print("+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20)
    fewshot_output = finetune_forecast_trainer.evaluate(test_dataset)
    print(fewshot_output)
    print("+" * 60)

    plot_preds(
        trainer=finetune_forecast_trainer,
        dset=train_dataset,
        plot_dir=os.path.join("plots", "gemHouse"),
        plot_prefix="test_fewshot",
        channel=0,
    )

def get_model_from_clearML(filepath):
    dataset_models = Dataset.get(dataset_project='ForeSightNEXT/Electric Load Forecasting', dataset_name='BeeneraGEMHouseProcVal')
    dataset_models.get_mutable_local_copy(filepath, True)

def plot_preds(trainer, dset, plot_dir, num_plots=10, plot_prefix="valid", channel=-1):
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    random_indices = np.random.choice(len(dset), size=num_plots, replace=False)
    random_samples = torch.stack([dset[i]["past_values"] for i in random_indices])
    output = trainer.model(random_samples.to(device=device))
    y_hat = output.prediction_outputs[:, :, channel].detach().cpu().numpy()
    pred_len = y_hat.shape[1]

    # Set a more beautiful style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Adjust figure size and subplot spacing
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 20))
    for i, ri in enumerate(random_indices):
        batch = dset[ri]

        y = batch["future_values"][:pred_len, channel].squeeze().cpu().numpy()
        x = batch["past_values"][: 2 * pred_len, channel].squeeze().cpu().numpy()
        y = np.concatenate((x, y), axis=0)

        # Plot predicted values with a dashed line
        y_hat_plot = np.concatenate((x, y_hat[i, ...]), axis=0)
        axs[i].plot(y_hat_plot, label="Predicted", linestyle="--", color="orange", linewidth=2)

        # Plot true values with a solid line
        axs[i].plot(y, label="True", linestyle="-", color="blue", linewidth=2)

        # Plot horizon border
        axs[i].axvline(x=2 * pred_len, color="r", linestyle="-")

        axs[i].set_title(f"Example {random_indices[i]}")
        axs[i].legend()

    # Adjust overall layout
    plt.tight_layout()

    # Save the plot
    plot_filename = f"synthetic_{plot_prefix}_ch_{str(channel)}.pdf"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, plot_filename))


def zero_shot():
    model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r1", revision="main")



    zs_args = ["GemHouse", 96, 512, 96, None]
    # Do zeroshot
    zeroshot_trainer = Trainer(
        model=model
    )


    zeroshot_output = zeroshot_trainer.evaluate(train_dataset)
    print(zeroshot_output)

    plot_preds(
        trainer=zeroshot_trainer,
        dset=train_dataset,
        plot_dir=os.path.join("plots", "gemHouse"),
        plot_prefix="test_zeroshot",
        channel=0,
    )

fewshot_finetune_eval("gemHouse", 32)


