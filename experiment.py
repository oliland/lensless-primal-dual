import argparse
import os
import time
import timeit
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from piq import LPIPS, psnr, multi_scale_ssim
import torch
import torchvision
from torch.utils.data import DataLoader
import tqdm

from lensless.diffusercam import (
    LenslessLearning,
    LenslessLearningCollection,
    LenslessLearningInTheWild,
    region_of_interest,
)
from lensless.evaluate import EvaluationSystem
from lensless.model import ImageOptimizer
from lensless.model_colors import ImageOptimizerMixColors
from lensless.training import TrainingSystem

# Path to datasets

DEFAULT_LOGS_DIR = 'logs/'
DEFAULT_RESULTS_DIR = 'results/'

MODEL_CLASSES = {
    "learned-primal": ImageOptimizer,
    "learned-primal-and-model": ImageOptimizer,
    "learned-primal-dual": ImageOptimizer,
    "learned-primal-dual-and-model": ImageOptimizer,
    "learned-primal-dual-and-five-models": ImageOptimizer,
    "learned-primal-dual-and-color-mixing": ImageOptimizerMixColors,
}

# List of models that this script can train and evaluate.
TRAINABLE_MODELS = {
    "learned-primal": {
        "width": 5,
        "depth": 10,
        "learned_models": 0,
        "primal_only": True,
    },
    "learned-primal-and-model": {
        "width": 5,
        "depth": 10,
        "learned_models": 1,
        "primal_only": True,
    },
    "learned-primal-dual": {
        "width": 5,
        "depth": 10,
        "learned_models": 0,
    },
    "learned-primal-dual-and-model": {
        "width": 5,
        "depth": 10,
        "learned_models": 1,
    },
    "learned-primal-dual-and-five-models": {
        "width": 5,
        "depth": 10,
        "learned_models": 5,
    },
    "learned-primal-dual-and-color-mixing": {
        "depth": 10,
    },
}


# Benchmark models have not been re-trained, as this requires reproducing somebody elses environment exactly.
# Instead, a separate script has loaded the weights, evaluated the above datasets, and stored the results in a tensor.
# Having reproduced the exact outputs, we can re-evaluate these models using our own metrics.
BENCHMARK_MODELS = {
    "le-admm-u": "/home/oliland/Results/waller_lab/le-admm-u",
}

# The set of all models
MODELS = list(BENCHMARK_MODELS) + list(TRAINABLE_MODELS)

# Some models require custom postprocessing
POSTPROCESS_MODELS = {
    "le-admm-u",
    "admm-converged",
}

device = torch.device('cuda')


def main():
    print("Lensless Imaging Model Tournament. Add -h for help")
    
    args = parse_arguments()

    if args.train:
        fit_models(args)

    if args.eval:
        evaluate_trainable_models(args)

    if args.bench:
        benchmark_trainable_models(args)

    if args.parameters:
        show_trainable_model_parameters(args)
    
    if args.images:
        generate_images(args)

    print("Done")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help="Path to Lensless Learning dataset")
    parser.add_argument('--eval', action="store_true", help="Evaluate trainable models")
    parser.add_argument('--train', action="store_true", help="Fit trainable models")
    parser.add_argument('--report', action="store_true", help="Generate report from evals")
    parser.add_argument('--images', action="store_true", help="Write images from evaluated results")
    parser.add_argument('--version', default=0, help="Model version")
    parser.add_argument('--bench', action="store_true", help="Benchmark")
    parser.add_argument('--parameters', action="store_true", help="Show number of parameters")
    parser.add_argument('--checkpoint')
    parser.add_argument('--disable-unet', action="store_true")
    parser.add_argument('--models', nargs='*', default=TRAINABLE_MODELS, choices=MODELS)
    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--accelerator', type=str, default='gpu', help="Accelerator to use")
    parser.add_argument('--devices', type=int, default=1, help="Number of devices to use")
    parser.add_argument('--logs', default=DEFAULT_LOGS_DIR, type=Path, help="Logs directory")
    parser.add_argument('--results', default=DEFAULT_RESULTS_DIR, type=Path, help="Results directory")
    return parser.parse_args()


def fit_models(args):
    collection = LenslessLearningCollection(args.dataset_path)
    for name in args.models:
        model = load_model_with_name(collection, name)
        model = load_training_system(name=name, model=model, region_of_interest=collection.region_of_interest)
        run_experiment(
            name=name,
            model=model,
            train_dataset=collection.train_dataset,
            val_dataset=collection.val_dataset,
            params=args,
        )


def load_training_system(name, model, region_of_interest):
    return TrainingSystem(model=model, region_of_interest=region_of_interest)


def run_experiment(name, model, train_dataset, val_dataset, params):
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=4, persistent_workers=True)
    val_dataloader   = DataLoader(val_dataset, shuffle=False, batch_size=2, num_workers=4, persistent_workers=True)

    logger = TensorBoardLogger(params.logs, name=name)
    
    callbacks = [
        ModelCheckpoint(filename="weights"),
    ]

    trainer = Trainer(
        accelerator=params.accelerator,
        devices=params.devices,
        callbacks=callbacks,
        max_epochs=params.max_epochs,
        logger=logger,
        resume_from_checkpoint=params.checkpoint,
        accumulate_grad_batches=2,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    

def show_trainable_model_parameters(args):
    collection = LenslessLearningCollection(args.dataset_path)

    for name in args.models:
        if name not in TRAINABLE_MODELS:
            print(f"Ignoring non-trainable model {name}")
            continue

        model = load_model_with_name(collection, name)
        learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if hasattr(model, 'psfs') and model.psfs.requires_grad:
            psf_parameters = model.psfs.numel()
        else:
            psf_parameters = 0
        if hasattr(model, 'unet'):
            unet_parameters = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
        else:
            unet_parameters = 0
        print(f"{name}: Total parameters {learnable_parameters}, PSF parameters {psf_parameters}, U-Net parameters {unet_parameters}")
    

def benchmark(model, inputs):
    def wrapped():
        return model(inputs)
    return wrapped


def benchmark_trainable_models(args):
    collection = LenslessLearningCollection(args.dataset_path)

    number = 100
    inputs = torch.zeros(1, *collection.psf.shape).to(device)
    
    # make GPU go brrrr
    print("Warming up...")
    warmup = args.models[0]
    model = evaluate_model_from_checkpoint(collection, warmup, args.logs, args.version)
    timeit.timeit(benchmark(model, inputs), number=number)
    print("GPU is warm")

    for name in args.models:
        if name not in TRAINABLE_MODELS:
            print(f"Ignoring non-trainable model {name}")
            continue

        torch.cuda.empty_cache()
        model = evaluate_model_from_checkpoint(collection, name, args.logs, args.version)
        total_duration_seconds = timeit.timeit(benchmark(model, inputs), number=number)
        duration_seconds = total_duration_seconds / number
        duration_ms = round(duration_seconds * 1000)
        memory_allocated_mb = int(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
        print(f"Benchmark results for {name}: {duration_ms}ms / {memory_allocated_mb}Mb")


def generate_images(args):
    collection = LenslessLearningCollection(args.dataset_path)
    
    for name in args.models:
        postprocess = name in POSTPROCESS_MODELS

        # Test images
        images = load_images(results_path(args, name, "test.torch"), postprocess=postprocess)
        write_images(images, args, name)

        # PSFs or Learned Models
        try:
            psfs = load_images(results_path(args, name, "psfs.torch"))
            if len(psfs.shape) == 5:
                b, c, v, h, w = psfs.shape
            else:
                b, v, h, w = psfs.shape
                c = 3
                psfs = psfs.unsqueeze(2).tile(1, 1, c, 1, 1)
            write_images_grid(psfs.reshape(b * v, c, h, w).clip(0, 1), output_path(args, name, "psfs.png"))
        except FileNotFoundError:
            pass
    
        print(f"Wrote images for {name} to {args.results}")


def stats_from_images(name, dataset, images):
    output = []
    
    mse_loss = torch.nn.MSELoss()
    lpips_loss = LPIPS(weights=VGG16_Weights.IMAGENET1K_V1)

    for i in tqdm.trange(0, len(images)):
        x, y = dataset[i]
        x = x.unsqueeze(0)
        y = region_of_interest(y.unsqueeze(0)).to(device)
        y_hat = region_of_interest(images[i].unsqueeze(0)).to(device)
        output.append({
            'model': name,
            'mse': mse_loss(y_hat, y).item(),
            'psnr': psnr(y_hat, y).item(),
            'ms-ssim': multi_scale_ssim(y_hat, y).item(),
            'lpips': lpips_loss(y_hat, y).item(),
        })
    return output


def load_images(path, postprocess=False):
    output = torch.load(path)
    if postprocess:
        return postprocess_images(output)
    else:
        return output


def postprocess_images(image):
    """
    Postprocess images from Monakhova et al (2019)
    """
    return torch.flip(torch.clip(image, 0, 1), (1, 2))


def write_images(images, args, name):
    for i in tqdm.trange(0, len(images)):
        image = region_of_interest(images[i])
        image = (image.cpu() * 255.).byte()
        image_path = output_path(args, name, f"{i}.png")
        torchvision.io.write_png(image, str(image_path))


def write_images_grid(images, path):
    grid = torchvision.utils.make_grid([x.cpu() for x in images], nrow=5, padding=10)
    torchvision.io.write_png((grid * 255.).byte(), str(path))


def evaluate_model_from_checkpoint(collection, name, logs_dir, version=0):
    checkpoint = torch.load(logs_dir / f"{name}/version_{version}/checkpoints/weights.ckpt", map_location=device)
    model = load_model_with_name(collection, name)
    model = EvaluationSystem(model=model, checkpoint=checkpoint)
    model.to(device)
    return model


def load_model_with_name(collection, name):
    if name not in TRAINABLE_MODELS:
        raise ValueError(f"No model named {name}")
    
    model_spec = TRAINABLE_MODELS[name]
    model_class = MODEL_CLASSES[name]
    return model_class(collection.psf, **TRAINABLE_MODELS[name])


def evaluate_trainable_models(args):
    collection = LenslessLearningCollection(args.dataset_path)

    for name in args.models:
        if name not in TRAINABLE_MODELS:
            print(f"Ignoring non-trainable model {name}")
            continue

        print(f"Generating images for {name}")
        
        model = evaluate_model_from_checkpoint(collection, name, args.logs, args.version)
        
        # print("Saving train images...")
        # torch.save(evaluate_model(collection, model, collection.train_dataset, disable_unet=args.disable_unet), results_path(args, name, "train.torch"))

        print("Saving test images...")
        torch.save(evaluate_model(collection, model, collection.val_dataset, disable_unet=args.disable_unet), results_path(args, name, "test.torch"))
        
        if model.psfs is not None:
            print("Saving learned models...")
            torch.save(model.psfs.cpu(), results_path(args, name, "psfs.torch"))
    
    print("Finished evaluating models")


def evaluate_model(collection, model, dataset, disable_unet=False):
    output = []

    for i in tqdm.trange(0, len(dataset)):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        output.append(model(x, denoise=not disable_unet)[0])
    return torch.stack(output)


def milliseconds_elapsed(start_time, end_time):
    return int((end_time - start_time) * 1000)


def results_path(args, model_name, result_name):
    """
    Results for benchmark datasets are stored independently as they are evaluated differently
    """
    if model_name in BENCHMARK_MODELS:
        return Path(BENCHMARK_MODELS[model_name]) / result_name
    else:
        return output_path(args, model_name, result_name)


def output_path(args, model_name, result_name):
    if args.disable_unet:
        experiment_name = f"{model_name}-denoise-off"
    else:
        experiment_name = model_name
    output_dir = args.results / experiment_name
    os.makedirs(output_dir, exist_ok=True)
    return output_dir / result_name


if __name__ == "__main__":
    main()
