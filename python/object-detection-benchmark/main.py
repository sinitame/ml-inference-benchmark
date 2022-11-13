import argparse
from dataset import MnistDataset
from pathlib import Path
import onnxruntime as ort
import numpy as np
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="MNIST evaluation CLI")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.set_defaults(func=_load_model)

    return parser

def _load_model(args):
    return perform_benchmark(Path(args.model_path), Path(args.dataset_path))

def perform_benchmark(model_path: Path, dataset_path: Path):
    # Create ONNX runtime inference request
    sess = ort.InferenceSession(str(model_path))

    # Load MNIST dataset and labels
    dataset = MnistDataset(str(dataset_path / "t10k-images.idx3-ubyte"))
    dataset.load_labels(str(dataset_path / "t10k-labels.idx1-ubyte"))

    # Perform benchmark
    good_results = 0
    input_name = sess.get_inputs()[0].name
    start = time.time()
    for mnist_image in dataset.images:
        image = mnist_image.to_tensor()
        outputs = sess.run(None, {input_name: image})
        prediction=int(np.argmax(np.array(outputs).squeeze(), axis=0))
        good_results += prediction == mnist_image.label
    end = time.time()

    print("Duration in (s): ", round((end - start), 3))
    print("Precision: ", good_results / len(dataset.images))

if __name__ == "__main__":
    arg_parser = parse_arguments()
    args = arg_parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        arg_parser.print_help()
        exit(1)
