mod dataset;

use anyhow::Context;
use std::{time::Instant, path::PathBuf, fs::File};
use structopt::StructOpt;
use tract_onnx::prelude::*;

#[derive(Debug, StructOpt)]
pub struct EvaluationCli {
    #[structopt(long)]
    pub model_path: PathBuf,
    #[structopt(long)]
    pub dataset_path: PathBuf,
}


fn main() -> TractResult<()> {
    let args = EvaluationCli::from_args();
    let model = tract_onnx::onnx()
        // load the model
        .model_for_path(args.model_path)
        .context("Could not locate model")?
        // specify input type and shape
        .with_input_fact(0, f32::fact(&[1, 1, 28, 28]).into())?;

    // optimize the model
    let model = model
        .into_optimized()
        .context("During optimize")?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    // Load dataset and labels
    let images_path = args.dataset_path.join("t10k-images.idx3-ubyte");
    let mut images_file = File::open(&images_path).context("Could not open images file")?;
    let labels_path = args.dataset_path.join("t10k-labels.idx1-ubyte");
    let mut labels_file = File::open(&labels_path).context("Could not open labels file")?;
    let mut dataset = dataset::MnistDataset::load_images(&mut images_file)?;
    dataset.load_labels(&mut labels_file)?;

    // Perform inference on the dataset
    let mut good_results = 0;
    let start = Instant::now();
    for mnist_image in dataset.images.iter() {
        let image = mnist_image.to_tensor()?; 
        let result = model.run(tvec!(image))?;
        let best = result[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .zip(0..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if best.unwrap().1 == mnist_image.label.unwrap() as usize { good_results += 1 };
    }

    println!("Duration in (s): {}", start.elapsed().as_secs_f32());
    println!("Precision {}", good_results as f32 / dataset.images.len() as f32);
    Ok(())
}
