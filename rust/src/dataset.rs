use std::fs::File;
use std::io::{Cursor, Read};

use anyhow::{ensure, bail, Result};
use byteorder::{BigEndian, ReadBytesExt};
use tract_onnx::prelude::*;

pub struct MnistImage {
    // Row pixel values of the image (should have a len of 28x28)
    image: Vec<u8>,
    // Label of the image (number between 0 and 9)
    pub label: Option<u8>,
}

impl MnistImage {
    pub fn to_tensor(&self) -> Result<Tensor> {
        let tensor: Tensor = tract_ndarray::Array4::<f32>::from_shape_fn((1, 1, 28, 28), |(_, _, y, x)| {
            self.image[x + y * 28] as f32 / 255_f32
        }).into();
        Ok(tensor)
    }
}


pub struct MnistDataset {
    pub images: Vec<MnistImage>,
}

impl MnistDataset {
    pub fn load_images(f: &mut File) -> Result<MnistDataset> {
        // Put content of the file in a buffer
        let mut content = Vec::new();
        f.read_to_end(&mut content)?;
        
        // Access the elements in the buffer using a cursor
        let mut r = Cursor::new(&content);
        let magic_number = r.read_i32::<BigEndian>()?;

        let images = if magic_number == 2051 {
            let num_images = r.read_i32::<BigEndian>()?;
            let x = r.read_i32::<BigEndian>()?;
            let y = r.read_i32::<BigEndian>()?;
            let num_img_pixels = (x*y) as usize;

            let mut images = Vec::new();
            for _ in 0..num_images {
                let mut image = vec![0; num_img_pixels ];
                r.read_exact(&mut image)?;
                images.push(MnistImage { 
                    image,
                    label: None
                });
            }
            images
        } else {
            bail!("Invalid magic number for images")
        };

        Ok(MnistDataset {
            images,
        })
    }

    pub fn load_labels(&mut self, f: &mut File) -> Result<()> {
        // Put content of the file in a buffer
        let mut content = Vec::new();
        f.read_to_end(&mut content)?;
        
        // Access the elements in the buffer using a cursor
        let mut r = Cursor::new(&content);
        let magic_number = r.read_i32::<BigEndian>()?;

        if magic_number == 2049 {
            let num_labels = r.read_i32::<BigEndian>()? as usize;
            
            // Check that we have as many labels as images
            ensure!(num_labels == self.images.len(), "Mismatch between images dataset and labels");

            self.images.iter_mut().for_each(|img| img.label = Some(r.read_u8().unwrap()));
        } else {
            bail!("Invalid magic number for labels")
        }

        Ok(())
    }
}


mod test {
    #[test]
    fn test_dataset_loading() {
        let file_path = PathBuf::from("/Users/sinitame/Documents/dev/datasets/t10k-images.idx3-ubyte");
        let mut file = File::open(&file_path).unwrap();
        let dataset = MnistDataset::load_images(&mut file).unwrap();
        assert!(dataset.images.len() == 10000)
    }
    
    #[test]
    fn test_dataset_labels_loading() {
        let file_path = PathBuf::from("/Users/sinitame/Documents/dev/datasets/t10k-images.idx3-ubyte");
        let mut file = File::open(&file_path).unwrap();
        let mut dataset = MnistDataset::load_images(&mut file).unwrap();
        
        let labels_file_path = PathBuf::from("/Users/sinitame/Documents/dev/datasets/t10k-labels.idx1-ubyte");
        let mut labels_file = File::open(&labels_file_path).unwrap();
        dataset.load_labels(&mut labels_file).unwrap();
        assert!(dataset.images.len() == 10000)
    }
}
