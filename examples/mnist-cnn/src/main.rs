use anyhow::Result;

// Import our MNIST neural network module
mod mnist_nn;

fn main() -> Result<()> {
    // Initialize the logger
    pretty_env_logger::init();
    if let Some(path) = std::env::args().nth(1) {
        return mnist_nn::NeuralNetwork::predict_file(&path);
    }
    println!("Starting improved MNIST dense-network training with Mimir...");

    // Train the neural network on MNIST
    mnist_nn::train_mnist()?;

    Ok(())
}
