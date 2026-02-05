use anyhow::Result;

// Import our MNIST neural network module
mod mnist_nn;

fn main() -> Result<()> {
    // Initialize the logger
    pretty_env_logger::init();
    println!("Starting MNIST neural network training with Mimir...");

    // Train the neural network on MNIST
    mnist_nn::train_mnist()?;

    Ok(())
}
