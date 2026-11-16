mod mnist_nn;
mod model;

fn main() -> Result<(), model::Error> {
    pretty_env_logger::init();
    if let Some(path) = std::env::args().nth(1) {
        return mnist_nn::predict_file(&path);
    }
    mnist_nn::train_mnist()
}
