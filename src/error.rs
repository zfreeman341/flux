use thiserror::Error;

#[derive(Debug, Error)]
pub enum FluxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid workflow config: {0}")]
    Config(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, FluxError>;
