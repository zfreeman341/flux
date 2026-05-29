use thiserror::Error;

#[derive(Debug, Error)]
pub enum FluxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse workflow TOML: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("invalid workflow: {0}")]
    Config(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("tool call limit exceeded")]
    ToolCallLimitExceeded,

    #[error("template error: {0}")]
    Template(String),

    #[error("budget exceeded: spent ${spent:.4} of ${limit:.4} limit")]
    BudgetExceeded { spent: f64, limit: f64 },

    #[error("agent error: {0}")]
    Agent(String),
}

pub type Result<T> = std::result::Result<T, FluxError>;
