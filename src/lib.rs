pub mod agent;
pub mod anthropic;
pub mod cli;
pub mod engine;
pub mod error;
pub mod parser;
pub mod provider;
pub mod template;
pub mod validator;
pub mod workflow;

pub use error::{FluxError, Result};

/// Expands a leading `~/` to the user's home directory.
/// Paths without a leading `~/` are returned unchanged.
pub fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return format!("{}/{}", home.to_string_lossy(), rest);
    }
    path.to_string()
}
