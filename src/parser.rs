use std::path::Path;

use crate::workflow::WorkflowFile;

pub fn parse_workflow(path: &Path) -> crate::Result<WorkflowFile> {
    let content = std::fs::read_to_string(path)?;
    let workflow = toml::from_str(&content)?;
    Ok(workflow)
}
