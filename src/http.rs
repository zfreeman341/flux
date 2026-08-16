//! The deterministic step kind.
//!
//! An [`crate::workflow::HttpStep`] draws nothing from the budget: it makes a
//! plain request and hands the response downstream untouched. Everything that
//! could be called interpretation belongs to the transform that follows it.
//!
//! Failures here are per-item, not per-step. An endpoint that 500s produces a row
//! with that status and an empty payload; the remaining items still flow
//! downstream, which is the same graceful-skip posture the LLM path uses.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tracing::{info, warn};

use crate::artifact::Artifact;
use crate::registry::Registry;
use crate::template::FanItem;
use crate::workflow::{HttpMethod, HttpMode, HttpStep};

/// Some APIs return multi-megabyte documents, so the timeout is generous.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Shared by every HTTP step so connections are pooled across the whole run.
pub fn client() -> crate::Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .user_agent("flux/0.1")
        .build()
        .map_err(|e| crate::FluxError::Http(e.to_string()))
}

pub struct HttpExecutor {
    client: reqwest::Client,
}

impl HttpExecutor {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Runs a step over its fan-out items, or once with no item bound.
    pub async fn run(
        &self,
        step: &HttpStep<'_>,
        items: &[FanItem],
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, Artifact>,
        max_concurrent: usize,
        registry: &Registry,
    ) -> crate::Result<Artifact> {
        match &step.mode {
            HttpMode::Probe(name) => {
                registry
                    .probe(name)?
                    .run(items, &self.client, max_concurrent)
                    .await
            }
            HttpMode::Request {
                method,
                url,
                body,
                headers,
            } => {
                let plan = self.plan_requests(items, url, *body, inputs, step_outputs)?;
                let rows = self.execute(plan, *method, headers, max_concurrent).await;
                Ok(Artifact::records(step.output_type.clone(), rows))
            }
        }
    }

    // Rendered up front so a template error surfaces before any request is issued.
    fn plan_requests(
        &self,
        items: &[FanItem],
        url: &str,
        body: Option<&str>,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, Artifact>,
    ) -> crate::Result<Vec<(Value, String, Option<String>)>> {
        let render_one =
            |item: Option<&FanItem>| -> crate::Result<(Value, String, Option<String>)> {
                let url = crate::template::render(url, inputs, step_outputs, item)?;
                let body = body
                    .map(|b| crate::template::render(b, inputs, step_outputs, item))
                    .transpose()?;
                let tag = item.map(|i| i.value.clone()).unwrap_or(Value::Null);
                Ok((tag, url, body))
            };

        if items.is_empty() {
            return Ok(vec![render_one(None)?]);
        }
        items.iter().map(|i| render_one(Some(i))).collect()
    }

    async fn execute(
        &self,
        plan: Vec<(Value, String, Option<String>)>,
        method: HttpMethod,
        headers: &HashMap<String, String>,
        max_concurrent: usize,
    ) -> Vec<Value> {
        let semaphore = Arc::new(Semaphore::new(max_concurrent.max(1)));

        let futures = plan.into_iter().map(|(tag, url, body)| {
            let sem = Arc::clone(&semaphore);
            let client = self.client.clone();
            let headers = headers.clone();
            async move {
                let _permit = sem.acquire().await.unwrap();
                let mut req = match method {
                    HttpMethod::Get => client.get(&url),
                    HttpMethod::Post => client.post(&url),
                };
                for (k, v) in &headers {
                    req = req.header(k, v);
                }
                if let Some(b) = body {
                    req = req.header("content-type", "application/json").body(b);
                }

                let (status, payload) = match req.send().await {
                    Ok(resp) => {
                        let status = resp.status().as_u16();
                        // A non-JSON body is not an error worth aborting for; record it as null.
                        let payload = resp.json::<Value>().await.unwrap_or(Value::Null);
                        (status, payload)
                    }
                    Err(e) => {
                        warn!(url = %url, error = %e, "http request failed");
                        (0, Value::Null)
                    }
                };
                json!({"url": url, "status": status, "item": tag, "payload": payload})
            }
        });

        let rows: Vec<Value> = join_all(futures).await;
        let ok = rows
            .iter()
            .filter(|r| matches!(r.get("status").and_then(Value::as_u64), Some(200..=299)))
            .count();
        info!(requests = rows.len(), ok, "http step complete");
        rows
    }
}
