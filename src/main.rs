use openai_api_rs::v1::api::OpenAIClient;
use openai_api_rs::v1::chat_completion::{self, ChatCompletionRequest};
use std::env;

const ENDPOINT: &str = "http://localhost:8080/v1";
const DEFAULT_API_KEY: &str = "EMPTY";
const MODEL_NAME: &str = "DEFAULT";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("OPENAI_API_KEY")
        .unwrap_or(DEFAULT_API_KEY.to_owned())
        .to_string();

    let client = OpenAIClient::builder()
        .with_endpoint(ENDPOINT.to_owned())
        .with_api_key(api_key)
        .build()?;

    let req = ChatCompletionRequest::new(
        MODEL_NAME.to_string(),
        vec![chat_completion::ChatCompletionMessage {
            role: chat_completion::MessageRole::user,
            content: chat_completion::Content::Text(String::from("Hi!")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
    );

    let result = client.chat_completion(req).await?;

    println!("Content: {:?}", result.choices[0].message.content);
    println!("Response Headers: {:?}", result.headers);

    Ok(())
}
