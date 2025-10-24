import time
import uuid

def create_openai_response(model_name, text_content):
    """
    Formats a text response into an OpenAI Chat Completion compatible dictionary.
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,  # Dummy value, as we are not tracking tokens
            "completion_tokens": 0, # Dummy value
            "total_tokens": 0, # Dummy value
        },
    }
