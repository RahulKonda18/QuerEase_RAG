import requests
import json

def generate_response(query, text):
    url = "https://api.together.xyz/v1/chat/completions"
    content = "My Query is " + query  + "and related Text is " + text + " Generate a response based on the following info."
    # Dynamically insert query and text into the user message
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "context_length_exceeded_behavior": "error",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for a food delivery platform called Swiggy. I will give you some related texts based on that, answer for the user query in a friendly manner."},  # Optional system message
            {"role": "user", "content": content}  # User input message
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer 32775a38c135b4852b36d27dd4cfac8f4c1f77b1630ae47ec33242a39e08c7c2"
    }

    response = requests.post(url, json=payload, headers=headers)

    # Parse the response JSON
    json_data = json.loads(response.text)

    # Return the assistant's response
    return json_data["choices"][0]["message"]["content"]

# Example usage
query = "What is the capital of India?"
text = "The capital of India is New Delhi."
print(generate_response(query, text))