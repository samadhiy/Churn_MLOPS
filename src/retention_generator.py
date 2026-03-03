import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load variables from .env file
load_dotenv()

# 2. Initialize the client. 
# It will automatically look for the "OPENAI_API_KEY" environment variable.
client = OpenAI()

def generate_retention(customer_features):
    prompt = f"""
    Customer details:
    {customer_features}

    Generate a short, personalized retention incentive for this customer.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenAI: {e}"

if __name__ == "__main__":
    # Ensure we load the .env file even if running from a different folder
    load_dotenv()
    
    key = os.getenv("OPENAI_API_KEY")
    
    # Debugging print (Shows first 7 characters: sk-proj)
    if key:
        print(f"Debug: Found key starting with: {key[:7]}...")
    else:
        print("Debug: No key found in environment.")

    # Simplified check: just check if the key exists
    if not key or len(key) < 10:
        print("❌ ERROR: Valid API key not found in the .env file.")
    else:
        print("✅ API Key loaded successfully. Generating incentive...")
        print(generate_retention("Fiber user, 36 months tenure"))