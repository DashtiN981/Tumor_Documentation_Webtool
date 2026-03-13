from openai import OpenAI

client = OpenAI(api_key="API-Key-Here", base_url="Base-URL-Here")

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant!"
        },
        {
            "role": "user",
            "content": "What do you know about the KatherLab?",
        }
    ],
    # Model Name
      #model="llama-3.3-70b-instruct",
      model="GPT-OSS-120B", 
      #model="llama-3.3-70b-instruct-awq",
)

print(response.choices[0].message.content)