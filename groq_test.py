def get_ai_interpretation(audio_result, image_result, video_result):
    
    import requests
    import os
    import json
    
    audio_data = f"Audio classification: {audio_result['subcategory']} ({audio_result['subcategory_confidence']*100:.2f}%) in category {audio_result['category']} ({audio_result['category_confidence']*100:.2f}%)"
    
    image_data = "Image contains: "
    if "objects" in image_result and image_result["objects"]:
        image_data += ", ".join([f"{obj['tag']} ({obj['confidence']*100:.2f}%)" for obj in image_result["objects"][:5]])
    else:
        image_data += "No objects detected or an error occurred."
    
    video_data = "Video contains: "
    if "objects" in video_result and video_result["objects"]:
        video_data += ", ".join([f"{obj['tag']} ({obj['confidence']*100:.2f}%)" for obj in video_result["objects"][:5]])
    else:
        video_data += "No objects detected or an error occurred."
    
    prompt = f"""
    Based on the following analysis results, determine if the audio source matches the visual evidence:
    
    {audio_data}
    
    {image_data}
    
    {video_data}
    
    Does the audio prediction match what is seen in the image/video? 
    If yes, explain why the sound likely came from the detected object.
    If no, suggest what might have produced the sound instead (e.g., "dog sound was played through a speaker").
    Keep your answer concise and clear.
    """
    
    try:
        # Replace with your actual Groq API key
        api_key = os.environ.get("GROQ_API_KEY", "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")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Using Llama-3-8b model, but you can change to other available models
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        payload = {
            "model": "llama3-8b-8192",  # You can also use "mixtral-8x7b-32768" or other models
            "messages": [
                {"role": "system", "content": "You are an AI assistant that analyzes audio and visual data to determine if they match."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 300
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"Unexpected response format: {json.dumps(result)}"
        else:
            return f"API error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"AI interpretation error: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Example input data
    audio_result = {
        "category": "Animal",
        "category_confidence": 0.95,
        "subcategory": "Dog barking",
        "subcategory_confidence": 0.87
    }
    
    image_result = {
        "objects": [
            {"tag": "Dog", "confidence": 0.92},
            {"tag": "Person", "confidence": 0.85},
            {"tag": "Grass", "confidence": 0.78}
        ]
    }
    
    video_result = {
        "objects": [
            {"tag": "Dog", "confidence": 0.90},
            {"tag": "Person", "confidence": 0.88},
            {"tag": "Trees", "confidence": 0.72}
        ]
    }
    
    # Call the function with example data
    interpretation = get_ai_interpretation(audio_result, image_result, video_result)
    print("\nAI INTERPRETATION:")
    print(interpretation)
    
    # Another example with non-matching audio and visual
    print("\n\n--- SECOND EXAMPLE (MISMATCH) ---")
    
    audio_result_2 = {
        "category": "Vehicle",
        "category_confidence": 0.83,
        "subcategory": "Car engine",
        "subcategory_confidence": 0.76
    }
    
    image_result_2 = {
        "objects": [
            {"tag": "Dog", "confidence": 0.92},
            {"tag": "Person", "confidence": 0.85},
            {"tag": "Grass", "confidence": 0.78}
        ]
    }
    
    video_result_2 = {
        "objects": [
            {"tag": "Dog", "confidence": 0.90},
            {"tag": "Person", "confidence": 0.88},
            {"tag": "Trees", "confidence": 0.72}
        ]
    }
    
    interpretation_2 = get_ai_interpretation(audio_result_2, image_result_2, video_result_2)
    print("\nAI INTERPRETATION:")
    print(interpretation_2)