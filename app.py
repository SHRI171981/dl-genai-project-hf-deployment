import gradio as gr
from transformers import pipeline

# 1. SETUP MODEL
MODEL_ID = "shri171981/genai_project_deployment"

# 2. DEFINE LABELS
LABELS = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]

# 3. LOAD PIPELINE
pipe = pipeline("text-classification", model=MODEL_ID, top_k=None)

def predict(text):
    # Run inference
    results = pipe(text)[0]
    
    # Create a dictionary of { "Joy": 0.95, "Anger": 0.02 } for Gradio
    output_scores = {}
    for result in results:
        # Extract the ID (e.g., "LABEL_0" -> 0)
        label_id = int(result['label'].split('_')[-1])
        
        # Map ID to the text name (e.g., 0 -> "Anger")
        label_name = LABELS[label_id]
        
        # Add to dict
        output_scores[label_name] = result['score']
    
    return output_scores

# 4. UI
theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="slate",
)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Emotion Classifier
        ### Analyze the sentiment of your text across 5 different emotions.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text", 
                placeholder="Type something emotional here...",
                lines=3
            )
            submit_btn = gr.Button("Analyze Emotion", variant="primary")
            
            gr.Examples(
                examples=[
                    ["I can't believe you would betray me like this!"], # Anger
                    ["I heard a strange noise outside and I'm scared to look."], # Fear
                    ["I finally got the promotion! This is the best day ever!"], # Joy
                    ["I feel so lonely and empty inside."], # Sadness
                    ["Wow! I never expected a surprise party!"] # Surprise
                ],
                inputs=input_text
            )
            
        with gr.Column():
            # This component creates the bar chart automatically
            output_chart = gr.Label(label="Emotion Confidence Scores", num_top_classes=5)

    # Link the button to the function
    submit_btn.click(fn=predict, inputs=input_text, outputs=output_chart)

# 5. LAUNCH
if __name__ == "__main__":
    demo.launch()