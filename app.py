import gradio as gr
from transformers import pipeline

# 1. SETUP MODELS
# We put them in a list to loop through them easily
MODEL_IDS = [
    "shri171981/genai_model_dberta_base",
    "shri171981/genai_model_dberta_large",
    "shri171981/genai_model_roberta_base"
]

# 2. DEFINE LABELS
LABELS = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]

# 3. LOAD PIPELINES
pipelines = []
for model_id in MODEL_IDS:
    try:
        print(f"Loading {model_id}...")
        # top_k=None ensures we get probabilities for ALL classes, not just the top 1
        p = pipeline("text-classification", model=model_id, top_k=None)
        pipelines.append(p)
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")

def predict(text):
    # Initialize a dictionary to hold the sum of scores: {"Anger": 0.0, "Fear": 0.0, ...}
    final_scores = {label: 0.0 for label in LABELS}
    
    # 1. Loop through each model pipeline
    for pipe in pipelines:
        # Get raw result: [[{'label': 'LABEL_0', 'score': 0.9}, ...]]
        results = pipe(text)[0]
        
        # 2. Add this model's scores to the total
        for result in results:
            label_id = int(result['label'].split('_')[-1]) # e.g., "LABEL_0" -> 0
            label_name = LABELS[label_id]              # e.g., 0 -> "Anger"
            score = result['score']
            
            # Add to the running total
            final_scores[label_name] += score

    # 3. Average the scores
    # Divide each total by the number of models (3)
    num_models = len(pipelines)
    averaged_scores = {k: v / num_models for k, v in final_scores.items()}
    
    return averaged_scores

# 4. DEFINE UI
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """
        # 🤖 Ensemble Emotion Classifier
        ### merging predictions from 3 models (DeBERTa-Base, DeBERTa-Large, RoBERTa-Base)
        This app runs your text through **all three models** and averages their confidence scores for higher accuracy.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text", 
                placeholder="Type something emotional here...",
                lines=3
            )
            submit_btn = gr.Button("Analyze with Ensemble", variant="primary")
            
            gr.Examples(
                examples=[
                    ["I can't believe you would betray me like this!"], 
                    ["I heard a strange noise outside and I'm scared to look."], 
                    ["I finally got the promotion! This is the best day ever!"], 
                    ["I feel so lonely and empty inside."], 
                    ["Wow! I never expected a surprise party!"] 
                ],
                inputs=input_text
            )
            
        with gr.Column():
            # The chart will automatically show the averaged probabilities
            output_chart = gr.Label(label="Ensemble Confidence Scores", num_top_classes=5)

    # Link the button
    submit_btn.click(fn=predict, inputs=input_text, outputs=output_chart)

# 5. LAUNCH
if __name__ == "__main__":
    demo.launch()