from flask import Flask, request, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the model
DIR = "C:/Models"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_model = T5ForConditionalGeneration.from_pretrained(DIR).to(device)
my_tokenizer = T5Tokenizer.from_pretrained(DIR)

def generate_summary(article_text):
    input_text = f"summarize: {article_text}"
    inputs = my_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    summary_ids = my_model.generate(
        inputs["input_ids"],
        max_length=250,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )
    summary = my_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Set up the Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        article = request.form["article"]
        if article.strip():
            summary = generate_summary(article)
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
