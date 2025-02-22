import os
import model.train as train
from flask import Flask, render_template_string, request, render_template

# yes yes i know i know ugly global state. i'm sorry.
# need refactoring to be more respectable
base = 150
w2v_cp = 500
model = None
w2v = None
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        file_path = os.path.join(os.path.curdir, "totest.sol")
        file.save(file_path)

        return render_template('upload_success.html')
    return "No file selected!"

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return "Model not found"
    if w2v is None:
        return "Word2Vec model not found"
    if not os.path.exists("totest.sol"):
        return "No file uploaded"
    percent = train.analyze_file(model, w2v, "totest.sol")
    return render_template_string("{{ percent }}% likely", percent=percent)

if __name__ == '__main__':
    model, w2v = train.warmup(base, w2v_cp)
    app.run(debug=True)
