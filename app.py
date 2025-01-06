from flask import Flask, render_template, request, jsonify
from model.detector import analyze_comment

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    comment = data.get('comment', '')
    if not comment:
        return jsonify({'error': 'No comment provided'}), 400
    
    result = analyze_comment(comment)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
