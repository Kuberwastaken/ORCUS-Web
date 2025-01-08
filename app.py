from flask import Flask, request, jsonify
from flask_cors import CORS
from model.detector import analyze_comment

app = Flask(__name__)
CORS(app)

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
