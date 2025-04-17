import math

from flask import Flask, request, render_template, redirect, url_for, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def compute_tfidf(text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])

    words = vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray()[0]

    tokens = vectorizer.build_analyzer()(text)

    df = pd.DataFrame({
        'word': words,
        'tf': [tokens.count(w) for w in words],
        'idf': tfidf_values
    })

    df = df.sort_values(by='idf', ascending=False).head(50)
    return df


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    temp_path = 'storage/tfidf_temp.csv'
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename.endswith('.txt'):
            filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(filepath)

            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            table = compute_tfidf(text)
            table.to_csv('storage/tfidf_temp.csv', index=False)
            return redirect(url_for('show_results', page=1))
        else:
            return "Only .txt files are supported.", 400

    return render_template('index.html')

@app.route('/results/<int:page>')
def show_results(page):
    df = pd.read_csv('storage/tfidf_temp.csv')
    per_page = 10
    total_pages = math.ceil(len(df) / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    table = df.iloc[start:end].to_html(index=False)

    return render_template('results.html', table=table, page=page, total_pages=total_pages)

@app.route('/download')
def download_csv():
    return send_file('storage/tfidf_temp.csv', as_attachment=True, download_name='tfidf_results.csv')


if __name__ == '__main__':
    app.run(debug=True)
