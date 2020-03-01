from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename
import pandas as pd
from predictor.predictor import predict
import datetime


app = Flask(__name__, static_url_path='')

app.config['UPLOAD_FOLDER'] = 'downloads'

@app.route('/', methods=['GET', 'POST'])
def start():
    if request.method == 'POST':
        f = request.files['file']
        df = pd.read_csv(f)
        y = predict(df)
        df = pd.concat([df, y], axis=1)
        now = datetime.datetime.now()
        filepath = 'files/{}.csv'.format(now.strftime("%Y-%m-%d %H:%M:%S"))
        df.to_csv(filepath)
        return render_template('finished.html', url='/{}'.format(filepath))
    return render_template('start.html')


@app.route('/files/<path:path>')
def files(path):
    return send_from_directory('files', path)


if __name__ == '__main__':
   app.run(debug = True)
