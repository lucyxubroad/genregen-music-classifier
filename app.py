import os
import torch
import torchaudio
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import urllib.request

import torch.nn as nn
import torch.nn.functional as F

from flask import Flask,render_template,url_for,request, flash, redirect, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename
from words import words

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.fc1 = nn.Linear(364416, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 364416)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

labels = ['country', 'hiphop/pop', 'classical', 'rock', 'jazz']

# Return true if uploaded file is of correct type (.wav)
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ALLOWED_EXTENSIONS = {'wav', 'mp3'}
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory where we want to save uploads
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/', methods=['GET'])
def home():
	return render_template('home.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        lyrics = request.form.get('lyrics')
        if lyrics and len(lyrics) > 0:
            lyrics_classification = words.classify_lyrics(lyrics)
            # return render_template('home.html', lyrics_classification=lyrics_classication)
        if 'file' not in request.files:
            flash('No file part')
            return render_template('home.html', lyrics_classification=lyrics_classification)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('home.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path_name)
            waveform, sample_rate = torchaudio.load(path_name)
            plt.plot(sample_rate)
            figure = plt.figure(figsize=(15, 4), dpi=100)
            plt.plot(waveform.t().numpy())
            plt.axis('off')
            axes = plt.gca()
            axes.set_ylim([-1.25, 1.25])
            figure_location = os.path.join(app.config['UPLOAD_FOLDER'], filename + '.png')
            figure.savefig(figure_location, bbox_inches="tight")
            waveform_name = './static/uploads/'+filename+'.png'
            transform = transforms.Compose([transforms.ToTensor()])
            img = Image.open(waveform_name).convert('RGB')
            img_t = transform(img)
            img_t = img_t.unsqueeze(0)
            PATH = '../img_processing/genregen_20e5gsNet.pth'
            model = Net()
            model.load_state_dict(torch.load(PATH))
            model.eval()
            out = model(img_t)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
            _, indices = torch.sort(out, descending=True)
            prediction_text = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
            return render_template('home.html', waveform_name=waveform_name, upload_name=filename, prediction_text=prediction_text, lyrics_classification=lyrics_classification)
        else:
            return redirect(request.url) 

if __name__ == '__main__':
	app.run(debug=True)