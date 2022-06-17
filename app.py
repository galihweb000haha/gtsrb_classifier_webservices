import os
import torch
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import torch.nn as nn

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TESTING'] = True

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class CustomMobileNetV2(nn.Module) :
  def __init__(self, output_size):
    super().__init__()
    self.mnet = mobilenet_v2(pretrained=False)
    self.freeze()
    self.mnet.classifier = nn.Sequential(
        nn.Linear(1280, 43),
        nn.Sigmoid()
    )
  def forward(self, x):
    return self.mnet(x)

  def freeze(self):
    for param in self.mnet.parameters():
      param.requires_grad = False

  def unfreeze(self):
    for param in self.mnet.parameters():
      param.requires_grad = True

classes = [ 'Speed limit (20km/h)',
            'Speed limit (30km/h)', 
            'Speed limit (50km/h)', 
            'Speed limit (60km/h)', 
            'Speed limit (70km/h)', 
            'Speed limit (80km/h)', 
            'End of speed limit (80km/h)', 
            'Speed limit (100km/h)', 
            'Speed limit (120km/h)', 
            'No passing', 
            'No passing veh over 3.5 tons', 
            'Right-of-way at intersection', 
            'Priority road', 
            'Yield', 
            'Stop', 
            'No vehicles', 
            'Veh > 3.5 tons prohibited', 
            'No entry', 
            'General caution', 
            'Dangerous curve left', 
            'Dangerous curve right', 
            'Double curve', 
            'Bumpy road', 
            'Slippery road', 
            'Road narrows on the right', 
            'Road work', 
            'Traffic signals', 
            'Pedestrians', 
            'Children crossing', 
            'Bicycles crossing', 
            'Beware of ice/snow',
            'Wild animals crossing', 
            'End speed + passing limits', 
            'Turn right ahead', 
            'Turn left ahead', 
            'Ahead only', 
            'Go straight or right', 
            'Go straight or left', 
            'Keep right', 
            'Keep left', 
            'Roundabout mandatory', 
            'End of no passing', 
            'End no passing veh > 3.5 tons' ]
            
model = CustomMobileNetV2(43)
model.load_state_dict(torch.load('model/model.pt', map_location='cpu'))

for param in model.parameters():
    print(param)

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "GET":
        return "Bam :v"

    elif request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
        transform = transforms.Compose([
            transforms.Resize([256, 256]), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])
        

        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        input = transform(img)

        input = input.unsqueeze(0)

        with torch.no_grad():
            model.eval() 
            output = model(input)
            pred = output > 0.5
        index = 0
        for idx, j in enumerate(pred.numpy()[0]):
            if j == True:
                index = idx

        return ({"prediksi": classes[index]})


if __name__ == "__main__":
    app.run(host="localhost", port="5000")
