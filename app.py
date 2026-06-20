from fastapi import FastAPI , UploadFile , File 
from PIL  import Image 
import torch 
import torch.nn as nn 
from torchvision import transforms ,models

app = FastAPI()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = models.resnet18(weights = None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load('best_model.pth',map_location=device)

)
model.eval()
model.to(device)
classes = ['NORMAL','PNEUMONIA']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
def home():
    return {'message':'Pneumonia detection api running'}
@app.post('/predict')
async def preidct(file:UploadFile = File(...)):
    image = Image.open(file.file).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs,dim=1)
        confidence ,pred = torch.max(probs,1)


        return {
            'Prediction':classes[pred.item()],
            'confidence': round(confidence.item()*100,2)
        }



