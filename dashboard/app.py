import torch
import torch.nn as nn
from PIL import Image
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import base64
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)  # Use the same architecture you trained
num_classes = 3  # Modify this as per your dataset
model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.fc.in_features, num_classes)
)

checkpoint = torch.load("models/resnet18_fitness_net.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
classes = checkpoint['classes']  # ['jogging', 'yoga', 'cycling']
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Function to make predictions
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
    return classes[predicted.item()], confidence[predicted.item()].item()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Fitness Activity Classifier", style={"textAlign": "center"}),

    # Image upload section
    dcc.Upload(
        id="upload-image",
        children=html.Div(["Drag and Drop or ", html.A("Select an Image")]),
        style={
            "width": "50%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "auto"
        },
        accept="image/*"
    ),

    html.Div(id="output-image-upload", style={"textAlign": "center", "marginTop": 20}),
    html.H2(id="prediction-result", style={"textAlign": "center", "marginTop": 20})
])

@app.callback(
    [Output("output-image-upload", "children"),
     Output("prediction-result", "children")],
    [Input("upload-image", "contents")]
)
def update_output(contents):
    if contents is not None:
        # Decode uploaded image
        content_type, content_string = contents.split(",")
        decoded_image = base64.b64decode(content_string)
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(decoded_image)

        predicted_class, confidence = predict_image(image_path)

        return html.Img(src=contents, style={"width": "50%"}), \
            f"Predicted Activity: {predicted_class} (Confidence: {confidence:.2f})"

    return None, "No image uploaded yet!"


if __name__ == '__main__':
    app.run_server(debug=True)
