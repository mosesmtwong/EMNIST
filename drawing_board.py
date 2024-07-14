import tkinter as tk
import torch
from torch import nn
from torchvision.transforms import v2
from PIL import Image, ImageOps

device = "cuda" if torch.cuda.is_available() else "cpu"

root = tk.Tk()
root.title("Drawing_board")

label = tk.Label(root, text="Prediction: -", font=("Arial", 20))
label.pack(side=tk.TOP)

canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.pack(side=tk.TOP)


drawing = False
last_x, last_y = 0, 0

mapping = {
    0: 48,
    1: 49,
    2: 50,
    3: 51,
    4: 52,
    5: 53,
    6: 54,
    7: 55,
    8: 56,
    9: 57,
    10: 65,
    11: 66,
    12: 67,
    13: 68,
    14: 69,
    15: 70,
    16: 71,
    17: 72,
    18: 73,
    19: 74,
    20: 75,
    21: 76,
    22: 77,
    23: 78,
    24: 79,
    25: 80,
    26: 81,
    27: 82,
    28: 83,
    29: 84,
    30: 85,
    31: 86,
    32: 87,
    33: 88,
    34: 89,
    35: 90,
    36: 97,
    37: 98,
    38: 100,
    39: 101,
    40: 102,
    41: 103,
    42: 104,
    43: 110,
    44: 113,
    45: 114,
    46: 116,
}


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 47),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.CNN(x)


def start_draw(event):
    global drawing
    drawing = True


def draw(event):
    global drawing
    r = 15
    if drawing:
        canvas.create_oval(
            event.x - r,
            event.y - r,
            event.x + r,
            event.y + r,
            fill="black",
            outline="black",
        )


def stop_draw(event):
    global drawing
    drawing = False


def predict():
    canvas.postscript(file=r"temp/drawing.eps", colormode="color")
    im = Image.open(r"temp/drawing.eps")
    im = im.convert("L").resize((28, 28))
    im = ImageOps.invert(im)
    im.save(r"temp/drawing.png")

    model = NeuralNetwork()
    model.load_state_dict(torch.load(r"models/EMNIST/EMNIST_201epoch.pth"))
    model.to(device)
    model.eval()

    transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.1736,), (0.3248,)),
        ]
    )

    im_tensor = transform(im)
    im_tensor.unsqueeze_(0)
    im_tensor = im_tensor.to(device)
    pred = int(model(im_tensor).argmax(1)[0])
    pred = chr(mapping[pred])

    print(f"Prediction: {pred}")
    label.config(text=f"Prediction: {pred}")


def clear():
    canvas.delete("all")


button = tk.Button(root, text="Predict", command=predict)
button.pack(side=tk.BOTTOM)

button2 = tk.Button(root, text="Clear", command=clear)
button2.pack(side=tk.BOTTOM)


canvas.bind("<ButtonPress-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)


root.mainloop()
