import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from keras.models import load_model

# Load the trained model
model = load_model('digit_recognition_model.keras')

class DigitRecognizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        
        # Create canvas for drawing
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white', cursor="cross")
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W, columnspan=2)
        
        # Create buttons
        self.label = tk.Label(self, text="Draw a digit on the canvas", font=("Helvetica", 14))
        self.label.grid(row=1, column=0, columnspan=2)
        
        self.clear_button = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0)
        
        self.recognize_button = tk.Button(self, text="Recognize", command=self.predict_digit)
        self.recognize_button.grid(row=2, column=1)
        
        # Bind the mouse events to draw on the canvas
        self.canvas.bind("<B1-Motion>", self.paint)
        self.drawing = False
        
        # Create a blank image for drawing
        self.image = Image.new('L', (280, 280), 255)  # 'L' mode for grayscale, white background
        self.draw = ImageDraw.Draw(self.image)
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image.paste(255, [0, 0, 280, 280])  # Clear image to white

    def paint(self, event):
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')
        
    def predict_digit(self):
        # Convert the image to 28x28 pixels for the model
        img_resized = self.image.resize((28, 28))
        img_resized = ImageOps.invert(img_resized)  # Invert image colors
        img_array = np.array(img_resized)  # Convert to numpy array
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
        img_array = img_array.astype('float32')
        img_array /= 255.0  # Normalize
        
        # Predict the digit
        pred = model.predict(img_array)
        digit = np.argmax(pred)
        
        # Update label with the prediction
        self.label.configure(text=f"Prediction: {digit}")

# Run the application
app = DigitRecognizerApp()
app.mainloop()
