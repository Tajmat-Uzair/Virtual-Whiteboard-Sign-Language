import tkinter as tk
from hand_recognition_module import HandRecognitionModule

def recognize_signs():
    mod = HandRecognitionModule()
    mod.recognize_signs()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sign Language Recognition")

    button = tk.Button(root,text="Signs",command=recognize_signs)
    button.pack(pady=20)

    root.mainloop()

#gui for sign language 