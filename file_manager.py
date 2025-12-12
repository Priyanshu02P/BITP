import tkinter as tk
from tkinter import filedialog

def select_file():
    # Create a hidden root window (we don't want the full Tkinter GUI)
    root = tk.Tk()
    root.withdraw()

    # Open file selection dialog
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=(("All files", "*.*"), ("Image files", "*.jpg;*.png;*.jpeg"))
    )

    if file_path:
        print(f"✅ File selected:\n{file_path}")
        return file_path
    else:
        print("❌ No file selected.")
        return None