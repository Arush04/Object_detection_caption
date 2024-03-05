import os
import csv
import json
from flask import Flask, render_template, request
from transformers import pipeline
image_to_text = image_to_text = pipeline("image-to-text", model="./hugging_face")


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        files = request.files.getlist('file')  # Get list of uploaded files

        # If no files are uploaded
        if len(files) == 0:
            return "No selected file"

        json_data = []  # Store JSON data for all files
        for file in files:
            # Save the uploaded file to the uploads folder
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            # Run the detection command to save CSV file
            command = "python detect.py --weights yolov8n.pt --img 640 --conf 0.25 --source " + file_path + " --save-txt --save-csv"
            os.system(command)

            caption = image_to_text(file_path)

            # Convert CSV to JSON
            csv_file_path = os.path.join("uploads", os.path.splitext(file.filename)[0] + ".csv")
            with open(csv_file_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    json_data.append(json.dumps(row))

        # Return detection completion message and JSON data
        return render_template('result.html', caption=caption, json_data=json.dumps(json_data), image_names=[file.filename for file in files])

if __name__ == "__main__":
    app.run(debug=True)