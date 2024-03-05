import os
import csv
import json
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
image_to_text = image_to_text = pipeline("image-to-text", model="./hugging_face")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        # If the user does not select a file, the browser may also
        # submit an empty part without a filename
        if file.filename == '':
            return "No selected file"

        # Save the uploaded file to the uploads folder
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Run the detection command to save CSV file
        command = "python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source " + file_path + " --save-txt --save-csv"
        os.system(command)

        caption = image_to_text(file_path)

        print(file.filename)
        # Convert CSV to JSON
        json_data = []
        csv_file_path = os.path.join("uploads", os.path.splitext(file.filename)[0] + ".csv")
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                json_data.append(row)

        # Return detection completion message and JSON data
        return render_template('result.html', caption=caption,json_data=json.dumps(json_data), image_name=file.filename)

if __name__ == "__main__":
    app.run(debug=True)