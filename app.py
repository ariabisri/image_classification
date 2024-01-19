from flask import Flask, request, render_template
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor
# , request,
# , render_template, request, redirect, url_for



# base_options = core.BaseOptions(file_name="1.tflite")
# classification_options = processor.ClassificationOptions(max_results=2)
# options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
# classifier = vision.ImageClassifier.create_from_options(options)

# image = vision.TensorImage.create_from_file(image_path)
# classification_result = classifier.classify(image)

# model = load_model('Resnet50.h5')
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions

# from keras.applications.resnet50 import ResNet50

app = Flask (__name__, template_folder="templates")
# model = ResNet50()



@app.route('/')
def hello_world():
    return "Hai"

@app.route('/ml', methods= ["GET", "POST"])
def image_classification():
    if request.method=="POST":
        gambar=request.files['gambar']
        img_path='static/'+gambar.filename
        gambar.save(img_path)
        base_options = core.BaseOptions(file_name="3.tflite")
        classification_options = processor.ClassificationOptions(max_results=2)
        options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
        classifier = vision.ImageClassifier.create_from_options(options)

        image = vision.TensorImage.create_from_file(img_path)
        classification_result = classifier.classify(image)



        # image = load_img(img_path, target_size=(224, 224))
        # image = img_to_array(image)
        # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # image = preprocess_input(image)
        # yhat = model.predict(image)
        # label = decode_predictions(yhat)
        # label = label[0][0]
        # classification = '%s (%.2f%%)' % (label[1], label[2]*100)

        # Initialization
        base_options2 = core.BaseOptions(file_name="2.tflite")
        detection_options2 = processor.DetectionOptions(max_results=2)
        options2 = vision.ObjectDetectorOptions(base_options=base_options2, detection_options=detection_options2)
        detector2 = vision.ObjectDetector.create_from_options(options2)

        # Alternatively, you can create an object detector in the following manner:
        # detector = vision.ObjectDetector.create_from_file(model_path)

        # Run inference
        image2 = vision.TensorImage.create_from_file(img_path)
        detection_result = detector2.detect(image2)
         # detection_result.save('/static/res.jpg')

        return render_template("class.html", cr=classification_result, img_path=img_path, dr=detection_result)
    else:
        return render_template("class.html")