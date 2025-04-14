from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input


# Loading Models,
# covid_model = load_model('models/covid.h5')
braintumor_model = load_model('models/braintumor_densenet121.keras')
alzheimer_model = load_model('models/alzheimer_densenet121.keras')
manyinone_model = load_model('models/combined_densenet121.keras')
pneumonia_model = load_model('models/pneumonia_densenet121.keras')
manyinone_model = load_model('models/manyinone_densenet121.keras')
filename = './models/symptomsbased_logreg.pkl'
with open(filename, 'rb') as file:
    symptomsbased_model = pickle.load(file)
diabetes_model = pickle.load(open('models/diabetes_logreg.pkl', 'rb'))
# heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))

# import pickle


# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

navbar_html = """<nav class="navbar navbar-expand-lg" style="transition: none !important; background-color: #0b5ed7">
        <div class="container-fluid">
          <a style = "margin-left: 2ch; color: white;" class="navbar-brand" href="/">Salubritas</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent"
            aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarSupportedContent">
            <ul class="navbar-nav ms-auto mb-2 mb-lg-0" id="nb-s-c-ul">
              
            </ul>
          </div>
        </div>
    </nav>"""
    
footer = """<footer class='text-light bg-dark position-relative d-flex flex-column align-items-center justify-content-center'>
        <p class='text-center my-0 mx-0 py-2 px-2'>
            <p>This is a very simple attempt at automation of a very complex problem of diagnosis.</p>
            <p>The scope covered is too little and the data used to train isnt as vast as we would have liked it to be.</p>
            <p>The diagnosis shouldn't be taken seriously but can be used as a reference point.</p>
            <p>Thank You!</p>
        </p>
    </footer>"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

############################################# BRAIN TUMOR FUNCTIONS ################################################

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html', navbar_html = navbar_html, footer = footer)





# braintumor_model = None

types = {
    'imagebased': {'braintumor': {'model': braintumor_model, 'type': 'Categorical', 'op_possibilities': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'], 'links':{
    'Glioma': 'https://www.cancer.gov/types/brain/patient/adult-glioma-treatment-pdq',
    'Meningioma': 'https://www.mayoclinic.org/diseases-conditions/meningioma/symptoms-causes/syc-20355643',
    'No Tumor': 'https://my.clevelandclinic.org/health/diseases/21881-tumor',  # General info
    'Pituitary': 'https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/symptoms-causes/syc-20350560'
}}, 'alzheimer': {'model': alzheimer_model, 'type': 'Categorical', 'op_possibilities': ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'], 'links':
    {'Mild Demented':'https://en.wikipedia.org/wiki/Alzheimer%27s_disease', 
     'Moderate Demented':'https://en.wikipedia.org/wiki/Alzheimer%27s_disease', 
     'Non Demented':'https://en.wikipedia.org/wiki/Alzheimer%27s_disease', 
     'Very Mild Demented':'https://en.wikipedia.org/wiki/Alzheimer%27s_disease'}
    }, 'pneumonia': {'model': pneumonia_model, 'type': 'Binary', 'op_possibilities': ['Pneumonia'], 'links': 
        {'Pneumonia': 'https://lunghealth.ca/what-is-pneumonia-and-why-is-it-important-to-know/?gad_source=1&gclid=CjwKCAjw5PK_BhBBEiwAL7GTPRsOvrMjOWz3t5vMdzWV6qw0sT6Nb9-R7ojIa_oaqBO8E-x9LJTSbRoCQW0QAvD_BwE', 'No Pneumonia': 'https://lunghealth.ca/what-is-pneumonia-and-why-is-it-important-to-know/?gad_source=1&gclid=CjwKCAjw5PK_BhBBEiwAL7GTPRsOvrMjOWz3t5vMdzWV6qw0sT6Nb9-R7ojIa_oaqBO8E-x9LJTSbRoCQW0QAvD_BwE'}
        }, 'manyinone': {'model': manyinone_model, 'type': 'Categorical', 'op_possibilities': ['Mild Demented', 'Moderate Demented', 'Normal (Pneumonia)', 'Non Demented', 'Pneumonia', 'Very Mild Demented', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary'], 'links': 
            {'Glioma': 'https://www.cancer.gov/types/brain/patient/adult-glioma-treatment-pdq', 
            'Mild Demented':'https://en.wikipedia.org/wiki/Alzheimer%27s_disease', 
            'Moderate Demented':'https://en.wikipedia.org/wiki/Alzheimer%27s_disease', 
            'Non Demented':'https://en.wikipedia.org/wiki/Alzheimer%27s_disease', 
            'Very Mild Demented':'https://en.wikipedia.org/wiki/Alzheimer%27s_disease', 
            'Meningioma': 'https://www.mayoclinic.org/diseases-conditions/meningioma/symptoms-causes/syc-20355643', 
            'Pneumonia': 'https://lunghealth.ca/what-is-pneumonia-and-why-is-it-important-to-know/?gad_source=1&gclid=CjwKCAjw5PK_BhBBEiwAL7GTPRsOvrMjOWz3t5vMdzWV6qw0sT6Nb9-R7ojIa_oaqBO8E-x9LJTSbRoCQW0QAvD_BwE',   
            'No Tumor': 'https://my.clevelandclinic.org/health/diseases/21881-tumor',  # General info
            'Pituitary': 'https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/symptoms-causes/syc-20350560'}
        },},
    'textbased': {'diabetes': {'model': diabetes_model, 'type': 'Binary', 'op_possibilities': ['Diabetes'], 'links': {'Diabetes': 'https://my.clevelandclinic.org/health/diseases/7104-diabetes', 'No Diabetes': 'https://my.clevelandclinic.org/health/diseases/7104-diabetes'}}, 'heartdisease': '5005',
                'symptomsbased': {'model': symptomsbased_model, 'type': 'Categorical', 'op_possibilities': 
                    ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo'], 'links': {
                    'Fungal infection': 'https://www.cdc.gov/fungal/index.html',
                    'Allergy': 'https://www.mayoclinic.org/diseases-conditions/allergies/symptoms-causes/syc-20351497',
                    'GERD': 'https://www.niddk.nih.gov/health-information/digestive-diseases/acid-reflux-ger-gerd-adults',
                    'Chronic cholestasis': 'https://www.ncbi.nlm.nih.gov/books/NBK459302/',
                    'Drug Reaction': 'https://www.webmd.com/allergies/drug-allergy-symptoms-treatments',
                    'Peptic ulcer diseae': 'https://www.mayoclinic.org/diseases-conditions/peptic-ulcer/symptoms-causes/syc-20354223',
                    'AIDS': 'https://www.cdc.gov/hiv/basics/whatishiv.html',
                    'Diabetes': 'https://www.cdc.gov/diabetes/basics/diabetes.html',
                    'Gastroenteritis': 'https://www.mayoclinic.org/diseases-conditions/viral-gastroenteritis/symptoms-causes/syc-20378847',
                    'Bronchial Asthma': 'https://www.aafa.org/asthma.aspx',
                    'Hypertension': 'https://www.cdc.gov/bloodpressure/about.htm',
                    'Migraine': 'https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201',
                    'Cervical spondylosis': 'https://www.mayoclinic.org/diseases-conditions/cervical-spondylosis/symptoms-causes/syc-20370787',
                    'Paralysis (brain hemorrhage)': 'https://www.stroke.org/en/about-stroke/types-of-stroke/hemorrhagic-strokes-bleeds',
                    'Jaundice': 'https://my.clevelandclinic.org/health/diseases/17842-jaundice',
                    'Malaria': 'https://www.cdc.gov/malaria/about/index.html',
                    'Chicken pox': 'https://www.cdc.gov/chickenpox/about/index.html',
                    'Dengue': 'https://www.cdc.gov/dengue/index.html',
                    'Typhoid': 'https://www.cdc.gov/typhoid-fever/index.html',
                    'hepatitis A': 'https://www.cdc.gov/hepatitis/hav/index.htm',
                    'Hepatitis B': 'https://www.cdc.gov/hepatitis/hbv/index.htm',
                    'Hepatitis C': 'https://www.cdc.gov/hepatitis/hcv/index.htm',
                    'Hepatitis D': 'https://www.cdc.gov/hepatitis/hdv/index.htm',
                    'Hepatitis E': 'https://www.cdc.gov/hepatitis/hev/index.htm',
                    'Alcoholic hepatitis': 'https://www.mayoclinic.org/diseases-conditions/alcoholic-hepatitis/symptoms-causes/syc-20351388',
                    'Tuberculosis': 'https://www.cdc.gov/tb/default.htm',
                    'Common Cold': 'https://www.cdc.gov/dotw/common-cold/index.html',
                    'Pneumonia': 'https://www.lung.org/lung-health-diseases/lung-disease-lookup/pneumonia',
                    'Dimorphic hemmorhoids(piles)': 'https://www.mayoclinic.org/diseases-conditions/hemorrhoids/symptoms-causes/syc-20360268',
                    'Heart attack': 'https://www.heart.org/en/health-topics/heart-attack/about-heart-attacks',
                    'Varicose veins': 'https://www.mayoclinic.org/diseases-conditions/varicose-veins/symptoms-causes/syc-20350643',
                    'Hypothyroidism': 'https://www.niddk.nih.gov/health-information/endocrine-diseases/hypothyroidism',
                    'Hyperthyroidism': 'https://www.niddk.nih.gov/health-information/endocrine-diseases/hyperthyroidism',
                    'Hypoglycemia': 'https://www.cdc.gov/diabetes/basics/low-blood-sugar.html',
                    'Osteoarthristis': 'https://www.arthritis.org/diseases/osteoarthritis',
                    'Arthritis': 'https://www.cdc.gov/arthritis/basics/index.html',
                    '(vertigo) Paroymsal  Positional Vertigo': 'https://www.hopkinsmedicine.org/health/conditions-and-diseases/benign-paroxysmal-positional-vertigo-bppv',
                    'Acne': 'https://www.aad.org/public/diseases/acne/what-is',
                    'Urinary tract infection': 'https://www.mayoclinic.org/diseases-conditions/urinary-tract-infection/symptoms-causes/syc-20353447',
                    'Psoriasis': 'https://www.psoriasis.org/about-psoriasis/',
                    'Impetigo': 'https://www.cdc.gov/groupastrep/diseases-public/impetigo.html'
                }
            }
        }
    }


import requests
from diabetes import normalize

def collapse(obj):
    if type(obj)==str:
        return obj
    else:
        return collapse(obj[0])


@app.route('/predict/<diag_type>', methods = ['GET', 'POST'])
def predict(diag_type):
    print(diag_type + " "  )
    print(request.method)
    diag_type = diag_type.replace(" ", "").lower()
    if request.method == 'GET':
        print('get')
        return render_template(f'{diag_type} copy.html', navbar_html = navbar_html, active = diag_type, footer = footer)
    else:        
        if diag_type in types['imagebased']:   
            dt_data = types['imagebased'][diag_type]
            model = dt_data['model']
            possibilities = dt_data['op_possibilities']
            type = dt_data['type']
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            # cv2.imshow(img)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = model.predict(img)
            print(pred)
            index = 0
            if type=='Categorical':
                print('cat')
                index = list(pred[0]).index(np.max(pred[0]))
                res = possibilities[index]
            elif type == 'Binary':
                index = 0
                print('checking')
                if pred>0.5:
                    res = possibilities[0]
                else:
                    res = "No " + possibilities[0]
            c_res = collapse(res)
            print(c_res)
            link = dt_data['links'][c_res]
            return render_template('prediction.html', res = res, link= link, filename = filename, footer = footer, navbar = navbar_html)
        else:
            dt_data = types['textbased'][diag_type]
            model = dt_data['model']
            type = dt_data['type']
            possibilities = dt_data['op_possibilities']
            print(request.form)
            if diag_type == 'symptomsbased':
                strinp = request.form['list']
                print(strinp)
                inp = []
                for i in range(len(strinp)):
                    inp.append(int(strinp[i]))
                params = []
                params.append(inp)
                features = np.array(params)
                # print(model.__dict__())
                pred = model.predict(features)
            elif diag_type=='diabetes':
                inp = normalize(request.form)
                x = []
                i = []
                for val in inp.values():
                    i.append(val)
                x.append(i)
                pred = model.predict(x)
                
            if type=='Categorical':
                res = pred
            else:
                print(pred)
                if pred<0.5:
                    res = possibilities[0]
                else:
                    res = "No " + possibilities[0]            
            
            c_res = collapse(res)
            link = dt_data['links'][c_res]
            return render_template('prediction.html', res=c_res, link = link, footer = footer, navbar = navbar_html)

        


    




########################### Result Functions ########################################


# @app.route('/resultc', methods=['POST'])
# def resultc():
#     if request.method == 'POST':
#         firstname = request.form['firstname']
#         lastname = request.form['lastname']
#         email = request.form['email']
#         phone = request.form['phone']
#         gender = request.form['gender']
#         age = request.form['age']
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             flash('Image successfully uploaded and displayed below')
#             img = cv2.imread('static/uploads/'+filename)
#             img = cv2.resize(img, (224, 224))
#             img = img.reshape(1, 224, 224, 3)
#             img = img/255.0
#             pred = covid_model.predict(img)
#             if pred < 0.5:
#                 pred = 0
#             else:
#                 pred = 1
#             # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
#             return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

#         else:
#             flash('Allowed image types are - png, jpg, jpeg')
#             return redirect(request.url)


@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = crop_imgs([img])
            img = img.reshape(img.shape[1:])
            img = preprocess_imgs([img], (224, 224))
            pred = braintumor_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Brain Tumor test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
            return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        pred = diabetes_model.predict(
            [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        cpm = request.form['concave_points_mean']
        am = request.form['area_mean']
        rm = request.form['radius_mean']
        pm = request.form['perimeter_mean']
        cm = request.form['concavity_mean']
        pred = breastcancer_model.predict(
            np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Breast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resulta', methods=['GET', 'POST'])
def resulta():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3)
            img = img/255.0
            pred = alzheimer_model.predict(img)
            pred = pred[0].argmax()
            print(pred)
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resulta.html', filename=filename, fn=firstname, ln=lastname, age=age, r=0, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')


@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultp.html', filename=filename, r=pred)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
