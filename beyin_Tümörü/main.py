from flask import Flask, request, render_template,flash
import tensorflow as tf
import cv2
import numpy as np
import keras
from keras.models import load_model
import pickle
import pandas as pd
import openai
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

openai.api_key=os.getenv("openai.api_key")

model = load_model('beyin_tumor_model.keras')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send_email', methods=['POST'])
def send_email():
    if request.method == 'POST':
        name = request.form['ad']
        email = request.form['email']
        message = request.form['mesaj']

        # E-posta gönderme işlemi
        msg = EmailMessage()
        msg.set_content(f"Name: {name}\nEmail: {email}\nMessage: {message}")

        msg['Subject'] = 'Beyin Tümörü Tahmin Projesi'
        msg['From'] = email
        msg['To'] = os.getenv("mail")

        try:
            smtp = smtplib.SMTP('smtp.gmail.com', 587)
            smtp.starttls()
            smtp.login(os.getenv("mail"), os.getenv("sıfre"))
            smtp.sendmail(email, os.getenv("mail"), msg.as_string())
            smtp.quit()

            return render_template('iletişim.html')

        except Exception as e:
            return f'E-posta gönderme hatası: {str(e)}'

@app.route('/beyin-tumor')
def beyin_tumor():
    return render_template('beyin_tumor_bilgi.html')

@app.route('/iletisim')
def iletisim():
    return render_template('iletişim.html')

@app.route('/hakkımızda')
def hakkımızda():
    return render_template('hakkımızda.html')

@app.route('/beyin-tumor-tahmin', methods=['GET', 'POST'])
def beyin_tumor_tahmin():
    if request.method == 'POST':

        uploaded_image = request.files['file']
        if uploaded_image:
            image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (256, 256))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)
            predictions = model.predict(image)
            class_index = np.argmax(predictions)
            class_labels = {
                0: "Glioma Tümör",
                1: "Meningioma Tümör",
                2: "Tümör Değil",
                3: "Pituitary Tümör"
            }
            predicted_class = class_labels[class_index]

            image_path = 'static/uploaded_image.jpg'
            cv2.imwrite(image_path, cv2.cvtColor(image[0] * 255, cv2.COLOR_BGR2RGB))
            Beyin_chatbot = "Mr görüntünüzden tümörün olmadığını tespit etsekte yinede bir uzmanla görüşmenizde fayda var."
            if class_index != 2:
                completion = openai.completions.create(
                    model="text-davinci-003",
                    prompt="Sen beyin ve sinir hastalıkları bölümünde çalışan uzman bir doktorsun. Ve sana sorduğum soruları uzmanlığını kullanarak ayrıntılı bir şekilde  yanıtlamanı istiyorum. "+predicted_class+" hastalığına sahip bir hastanın semptomları ve belirtiler nelerdir ? Sahip olduğu tümörün özellikleri nedir ?"+predicted_class+"ün Tanı yöntemleri nedir ? Hastalığın ilerleyişi ve takip düreci nasıl olmalıdır ? Bu hastanın yaşam tarzı değişikliklerin nedir ? Ve nasıl bir tedavi süreci planlanması gerektiğini bilimsel altyapısı ile  ayrıntılı olarak söylermisin ?",
                    max_tokens=2000,
                    temperature=0.5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                Beyin_chatbot = completion.choices[0].text.strip()
            return render_template('result.html', predicted_class=predicted_class, chatbot=Beyin_chatbot)

    return render_template('beyin_tumor_tahmin.html')

if __name__ == '__main__':
    app.run(debug=True)