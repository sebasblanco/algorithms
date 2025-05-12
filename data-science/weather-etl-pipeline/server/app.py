from flask import Flask, render_template, request, redirect, flash, session
import requests
import datetime as dt
import sqlite3

app = Flask(__name__, template_folder="../client/templates", static_folder="../static")
app.config['SECRET_KEY'] = 'your_very_secret_key'

API_BASE_URL = 'http://localhost:5001/api'

# extract
def get_weather_from_api(city):
    try:
        response = requests.get(f'{API_BASE_URL}/weather', params={'city': city})
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        return None

# trasnform 
def transform_weather_data(raw_data):
    if not raw_data:
        return None
    transformed_data = {
        'location': raw_data.get('location'),
        'region': raw_data.get('region'),
        'country': raw_data.get('country'),
        'local_time': raw_data.get('local_time'),
        'temp_f': raw_data.get('temp_f'),
        'condition': raw_data.get('condition'),
        'wind_mph': raw_data.get('wind_mph'),
        'wind_dir': raw_data.get('wind_dir'),
        'uv': raw_data.get('uv'),
        'moon_phase': raw_data.get('moon_phase'),
    }
    return transformed_data

# load
def load_weather_data_to_db(data):
    if not data:
        return
    conn = sqlite3.connect('weather_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT,
            region TEXT,
            country TEXT,
            local_time TEXT,
            temp_f REAL,
            condition TEXT,
            wind_mph REAL,
            wind_dir TEXT,
            uv REAL,
            moon_phase TEXT
        )
    ''')
    cursor.execute('''
        INSERT INTO weather (location, region, country, local_time, temp_f, condition, wind_mph, wind_dir, uv, moon_phase)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['location'], data['region'], data['country'], data['local_time'],
        data['temp_f'], data['condition'], data['wind_mph'],
        data['wind_dir'], data['uv'], data['moon_phase']
    ))
    conn.commit()
    conn.close()

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        city = request.form['City']
        if not city:
            flash("Please enter a city name.")
            return render_template('index.html')
        try:
            data = get_weather_from_api(city)
            if data is None:
                flash("Invalid city")
                return render_template('index.html')
            session['city'] = city
            return redirect('/weather')
        except Exception as e:
            print(f"Error connecting to API: {str(e)}")
            return redirect('/')
    return render_template('index.html')

@app.route('/weather', methods=['GET']) 
def weather():
    city = session.get('city')
    if not city:
            flash("Please enter a city name.")
            return redirect('/')
    try:
        raw_data = get_weather_from_api(city)
        if raw_data is None:
            flash("Unable to fetch weather data. Please try again.")
            return redirect('/')
        transformed_data = transform_weather_data(raw_data)
        if transformed_data is None:
            flash("Error processing weather data.")
            return redirect('/')
        load_weather_data_to_db(transformed_data)
        return render_template('index.html', weather=transformed_data)
    except Exception as e:
            print(f"Error connecting to API: {str(e)}")
            return redirect('/')
    
if __name__ == '__main__':
   app.run(port=5000, debug=True)