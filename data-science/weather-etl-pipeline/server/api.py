from flask import Flask, jsonify, request
import requests
import datetime as dt

app = Flask(__name__)

def get_current_weather(city):    
    date = dt.date.today()
    current_weather_url = f"https://api.weatherapi.com/v1/current.json?key=4eb55adc97d8437b936141458242410&q={city}&aqi=yes"
    current_astronomy_url = f"https://api.weatherapi.com/v1/astronomy.json?key=4eb55adc97d8437b936141458242410&q={city}&dt={date}"
    
    try:
        current_weather_response = requests.get(current_weather_url)
        current_astronomy_response = requests.get(current_astronomy_url)

        if current_weather_response.status_code == 200 and current_astronomy_response.status_code == 200:
            current_weather_data = current_weather_response.json()
            current_astronomy_data = current_astronomy_response.json()
            
            return {
                'location': current_weather_data['location']['name'],
                'region': current_weather_data['location']['region'],
                'country': current_weather_data['location']['country'],
                'local_time': current_weather_data['location']['localtime'],
                'temp_f': current_weather_data['current']['temp_f'],
                'condition': current_weather_data['current']['condition']['text'],
                'wind_mph': current_weather_data['current']['wind_mph'],
                'wind_dir': current_weather_data['current']['wind_dir'],
                'uv': current_weather_data['current']['uv'],
                'moon_phase': current_astronomy_data['astronomy']['astro']['moon_phase']
            }
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city')
    if not city:
        return jsonify({'error': 'City parameter is required'}), 400
    
    weather_data = get_current_weather(city)
    if weather_data is None:
        return jsonify({'error': 'Could not fetch weather data'}), 404
    
    return jsonify(weather_data)

if __name__ == '__main__':
    app.run(port=5001, debug=True)