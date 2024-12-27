import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timezone
import time
import certifi
import websocket

# Variables globales
app_id = '64652'
token = '6fUPldSoQiIrtjt'
symbol = 'R_100'
amount = 1
ticks_data = []  # Almacena los ticks recibidos
candles = []  # Lista para almacenar las velas generadas
contract_open = False  # Controla si hay un contrato abierto
model = None
trade_history = []
consecutive_losses = 0  # Contador de pérdidas consecutivas

# Escalador para normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))

def on_open(ws):
    print("Conexión abierta.")
    authorize_message = {
        "authorize": token
    }
    ws.send(json.dumps(authorize_message))

def on_message(ws, message):
    global contract_open, ticks_data, amount, consecutive_losses
    data = json.loads(message)

    if 'error' in data.keys():
        print('Error:', data['error']['message'])

    elif data.get("msg_type") == "authorize":
        subscribe_to_candles(ws)

    elif data.get("msg_type") == "candles":
        process_candles(ws, data['candles'])

    elif data.get("msg_type") == "tick":
        tick = data['tick']
        ticks_data.append(tick)
        process_ticks(ws)

    elif data.get("msg_type") == "buy":
        contract_id = data['buy']['contract_id']
        contract_open = True  # Marcar contrato como abierto
        print(f"Operación ejecutada. ID del contrato: {contract_id}")
        subscribe_to_contract(ws, contract_id)

    elif data.get("msg_type") == "proposal_open_contract":
        if data['proposal_open_contract']['is_sold']:
            profit = data['proposal_open_contract']['profit']
            trade_history.append(profit)
            if profit > 0:
                print(f"El contrato ha sido vendido. Ganancia: {profit}")
                amount = 1
                consecutive_losses = 0  # Reiniciar el contador de pérdidas consecutivas
            elif profit < 0:
                print("El contrato perdió.")
                amount = amount * 2
                consecutive_losses += 1  # Incrementar el contador de pérdidas consecutivas
            else:
                print("El contrato terminó en empate.")
            print("El contrato ha finalizado. Buscando una nueva señal...")
            contract_open = False  # Contrato finalizado, se puede abrir otro
            process_ticks(ws)

def subscribe_to_candles(ws):
    candles_message = {
        "ticks_history": symbol,
        "end": "latest",
        "style": "candles",
        "count": 450,
        "granularity": 300  # 1 vela por minuto
    }
    ws.send(json.dumps(candles_message))

def process_candles(ws, received_candles):
    global candles
    for candle in received_candles:
        timestamp = datetime.fromtimestamp(candle['epoch'], timezone.utc)
        new_candle = {
            'timestamp': timestamp,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close']
        }
        candles.append(new_candle)
    subscribe_to_ticks(ws)

def subscribe_to_ticks(ws):
    ticks_message = {
        "ticks": symbol,
        "subscribe": 1
    }
    ws.send(json.dumps(ticks_message))
    print("Suscripción a ticks enviada.")

def process_ticks(ws):
    global candles

    # Crear velas manualmente a partir de los ticks recibidos
    if len(ticks_data) > 0:
        tick_time = datetime.fromtimestamp(ticks_data[-1]['epoch'], timezone.utc)
        tick_close = ticks_data[-1]['quote']

        if len(candles) > 0 and candles[-1]['timestamp'].minute == tick_time.minute:
            candles[-1]['close'] = tick_close
            candles[-1]['high'] = max(candles[-1]['high'], tick_close)
            candles[-1]['low'] = min(candles[-1]['low'], tick_close)
        else:
            new_candle = {
                'timestamp': tick_time,
                'open': tick_close,
                'high': tick_close,
                'low': tick_close,
                'close': tick_close
            }
            candles.append(new_candle)

        if len(candles) > 90:
            candles = candles[-90:]

        analyze_market(ws)

def train_model():
    global model
    df = pd.DataFrame(candles)
    df['returns'] = df['close'].pct_change()
    df['sma'] = df['close'].rolling(window=12).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['jaw'], df['teeth'], df['lips'] = calculate_alligator(df)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df.dropna(inplace=True)


    features = df[['returns', 'sma', 'rsi', 'jaw', 'teeth', 'lips', 'macd', 'macd_signal', 'macd_hist']]
    features_scaled = scaler.fit_transform(features)


    X, y = [], []
    for i in range(60, len(features_scaled)):
        X.append(features_scaled[i-60:i]) 
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0)  # 1 si sube, 0 si baja

    X = np.array(X)
    y = np.array(y)

    # Definir el modelo LSTM
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))  # Salida binaria (1 si sube, 0 si baja)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X, y, epochs=50, batch_size=32)  # Aumentar el número de épocas
    print("Modelo entrenado.")

def analyze_market(ws):
    global candles, contract_open, amount, model, consecutive_losses
    print("Analizando velas...")

    if len(candles) < 90:
        print("No hay suficientes velas para realizar el análisis.")
        return

    if model is None:
        train_model()

    df = pd.DataFrame(candles)
    df['returns'] = df['close'].pct_change()
    df['sma'] = df['close'].rolling(window=12).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['jaw'], df['teeth'], df['lips'] = calculate_alligator(df)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df.dropna(inplace=True)

    
    features = df[['returns', 'sma', 'rsi', 'jaw', 'teeth', 'lips', 'macd', 'macd_signal', 'macd_hist']].iloc[-60:]
    features_scaled = scaler.transform(features)

    X = np.array([features_scaled])
    prediction = model.predict(X)[0][0]

    if not contract_open:

        last_candle = candles[-1]
        is_red_candle = last_candle['close'] < last_candle['open']  # Vela roja
        is_green_candle = last_candle['close'] > last_candle['open']  # Vela verde

        if consecutive_losses < 5:
            if is_green_candle and (df['lips'].iloc[-1] > df['jaw'].iloc[-1] and df['teeth'].iloc[-1] > df['jaw'].iloc[-1] and
                df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and
                df['rsi'].iloc[-1] > 70): 
                print("Señal de compra detectada. Ejecutando operación Rise.")
                execute_rise_trade(ws)
            elif is_red_candle and (df['lips'].iloc[-1] < df['jaw'].iloc[-1] and df['teeth'].iloc[-1] < df['jaw'].iloc[-1] and
                  df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and
                  df['rsi'].iloc[-1] < 30):  
                print("Señal de venta detectada. Ejecutando operación Fall.")
                execute_fall_trade(ws)
        else:
            print("Dos pérdidas consecutivas. Analizando más a fondo el mercado antes de realizar una nueva operación.")
        

def execute_rise_trade(ws):
    global contract_id
    rise_trade_message = {
        "buy": 1,
        "subscribe": 1,
        "price": 1000,
        "parameters": {
            "amount": amount,
            "basis": "stake",
            "contract_type": "CALL",
            "currency": "USD",
            "duration": 2,
            "duration_unit": "m",
            "symbol": symbol
        }
    }
    ws.send(json.dumps(rise_trade_message))
    print("Operación Rise enviada. Esperando confirmación...")

def execute_fall_trade(ws):
    global contract_id
    fall_trade_message = {
        "buy": 1,
        "subscribe": 1,
        "price": 10000,
        "parameters": {
            "amount": amount,
            "basis": "stake",
            "contract_type": "PUT",
            "currency": "USD",
            "duration": 2,
            "duration_unit": "m",
            "symbol": symbol
        }
    }
    ws.send(json.dumps(fall_trade_message))
    print("Operación Fall enviada. Esperando confirmación...")

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_alligator(df):
    jaw = df['close'].rolling(window=8).mean().shift(3)
    teeth = df['close'].rolling(window=2).mean().shift(2)
    lips = df['close'].rolling(window=2).mean().shift(1)
    return jaw, teeth, lips

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def on_error(ws, error):
    print("Error en WebSocket:", error)

def on_close(ws, close_status_code, close_msg):
    print("Conexión cerrada. Intentando reconectar...")
    time.sleep(10)
    ws.run_forever()

def subscribe_to_contract(ws, contract_id):
    contract_message = {
        "proposal_open_contract": 1,
        "contract_id": contract_id
    }
    ws.send(json.dumps(contract_message))

ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=" + app_id
ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
ws.run_forever(sslopt={"ca_certs": certifi.where()})