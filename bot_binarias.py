import yfinance as yf
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import datetime

def get_forex_data(symbol, period='365d'):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    df.columns = [col.lower() for col in df.columns]
    return df

def get_stock_data(symbol, period='365d'):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    df.columns = [col.lower() for col in df.columns]
    return df

def get_crypto_data(symbol, exchange='binance', timeframe='1d', limit=365):
    ex = getattr(ccxt, exchange)()
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        st.warning(f"Error fetching crypto data for {symbol}: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    df = df.copy()
    if 'close' not in df.columns:
        st.error("ERROR: DataFrame no tiene columna 'close', no se pueden calcular indicadores.")
        return pd.DataFrame()
    try:
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['sma50'] = ta.sma(df['close'], length=50)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)
    except Exception as e:
        st.error(f"Error al calcular indicadores técnicos: {e}")
        return pd.DataFrame()
    return df

def prepare_data(df):
    df = df.copy()
    features = df[['close', 'rsi', 'sma50']]
    target = df['target']
    return features, target

def train_and_predict(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, preds)
    return model, acc

def mostrar_resumen(sym, df):
    try:
        if df.empty or len(df) < 60:
            st.info(f"No hay datos suficientes para {sym}")
            return

        X, y = prepare_data(df)
        if X.empty or y.empty:
            st.info(f"No hay datos de características o etiquetas para {sym}")
            return

        model, accuracy = train_and_predict(X, y)

        last_row = df.iloc[-1]
        last_features = pd.DataFrame([last_row[['close', 'rsi', 'sma50']].values], columns=['close', 'rsi', 'sma50'])

        pred_proba = model.predict_proba(last_features)[0,1]
        rsi_val = last_row['rsi']
        sma50_val = last_row['sma50']
        price = last_row['close']

        if pred_proba > 0.6 and rsi_val < 70:
            accion = "📈 Comprar (CALL)"
            color = "🟢"
        elif pred_proba < 0.4 and rsi_val > 30:
            accion = "📉 Vender (PUT)"
            color = "🔴"
        else:
            accion = "⏸️ Esperar"
            color = "🟡"

        with st.expander(f"{sym} | Recomendación: {color} {accion}"):
            st.subheader(sym)
            st.metric(label="Precio Actual", value=f"{price:.4f}")
            st.metric(label="RSI", value=f"{rsi_val:.1f}")
            st.metric(label="SMA50", value=f"{sma50_val:.4f}")
            st.metric(label="Precisión Modelo", value=f"{accuracy*100:.2f}%")

            st.write(f"### Probabilidad de que suba: {pred_proba:.2f}")
            st.write(f"### Acción sugerida: {color} {accion}")

            st.write(f"📅 Fecha última vela: {last_row.name.date()}")
            st.write(f"📊 Volumen: {last_row['volume']:.2f}")
            st.write(f"🧠 Última predicción: {'Sube 📈' if last_row['target'] == 1 else 'Baja 📉'}")
            st.write(f"📉 Riesgo sugerido: SL=2%, TP=4%")

            monto = st.number_input(f"Ingresar monto para operar en {sym}", value=10.0, step=1.0, key=f"monto_{sym}")
            posible_ganancia = monto * 0.7
            st.success(f"Si aciertas: ganas ${posible_ganancia:.2f}")

            st.text_area("📝 Comentarios personalizados", placeholder="Escribe tus ideas o notas aquí...", key=f"coment_{sym}")

            st.line_chart(df[['close', 'sma50']].tail(100))

        st.markdown("---")

    except Exception as e:
        st.error(f"Error procesando {sym}: {e}")

def main():
    st.set_page_config(page_title="Bot Trading con IA", layout="wide")
    st.title("📊 Bot de Trading con Predicción y Recomendaciones")

    forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ARB/USDT']
    stock_symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA']

    if "data" not in st.session_state:
        st.session_state.data = {}
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = None

    def cargar_datos():
        data = {}
        with st.spinner("📥 Descargando y procesando datos..."):
            for sym in forex_symbols:
                df = get_forex_data(sym)
                df = add_technical_indicators(df)
                data[sym] = df

            for sym in crypto_symbols:
                df = get_crypto_data(sym)
                df = add_technical_indicators(df)
                data[sym] = df

            for sym in stock_symbols:
                df = get_stock_data(sym)
                df = add_technical_indicators(df)
                data[sym] = df

        st.session_state.data = data
        st.session_state.last_update_time = datetime.datetime.now()

    if st.button("🔄 Actualizar datos ahora"):
        cargar_datos()

    if not st.session_state.data:
        cargar_datos()

    if st.session_state.last_update_time is not None:
        tiempo_transcurrido = datetime.datetime.now() - st.session_state.last_update_time
        minutos = int(tiempo_transcurrido.total_seconds() // 60)
        segundos = int(tiempo_transcurrido.total_seconds() % 60)
        st.info(f"Última actualización: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')} | Hace {minutos} min {segundos} seg")

    tabs_main = st.tabs(["💱 Forex", "💰 Criptomonedas", "📈 Acciones (USA)"])

    with tabs_main[0]:
        st.subheader("Forex")
        for sym in forex_symbols:
            if sym in st.session_state.data and not st.session_state.data[sym].empty:
                mostrar_resumen(sym, st.session_state.data[sym])
            else:
                st.info(f"No hay datos para {sym}")

    with tabs_main[1]:
        st.subheader("Criptomonedas")
        for sym in crypto_symbols:
            if sym in st.session_state.data and not st.session_state.data[sym].empty:
                mostrar_resumen(sym, st.session_state.data[sym])
            else:
                st.info(f"No hay datos para {sym}")

    with tabs_main[2]:
        st.subheader("Acciones (USA)")
        for sym in stock_symbols:
            if sym in st.session_state.data and not st.session_state.data[sym].empty:
                mostrar_resumen(sym, st.session_state.data[sym])
            else:
                st.info(f"No hay datos para {sym}")

if __name__ == "__main__":
    main()
