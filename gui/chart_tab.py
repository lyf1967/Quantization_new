import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 TensorFlow 警告
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QSpinBox, QTextEdit, QFileDialog, QFrame, QLineEdit
from PyQt5.QtCore import Qt
from mt5.mt5_handler import MT5Handler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from io import BytesIO
import threading

class ChartTab(QWidget):
    def __init__(self, mt5_handler: MT5Handler):
        super().__init__()
        self.mt5_handler = mt5_handler
        self.scaler = MinMaxScaler()
        self.model = None
        self.trading_thread = None
        self.stop_trading_flag = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.setStyleSheet("""
            QWidget {
                background-color: #252537;
                color: #cdd6f4;
            }
            QLabel {
                font-size: 14px;
            }
            QComboBox, QSpinBox, QTextEdit, QLineEdit {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 5px;
            }
            QPushButton {
                background-color: #45475a;
                color: #89b4fa;
                border: 1px solid #585b70;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
        """)

        # LSTM 训练区
        training_frame = QFrame()
        training_layout = QVBoxLayout()
        training_frame.setLayout(training_layout)

        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("交易品种："))
        self.symbol_combo = QComboBox()
        symbol_layout.addWidget(self.symbol_combo)
        training_layout.addLayout(symbol_layout)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("时间范围（天）："))
        self.days = QSpinBox()
        self.days.setRange(1, 365)
        self.days.setValue(30)
        params_layout.addWidget(self.days)
        params_layout.addWidget(QLabel("隐藏层单元数："))
        self.units = QSpinBox()
        self.units.setRange(10, 200)
        self.units.setValue(50)
        params_layout.addWidget(self.units)
        params_layout.addWidget(QLabel("训练轮数："))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(10)
        params_layout.addWidget(self.epochs)
        training_layout.addLayout(params_layout)

        train_button_layout = QHBoxLayout()
        self.train_button = QPushButton("开始训练")
        self.train_button.clicked.connect(self.start_training)
        self.save_model_button = QPushButton("保存模型")
        self.save_model_button.clicked.connect(self.save_model)
        self.save_model_button.setEnabled(False)
        train_button_layout.addWidget(self.train_button)
        train_button_layout.addWidget(self.save_model_button)
        training_layout.addLayout(train_button_layout)

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMinimumHeight(150)
        training_layout.addWidget(self.training_log)

        # AI 交易区
        ai_frame = QFrame()
        ai_layout = QVBoxLayout()
        ai_frame.setLayout(ai_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("加载模型："))
        self.model_path = QLineEdit()
        self.model_path.setReadOnly(True)
        self.browse_model_button = QPushButton("浏览")
        self.browse_model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.browse_model_button)
        ai_layout.addLayout(model_layout)

        ai_control_layout = QHBoxLayout()
        self.start_ai_button = QPushButton("开始AI交易")
        self.start_ai_button.clicked.connect(self.start_ai_trading)
        self.stop_ai_button = QPushButton("停止AI交易")
        self.stop_ai_button.clicked.connect(self.stop_ai_trading)
        ai_control_layout.addWidget(self.start_ai_button)
        ai_control_layout.addWidget(self.stop_ai_button)
        ai_layout.addLayout(ai_control_layout)

        self.ai_log = QTextEdit()
        self.ai_log.setReadOnly(True)
        self.ai_log.setMinimumHeight(150)
        ai_layout.addWidget(self.ai_log)

        layout.addWidget(QLabel("<b>LSTM 模型训练</b>"))
        layout.addWidget(training_frame)
        layout.addWidget(QLabel("<b>AI 交易</b>"))
        layout.addWidget(ai_frame)
        layout.addStretch()

        self.setLayout(layout)
        self.load_symbols()

    def load_symbols(self):
        symbols = self.mt5_handler.get_symbols()
        self.symbol_combo.clear()
        if not symbols:
            self.ai_log.append("无法加载交易品种，请检查MT5连接")
            return
        for symbol in symbols:
            self.symbol_combo.addItem(symbol)
        index = self.symbol_combo.findText("XAUUSD")
        if index >= 0:
            self.symbol_combo.setCurrentIndex(index)

    def start_training(self):
        self.training_log.clear()
        self.train_button.setEnabled(False)
        symbol = self.symbol_combo.currentText()
        if not self.mt5_handler.symbol_info(symbol):
            self.training_log.append(f"无效交易品种: {symbol}")
            self.train_button.setEnabled(True)
            return
        days = self.days.value()
        units = self.units.value()
        epochs = self.epochs.value()

        def train_model():
            try:
                # 获取历史数据
                to_date = pd.Timestamp.now()
                from_date = to_date - pd.Timedelta(days=days)
                data = self.mt5_handler.get_ohlc_data(symbol, timeframe=mt5.TIMEFRAME_M1, count=days * 1440)
                if data.empty:
                    self.training_log.append("无法获取数据，请检查品种或MT5连接")
                    return

                # 数据预处理
                close_prices = data['close'].values.reshape(-1, 1)
                scaled_data = self.scaler.fit_transform(close_prices)
                X, y = [], []
                time_steps = 60
                for i in range(time_steps, len(scaled_data)):
                    X.append(scaled_data[i-time_steps:i])
                    y.append(scaled_data[i])
                X = np.array(X)
                y = np.array(y)

                # 划分训练集和测试集
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # 构建 LSTM 模型
                self.model = Sequential([                    LSTM(units=units, return_sequences=True, input_shape=(time_steps, 1)),                    LSTM(units=units),                    Dense(1)                ])
                self.model.compile(optimizer='adam', loss='mse')

                # 训练模型
                history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)
                self.training_log.append(f"训练完成：{epochs} 轮，损失：{history.history['loss'][-1]:.6f}")
                
                # 绘制损失曲线
                plt.figure(figsize=(8, 4))
                plt.plot(history.history['loss'], label='训练损失')
                plt.plot(history.history['val_loss'], label='验证损失')
                plt.title('LSTM 训练损失曲线')
                plt.xlabel('轮次')
                plt.ylabel('损失')
                plt.legend()
                plt.grid(True)
                buf = BytesIO()
                plt.savefig(buf, format='png', facecolor='#1e1e2e', edgecolor='#cdd6f4')
                plt.close()
                self.training_log.append("<img src='data:image/png;base64,{}'>".format(
                    buf.getvalue().hex()))
                self.save_model_button.setEnabled(True)
            except Exception as e:
                self.training_log.append(f"训练失败：{str(e)}")
            finally:
                self.train_button.setEnabled(True)

        threading.Thread(target=train_model, daemon=True).start()

    def save_model(self):
        if not self.model:
            self.training_log.append("无模型可保存")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "HDF5 Files (*.h5)")
        if file_path:
            try:
                self.model.save(file_path)
                self.training_log.append(f"模型已保存至：{file_path}")
            except Exception as e:
                self.training_log.append(f"保存模型失败：{str(e)}")

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "HDF5 Files (*.h5)")
        if file_path:
            try:
                self.model = load_model(file_path)
                self.model_path.setText(file_path)
                self.ai_log.append(f"模型加载成功：{file_path}")
            except Exception as e:
                self.ai_log.append(f"加载模型失败：{str(e)}")

    def start_ai_trading(self):
        if not self.model:
            self.ai_log.append("请先加载模型")
            return
        if self.trading_thread and self.trading_thread.is_alive():
            self.ai_log.append("AI交易已在运行")
            return
        symbol = self.symbol_combo.currentText()
        if not self.mt5_handler.symbol_info(symbol):
            self.ai_log.append(f"无效交易品种: {symbol}")
            return
        sl = 50  # Default SL
        tp = 100  # Default TP
        self.stop_trading_flag = False

        def trading_loop():
            while not self.stop_trading_flag:
                try:
                    data = self.mt5_handler.get_ohlc_data(symbol, timeframe=mt5.TIMEFRAME_M1, count=61)
                    if data.empty:
                        self.ai_log.append("无法获取数据，暂停预测")
                        threading.Event().wait(1)
                        continue
                    close_prices = data['close'].values.reshape(-1, 1)
                    scaled_data = self.scaler.transform(close_prices)
                    X = scaled_data[-60:].reshape(1, 60, 1)
                    pred = self.model.predict(X, verbose=0)[0][0]
                    pred_price = self.scaler.inverse_transform([[pred]])[0][0]
                    current_price = self.mt5_handler.get_current_price(symbol)
                    
                    self.ai_log.append(f"预测价格：{pred_price:.2f}, 当前价格：{current_price:.2f}")
                    
                    if pred_price > current_price * 1.005:
                        if self.mt5_handler.execute_trade(symbol, 0.1, sl, tp, 'buy'):
                            self.ai_log.append(f"执行买入交易 - 品种: {symbol}, SL: {sl}, TP: {tp}")
                    elif pred_price < current_price * 0.995:
                        if self.mt5_handler.execute_trade(symbol, 0.1, sl, tp, 'sell'):
                            self.ai_log.append(f"执行卖出交易 - 品种: {symbol}, SL: {sl}, TP: {tp}")
                    
                    threading.Event().wait(1)
                except Exception as e:
                    self.ai_log.append(f"交易错误：{str(e)}")

        self.trading_thread = threading.Thread(target=trading_loop, daemon=True)
        self.trading_thread.start()
        self.ai_log.append(f"AI交易启动 - 品种: {symbol}")

    def stop_ai_trading(self):
        self.stop_trading_flag = True
        if self.trading_thread:
            self.trading_thread.join()
            self.trading_thread = None
        self.ai_log.append("AI交易停止")