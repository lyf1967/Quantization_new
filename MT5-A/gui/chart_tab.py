import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QSpinBox, QPlainTextEdit, QFileDialog, QFrame, QLineEdit, QDoubleSpinBox, QCheckBox
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QObject
import sys
sys.path.append("../")
sys.path.append("./")
from mt5.mt5_handler import MT5Handler
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout
import matplotlib.pyplot as plt
from io import BytesIO
import threading
import traceback
import time
import tensorflow as tf
import joblib
import pandas_ta as ta
from datetime import datetime
from . import train_patchmixer 
from .train_patchmixer import PatchEmbedding, PatchMixerBlock, PatchMixerModel, calculate_technical_indicators

class ChartTab(QWidget):
    log_signal = pyqtSignal(str)
    ai_log_signal = pyqtSignal(str)

    def __init__(self, mt5_handler: MT5Handler):
        super().__init__()
        self.mt5_handler = mt5_handler
        self.scaler = RobustScaler()
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
            QComboBox, QSpinBox, QPlainTextEdit, QLineEdit, QDoubleSpinBox, QCheckBox {
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

        training_frame = QFrame()
        training_layout = QVBoxLayout()
        training_frame.setLayout(training_layout)

        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("交易品种："))
        self.symbol_search = QLineEdit()
        self.symbol_search.setPlaceholderText("搜索交易品种...")
        self.symbol_search.textChanged.connect(self.filter_training_symbols)
        self.symbol_combo = QComboBox()
        self.symbol_combo.setMinimumWidth(200)
        symbol_layout.addWidget(self.symbol_search)
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
        self.epochs.setRange(1, 500)
        self.epochs.setValue(50)
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

        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMinimumHeight(150)
        self.training_log.setMaximumBlockCount(200)
        training_layout.addWidget(self.training_log)

        self.log_signal.connect(self.append_training_log)
        self.ai_log_signal.connect(self.append_ai_log)

        ai_frame = QFrame()
        ai_layout = QVBoxLayout()
        ai_frame.setLayout(ai_layout)

        ai_symbol_layout = QHBoxLayout()
        ai_symbol_layout.addWidget(QLabel("交易品种："))
        self.ai_symbol_search = QLineEdit()
        self.ai_symbol_search.setPlaceholderText("搜索交易品种...")
        self.ai_symbol_search.textChanged.connect(self.filter_symbols)
        self.ai_symbol_combo = QComboBox()
        self.ai_symbol_combo.setMinimumWidth(200)
        ai_symbol_layout.addWidget(self.ai_symbol_search)
        ai_symbol_layout.addWidget(self.ai_symbol_combo)
        ai_layout.addLayout(ai_symbol_layout)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("交易量："))
        self.volume = QDoubleSpinBox()
        self.volume.setRange(0.01, 100.0)
        self.volume.setValue(0.1)
        params_layout.addWidget(self.volume)
        params_layout.addWidget(QLabel("止损点位："))
        self.sl = QDoubleSpinBox()
        self.sl.setRange(0, 50000)
        self.sl.setValue(0)
        params_layout.addWidget(self.sl)
        params_layout.addWidget(QLabel("止盈点位："))
        self.tp = QDoubleSpinBox()
        self.tp.setRange(0, 50000)
        self.tp.setValue(3000)
        params_layout.addWidget(self.tp)
        params_layout.addWidget(QLabel("动态止损："))
        self.dynamic_sl = QCheckBox("启用")
        params_layout.addWidget(self.dynamic_sl)
        params_layout.addWidget(QLabel("动态止盈："))
        self.dynamic_tp = QCheckBox("启用")
        params_layout.addWidget(self.dynamic_tp)
        params_layout.addWidget(QLabel("最大开单量："))
        self.max_positions = QSpinBox()
        self.max_positions.setRange(1, 500)
        self.max_positions.setValue(50)
        params_layout.addWidget(self.max_positions)
        ai_layout.addLayout(params_layout)

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

        self.ai_log = QPlainTextEdit()
        self.ai_log.setReadOnly(True)
        self.ai_log.setMinimumHeight(150)
        self.ai_log.setMaximumBlockCount(20)
        ai_layout.addWidget(self.ai_log)

        layout.addWidget(QLabel("<b>Transformer 价格预测模型训练</b>"))
        layout.addWidget(training_frame)
        layout.addWidget(QLabel("<b>AI 交易</b>"))
        layout.addWidget(ai_frame)
        layout.addStretch()

        self.setLayout(layout)
        self.load_symbols()

    @pyqtSlot(str)
    def append_training_log(self, text):
        self.training_log.appendPlainText(text)

    @pyqtSlot(str)
    def append_ai_log(self, text):
        self.ai_log.appendPlainText(text)

    def load_symbols(self):
        symbols = self.mt5_handler.get_symbols()
        self.symbol_combo.clear()
        self.ai_symbol_combo.clear()
        if not symbols:
            self.ai_log_signal.emit("无法加载交易品种，请检查MT5连接")
            return
        for symbol in symbols:
            self.symbol_combo.addItem(symbol)
            self.ai_symbol_combo.addItem(symbol)
        index = self.ai_symbol_combo.findText("XAUUSD")
        if index >= 0:
            self.symbol_combo.setCurrentIndex(index)
            self.ai_symbol_combo.setCurrentIndex(index)

    def filter_training_symbols(self):
        search_text = self.symbol_search.text().lower()
        self.symbol_combo.clear()
        symbols = self.mt5_handler.get_symbols()
        for symbol in symbols:
            if search_text in symbol.lower():
                self.symbol_combo.addItem(symbol)

    def filter_symbols(self):
        search_text = self.ai_symbol_search.text().lower()
        self.ai_symbol_combo.clear()
        symbols = self.mt5_handler.get_symbols()
        for symbol in symbols:
            if search_text in symbol.lower():
                self.ai_symbol_combo.addItem(symbol)

    def start_training(self):
        self.training_log.clear()
        self.train_button.setEnabled(False)
        symbol = self.symbol_combo.currentText()
        self.log_signal.emit(f"开始训练 - 品种: {symbol}, 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if not self.mt5_handler.symbol_info(symbol):
            self.log_signal.emit(f"无效交易品种: {symbol}")
            self.train_button.setEnabled(True)
            return
        if not mt5.initialize():
            self.log_signal.emit("MT5初始化失败，请检查MT5终端")
            self.train_button.setEnabled(True)
            return
        days = self.days.value()
        units = self.units.value()
        epochs = self.epochs.value()

        self.training_thread = QThread()
        self.training_worker = TrainingWorker(self, symbol, days, units, epochs)
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.finished.connect(self.training_worker.deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        self.training_thread.start()

    def save_model(self):
        if not self.model:
            self.training_log.appendPlainText("无模型可保存")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "HDF5 Files (*.h5)")
        if file_path:
            try:
                self.model.save(file_path)
                self.training_log.appendPlainText(f"模型已保存至：{file_path}")
                scaler_path = file_path.replace('.h5', '_scaler.joblib')
                joblib.dump(self.scaler, scaler_path)
                self.training_log.appendPlainText(f"Scaler 已保存至：{scaler_path}")
            except Exception as e:
                self.training_log.appendPlainText(f"保存模型或Scaler失败：{str(e)}")

    def browse_model(self):
        # file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "HDF5 Files (*.h5)")
        file_path = QFileDialog.getExistingDirectory(
                self, 
                "选择目录",  # 对话框标题
                "",         # 初始目录（空字符串表示当前目录）
                options=QFileDialog.ShowDirsOnly  # 可选参数：只显示目录
            )
        if file_path:
            try:
                # self.model = load_model(file_path)
                self.model = load_model(os.path.join(file_path, "patchmixer_model.keras"), custom_objects={
                        'PatchEmbedding': PatchEmbedding,
                        'PatchMixerBlock': PatchMixerBlock,
                        'PatchMixerModel': PatchMixerModel
                    })
                self.model_path.setText(file_path)
                self.ai_log_signal.emit(f"模型加载成功：{file_path}")
                # scaler_path = file_path.replace('.h5', '_scaler.joblib')
                scaler_path = os.path.join(file_path, "patchmixer_scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    self.ai_log_signal.emit(f"Scaler 加载成功：{scaler_path}")
                else:
                    self.ai_log_signal.emit(f"未找到 Scaler 文件：{scaler_path}，将在交易时动态拟合")
            except Exception as e:
                self.ai_log_signal.emit(f"加载模型或Scaler失败：{str(e)}")

    def start_ai_trading(self):
        if not self.model:
            self.ai_log_signal.emit("请先加载模型")
            return
        if self.trading_thread and self.trading_thread.isRunning():
            self.ai_log_signal.emit("AI交易已在运行")
            return
        symbol = self.ai_symbol_combo.currentText()
        if not self.mt5_handler.symbol_info(symbol):
            self.ai_log_signal.emit(f"无效交易品种: {symbol}")
            return
        if not mt5.initialize():
            self.ai_log_signal.emit("MT5初始化失败，请检查MT5终端")
            return
        volume = self.volume.value()
        sl = self.sl.value() if not self.dynamic_sl.isChecked() else 0
        tp = self.tp.value() if not self.dynamic_tp.isChecked() else 0
        max_positions = self.max_positions.value()
        dynamic_sl = self.dynamic_sl.isChecked()
        dynamic_tp = self.dynamic_tp.isChecked()
        self.stop_trading_flag = False

        self.trading_thread = QThread()
        self.trading_worker = TradingWorker(self, symbol, volume, sl, tp, max_positions, dynamic_sl, dynamic_tp)
        self.trading_worker.moveToThread(self.trading_thread)
        self.trading_thread.started.connect(self.trading_worker.run)
        self.trading_worker.finished.connect(self.trading_thread.quit)
        self.trading_worker.finished.connect(self.trading_worker.deleteLater)
        self.trading_thread.finished.connect(self.trading_thread.deleteLater)
        self.trading_thread.start()

        self.ai_log_signal.emit(f"AI交易启动 - 品种: {symbol}, 交易量: {volume}, 最大开单量: {max_positions}, 动态止损: {'启用' if dynamic_sl else '禁用'}, 动态止盈: {'启用' if dynamic_tp else '禁用'}")

    def stop_ai_trading(self):
        self.stop_trading_flag = True
        if self.trading_thread:
            self.trading_thread.quit()
            self.trading_thread.wait()
            self.trading_thread = None
        self.ai_log_signal.emit("AI交易停止")

    def calculate_atr(self, data, period=14):
        if data.empty:
            return 0.0
        data['tr'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low'] - data['close'].shift()).abs()
        ], axis=1).max(axis=1)
        return data['tr'].rolling(window=period).mean().iloc[-1]

class TrainingWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, parent, symbol, days, units, epochs):
        super().__init__()
        self.parent = parent
        self.symbol = symbol
        self.days = days
        self.units = units
        self.epochs = epochs
    

    def run(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.set_visible_devices(gpus, 'GPU')
                self.parent.log_signal.emit(f"使用GPU设备: {gpus}")
            else:
                self.parent.log_signal.emit("未检测到GPU，使用CPU")

            self.parent.log_signal.emit("=== 数据获取 ===")
            to_date = pd.Timestamp.now()
            from_date = to_date - pd.Timedelta(days=self.days)
            # 1440 = 60分钟 * 24小时
            data = self.parent.mt5_handler.get_ohlc_data(self.symbol, timeframe=mt5.TIMEFRAME_M1, count=self.days * 1440 + 100)
            self.parent.log_signal.emit(f"data.columns: {data.columns}")

            self.parent.log_signal.emit(f"获取数据完成，行数: {len(data)}, 时间范围: {from_date} 至 {to_date}")
            if data.empty:
                self.parent.log_signal.emit("无法获取数据，请检查品种或MT5连接")
                return

            if data['close'].isnull().any() or not np.all(np.isfinite(data['close'])):
                self.parent.log_signal.emit("收盘价包含NaN或非有限值，尝试填补")
                data['close'] = data['close'].fillna(method='ffill').fillna(method='bfill')
                if data['close'].isnull().any():
                    self.parent.log_signal.emit("收盘价填补失败，数据无效")
                    return

            self.parent.log_signal.emit("=== 价格数据统计 ===")
            price_values = data['close'].values.reshape(-1, 1)
            if np.any(np.isnan(price_values)) or np.any(~np.isfinite(price_values)):
                self.parent.log_signal.emit("价格数据仍包含NaN或非有限值，训练中止")
                return
            self.parent.log_signal.emit(f"价格统计 - 最大值: {price_values.max():.2f}, 最小值: {price_values.min():.2f}, 均值: {price_values.mean():.2f}")
            price_range = price_values.max() - price_values.min()
            self.parent.log_signal.emit(f"价格范围: {price_range:.2f}")

            self.parent.log_signal.emit("=== 数据预处理 ===")
            # 最大最小归一化
            # X_std = (X - X.min) / (X.max - X.min)
            # X_scaled = X_std * (max - min) + min
            scaled_data = self.parent.scaler.fit_transform(price_values)
            if np.any(np.isnan(scaled_data)) or np.any(~np.isfinite(scaled_data)):
                self.parent.log_signal.emit("Scaler 输出包含NaN或非有限值，训练中止")
                return
            X, y = [], []
            time_steps = 60
            for i in range(time_steps, len(scaled_data)):
                X.append(scaled_data[i-time_steps:i])
                y.append(scaled_data[i])
            X = np.array(X)
            y = np.array(y)
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                self.parent.log_signal.emit("训练数据X或y包含NaN，训练中止")
                return
            self.parent.log_signal.emit(f"数据预处理完成，样本数: {len(X)}, 时间步长: {time_steps}")

            self.parent.log_signal.emit("=== 数据分割 ===")
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            self.parent.log_signal.emit(f"训练集样本: {len(X_train)}, 验证集样本: {len(X_test)}")
            train_price = self.parent.scaler.inverse_transform(y_train)
            test_price = self.parent.scaler.inverse_transform(y_test)
            self.parent.log_signal.emit(f"训练集价格统计 - 最大值: {train_price.max():.2f}, 最小值: {train_price.min():.2f}, 均值: {train_price.mean():.2f}")
            self.parent.log_signal.emit(f"验证集价格统计 - 最大值: {test_price.max():.2f}, 最小值: {test_price.min():.2f}, 均值: {test_price.mean():.2f}")

            self.parent.log_signal.emit("=== 模型构建 ===")
            def transformer_block(inputs, units, num_heads=4, dropout=0.2):
                attention = MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads)(inputs, inputs)
                attention = Dropout(dropout)(attention)
                attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
                ffn = Dense(units, activation='relu')(attention)
                ffn = Dense(inputs.shape[-1])(ffn)
                ffn = Dropout(dropout)(ffn)
                return LayerNormalization(epsilon=1e-6)(attention + ffn)

            inputs = tf.keras.Input(shape=(time_steps, 1))
            x = transformer_block(inputs, self.units)
            x = transformer_block(x, self.units)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            outputs = Dense(1)(x)
            self.parent.model = tf.keras.Model(inputs, outputs)

            self.parent.model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss='mse')
            total_params = self.parent.model.count_params()
            self.parent.log_signal.emit(f"Transformer模型构建完成，层数: {len(self.parent.model.layers)}, 总参数量: {total_params}")

            self.parent.log_signal.emit("=== 模型训练 ===")
            total_start_time = time.time()
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            with open("training_loss.csv", "w") as f:
                f.write("Epoch,Train_Loss,Val_Loss,Train_RMSE,Val_RMSE,Val_MAE,Time\n")
            history = {'loss': [], 'val_loss': []}
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                hist = self.parent.model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping, lr_scheduler])
                train_loss = hist.history['loss'][0]
                val_loss = hist.history['val_loss'][0]
                history['loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                epoch_time = time.time() - epoch_start_time

                if np.isnan(train_loss) or np.isnan(val_loss):
                    self.parent.log_signal.emit(f"第 {epoch+1} 轮损失为NaN，训练中止")
                    return

                train_rmse = np.sqrt(train_loss) * price_range
                val_rmse = np.sqrt(val_loss) * price_range
                y_pred = self.parent.model.predict(X_test, verbose=0)
                y_pred_price = self.parent.scaler.inverse_transform(y_pred)
                y_test_price = self.parent.scaler.inverse_transform(y_test)
                val_mae = np.mean(np.abs(y_pred_price - y_test_price))

                self.parent.log_signal.emit(f"第 {epoch+1}/{self.epochs} 轮 ({(epoch+1)/self.epochs*100:.1f}%)")
                self.parent.log_signal.emit(f"  训练损失: {train_loss:.6f} (归一化 MSE), RMSE: {train_rmse:.2f}")
                self.parent.log_signal.emit(f"  验证损失: {val_loss:.6f} (归一化 MSE), RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")
                self.parent.log_signal.emit(f"  耗时: {epoch_time:.2f} 秒")

                with open("training_loss.csv", "a") as f:
                    f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{train_rmse:.2f},{val_rmse:.2f},{val_mae:.2f},{epoch_time:.2f}\n")

                if (epoch + 1) % 5 == 0:
                    self.parent.log_signal.emit("  样本预测（前 5 个验证集样本）：")
                    for i in range(min(5, len(y_test_price))):
                        self.parent.log_signal.emit(f"    样本 {i+1}: 实际价格: {y_test_price[i][0]:.2f}, 预测价格: {y_pred_price[i][0]:.2f}, 误差: {abs(y_test_price[i][0] - y_pred_price[i][0]):.2f}")

            total_time = time.time() - total_start_time
            self.parent.log_signal.emit(f"训练完成：{len(history['loss'])} 轮，总耗时: {total_time:.2f} 秒")

            self.parent.log_signal.emit("=== 最终评估 ===")
            y_pred = self.parent.model.predict(X_test, verbose=0)
            y_pred_price = self.parent.scaler.inverse_transform(y_pred)
            y_test_price = self.parent.scaler.inverse_transform(y_test)
            final_mse = np.mean((y_pred_price - y_test_price) ** 2)
            final_rmse = np.sqrt(final_mse)
            final_mae = np.mean(np.abs(y_pred_price - y_test_price))
            self.parent.log_signal.emit(f"最终验证集 MSE: {final_mse:.2f}")
            self.parent.log_signal.emit(f"最终验证集 RMSE: {final_rmse:.2f}")
            self.parent.log_signal.emit(f"最终验证集 MAE: {final_mae:.2f}")

            self.parent.log_signal.emit("=== 绘制损失曲线 ===")
            plt.figure(figsize=(6, 3))
            plt.plot(history['loss'], label='训练损失')
            plt.plot(history['val_loss'], label='验证损失')
            plt.title('Transformer 价格预测训练损失曲线')
            plt.xlabel('轮次')
            plt.ylabel('归一化 MSE')
            plt.legend()
            plt.grid(True)
            buf = BytesIO()
            plt.savefig(buf, format='png', facecolor='#1e1e2e', edgecolor='#cdd6f4')
            with open("loss_curve.png", "wb") as f:
                f.write(buf.getvalue())
            plt.close()
            self.parent.log_signal.emit("损失曲线已保存到 loss_curve.png")
            self.parent.log_signal.emit("<img src='data:image/png;base64,{}'>".format(buf.getvalue().hex()))
            self.parent.save_model_button.setEnabled(True)
        except Exception as e:
            self.parent.log_signal.emit(f"训练失败：{str(e)}\n{traceback.format_exc()}")
        finally:
            self.parent.log_signal.emit("=== 训练结束 ===")
            self.parent.train_button.setEnabled(True)
            self.finished.emit()

class TradingWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, parent, symbol, volume, sl, tp, max_positions, dynamic_sl, dynamic_tp):
        super().__init__()
        self.parent = parent
        self.symbol = symbol
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.max_positions = max_positions
        self.dynamic_sl = dynamic_sl
        self.dynamic_tp = dynamic_tp
        self.price_change_threshold = 0.003  # 价格变化百分比阈值（1%） 基线值：0.01  ori: 0.003
        self.profit_change_threshold = -0.2
        # 新增止损参数
        # self.max_drawdown = -50  # 最大允许亏损金额（单位：美元）
        self.max_drawdown = -1 * volume / 0.01 * 30 * max_positions  # 最大允许亏损金额  基线值：3
        self.take_profit = volume / 0.01 * 0.1 * max_positions  # 止盈利润  # 基线值：2
        self.last_stop_time = None  # 记录上次止损时间
        self.last_stop_loss_time = None
        self.last_take_profit_time = None
        self.silence_period = 600  # 静默时间（10分钟=600秒） # 未使用
        self.silence_period_stop_loss = 30 * 60   # 止损静默30分钟  # 基线值：30 * 60
        self.silence_period_take_profit = 1 * 60  # 止盈静默1分钟  基线值：10 * 60
        self.trade_time_gap = 0  # 两单之间的间隔（秒）
        self.prev_profit = 0
        self.max_profix = 0
        self.max_stop_loss_time = 120*60

    # 新增方法：平仓所有持仓
    def close_all_positions(self):
        positions = self.parent.mt5_handler.get_open_positions()
        if not positions:  # 检查空持仓
            self.parent.ai_log_signal.emit("无持仓需要平仓")
            return
        for position in positions:
            try:
                # 使用 close_specific_position 替代 close_position
                self.parent.mt5_handler.close_specific_position(position['symbol'], position['ticket'])
                self.parent.ai_log_signal.emit(
                    f"平仓 #{position['ticket']} {position['symbol']} {position['volume']}手"
                )
            except Exception as e:
                self.parent.ai_log_signal.emit(f"平仓失败 #{position['ticket']}: {str(e)}")

    # def calculate_technical_indicators(self, df, drop_nan=True):
    #     df['rsi'] = ta.rsi(df['close'], length=14)
    #     df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    #     macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    #     df['macd'] = macd['MACD_12_26_9']
    #     df['macd_signal'] = macd['MACDs_12_26_9']
    #     bollinger = ta.bbands(df['close'], length=20, std=2)
    #     df['bb_upper'] = bollinger['BBU_20_2.0']
    #     df['bb_middle'] = bollinger['BBM_20_2.0']
    #     df['bb_lower'] = bollinger['BBL_20_2.0']
    #     df['obv'] = ta.obv(df['close'], df['tick_volume'])
    #     df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    #     df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['tick_volume'])
    #     if drop_nan:
    #         df = df.dropna()
    #         return df
    
    #     df = df.fillna(method='ffill').fillna(method='bfill')
    #     # 二次检查：若仍有NaN则删除（通常不会出现）
    #     if df.isnull().values.any():
    #         print(f"填充后仍有{df.isnull().sum().sum()}个NaN，执行最终清理")
    #         df = df.dropna()
    #     return df


    def run(self):
        # try:
        #     self.parent.scaler.transform([[0]])
        #     # self.parent.ai_log_signal.emit("Scaler 已适配，范围: [{:.2f}, {:.2f}]".format(
        #     #     self.parent.scaler.data_min_[0], self.parent.scaler.data_max_[0]))
        # except:
        #     self.parent.ai_log_signal.emit("Scaler 未适配，尝试使用近期数据动态拟合")
        #     try:
        #         data = self.parent.mt5_handler.get_ohlc_data(self.symbol, timeframe=mt5.TIMEFRAME_M5, count=60)
        #         if data.empty:
        #             self.parent.ai_log_signal.emit("无法获取数据以拟合 Scaler，交易中止")
        #             return
        #         if data['close'].isnull().any() or not np.all(np.isfinite(data['close'])):
        #             self.parent.ai_log_signal.emit("收盘价包含NaN或非有限值，尝试填补")
        #             data['close'] = data['close'].fillna(method='ffill').fillna(method='bfill')
        #             if data['close'].isnull().any():
        #                 self.parent.ai_log_signal.emit("收盘价填补失败，交易中止")
        #                 return
        #         price_values = data['close'].values.reshape(-1, 1)
        #         if np.any(np.isnan(price_values)) or np.any(~np.isfinite(price_values)):
        #             self.parent.ai_log_signal.emit("价格数据仍包含NaN或非有限值，交易中止")
        #             return
        #         self.parent.scaler.fit(price_values)
        #         # self.parent.ai_log_signal.emit("Scaler 动态拟合成功，范围: [{:.2f}, {:.2f}]".format(
        #         #     self.parent.scaler.data_min_[0], self.parent.scaler.data_max_[0]))
        #     except Exception as e:
        #         self.parent.ai_log_signal.emit(f"Scaler 动态拟合失败：{str(e)}")
        #         return

        open_time = datetime.now()
        while not self.parent.stop_trading_flag:
            # 检查静默状态（10分钟内不交易）[1,2](@ref)
            # if self.last_stop_time and (time.time() - self.last_stop_time) < self.silence_period:
            #     # 显示剩余静默时间
            #     remaining = int(self.silence_period - (time.time() - self.last_stop_time))
            #     resume_time = time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
            #     self.parent.ai_log_signal.emit(f"🛑 静默期中，剩余时间: {remaining}秒，恢复时间: {resume_time}")
            #     QThread.msleep(1000)  # 用于强制当前线程休眠 1000 毫秒（即 1 秒）
            #     continue

            if self.last_stop_loss_time and (time.time() - self.last_stop_loss_time) < self.silence_period_stop_loss:
                # 显示剩余静默时间
                remaining = int(self.silence_period_stop_loss - (time.time() - self.last_stop_loss_time))
                resume_time = time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
                self.parent.ai_log_signal.emit(f"🛑 止损静默期中，剩余时间: {remaining}秒，恢复时间: {resume_time}")
                QThread.msleep(1000)  # 用于强制当前线程休眠 1000 毫秒（即 1 秒）
                continue

            if self.last_take_profit_time and (time.time() - self.last_take_profit_time) < self.silence_period_take_profit:
                # 显示剩余静默时间
                remaining = int(self.silence_period_take_profit - (time.time() - self.last_take_profit_time))
                resume_time = time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
                self.parent.ai_log_signal.emit(f"🛑 止盈静默期中，剩余时间: {remaining}秒，恢复时间: {resume_time}")
                QThread.msleep(1000)  # 用于强制当前线程休眠 1000 毫秒（即 1 秒）
                continue

            try:
                # 新增：账户盈亏检查（在现有代码前插入）[1,2](@ref)
                account_info = self.parent.mt5_handler.get_account_info()
                self.parent.ai_log_signal.emit(f"单次交易手数:{self.volume}")

                current_time = datetime.now()
                time_pass = (current_time - open_time).total_seconds()
                self.parent.ai_log_signal.emit(f"open_time:{open_time}, current_time:{current_time}, time_pass:{time_pass}")
    
                if account_info and ((account_info['profit'] <= self.max_drawdown) or (account_info['profit'] < 0 and time_pass > self.max_stop_loss_time)):
                    self.parent.ai_log_signal.emit(
                        f"⚠️ 触发总止损：浮动亏损 {account_info['profit']:.2f} 已达阈值 {self.max_drawdown}，平仓所有单"
                    )
                    self.close_all_positions()  # 平仓所有持仓[3,4](@ref)
                    self.prev_profit = 0
                    self.max_profix = 0
                    self.last_stop_loss_time = time.time()  # 记录止损时间

                    # 记录静默状态
                    resume_time = time.strftime("%H:%M:%S", time.localtime(time.time() + self.silence_period_stop_loss))
                    self.parent.ai_log_signal.emit(f"🛑 进入{self.silence_period_stop_loss/60}分钟止损静默期，恢复时间：{resume_time}")
                    continue  # 跳过本轮循环

                current_profit = account_info['profit']
                self.parent.ai_log_signal.emit(f"prev_profit: {self.prev_profit}, current_profit: {current_profit}")
                profit_change_pct2 = 0
                self.parent.ai_log_signal.emit(f"max_profix:{self.max_profix}")
                if self.max_profix > 0:
                    profit_change_pct2 = (current_profit - self.max_profix) / self.max_profix
                    self.parent.ai_log_signal.emit(f"profit_change_pct2= {profit_change_pct2}")

                # 触发止盈
                current_postion_num = len(self.parent.mt5_handler.get_open_positions())
                target_profit = self.take_profit / self.max_positions * current_postion_num
                self.parent.ai_log_signal.emit(
                    f"account_info: {account_info}, account_info['profit']:{account_info['profit']}")
                self.parent.ai_log_signal.emit(f"-1 * self.max_drawdown:{-1 * self.max_drawdown}")
                self.parent.ai_log_signal.emit(
                    f"account_info and account_info['profit'] >= target_profit: {account_info and account_info['profit'] >= target_profit}")
                self.parent.ai_log_signal.emit(f"price_change_pct2: {profit_change_pct2}")
                if account_info and account_info['profit'] >= target_profit:
                    self.parent.ai_log_signal.emit(f"profit_change_pct2 <= self.profit_change_threshold: {profit_change_pct2 <= self.profit_change_threshold}")
                    if profit_change_pct2 <= self.profit_change_threshold:
                        self.parent.ai_log_signal.emit(
                            f"⚠️ 触发总止盈：浮动盈利 {account_info['profit']:.2f} 已达阈值 {self.take_profit / self.max_positions * current_postion_num}，平仓所有单"
                        )
                        self.close_all_positions()  # 平仓所有持仓[3,4](@ref)
                        self.prev_profit = 0
                        self.max_profix = 0
                        self.last_take_profit_time = time.time()  # 记录止盈时间

                        # 记录静默状态
                        resume_time = time.strftime("%H:%M:%S", time.localtime(time.time() + self.silence_period_stop_loss))
                        self.parent.ai_log_signal.emit(f"🛑 进入{self.silence_period_take_profit/60}分钟止盈静默期，恢复时间：{resume_time}")
                        continue  # 跳过本轮循环

                current_positions = len(self.parent.mt5_handler.get_open_positions())
                self.parent.ai_log_signal.emit(f"account_info['profit']: {account_info['profit']}")

                if current_profit > self.max_profix:
                    self.max_profix = current_profit
                    self.parent.ai_log_signal.emit(f"update max_profix = {self.max_profix}")

                if current_positions >= self.max_positions:
                    self.parent.ai_log_signal.emit(f"已达到最大开单量 {self.max_positions}，暂停开新仓")
                    self.prev_profit = current_profit
                    QThread.msleep(1000)
                    continue

                data = self.parent.mt5_handler.get_ohlc_data(self.symbol, timeframe=mt5.TIMEFRAME_M5, count=60+20)  # ori: mt5.TIMEFRAME_M5
                self.parent.ai_log_signal.emit(f"data columns:{data.columns}")
                if data.empty:
                    self.parent.ai_log_signal.emit("无法获取数据，暂停预测")
                    QThread.msleep(1000)
                    continue
                self.parent.ai_log_signal.emit(f"获取K线数据，行数: {len(data)}")
                
                if data['close'].isnull().any() or not np.all(np.isfinite(data['close'])):
                    self.parent.ai_log_signal.emit(f"收盘价包含NaN（比例: {data['close'].isnull().mean():.2%}）或非有限值，尝试填补")
                    data['close'] = data['close'].fillna(method='ffill').fillna(method='bfill')
                    if data['close'].isnull().any():
                        self.parent.ai_log_signal.emit("收盘价填补失败，暂停预测")
                        QThread.msleep(1000)
                        continue
                
                data = calculate_technical_indicators(data, drop_nan=False)

                target_col = data.pop('close')  # 移除目标列并保存
                data.insert(0, 'close', target_col)  # 插入到第一列

                feature_columns = list(data.columns)
                
                price_values = data['close'].values.reshape(-1, 1)
                if np.any(np.isnan(price_values)) or np.any(~np.isfinite(price_values)):
                    self.parent.ai_log_signal.emit("价格数据仍包含NaN或非有限值，暂停预测")
                    QThread.msleep(1000)
                    continue
                
                scaled_data = self.parent.scaler.transform(data)
                if np.any(np.isnan(scaled_data)) or np.any(~np.isfinite(scaled_data)):
                    self.parent.ai_log_signal.emit("Scaler 输出包含NaN或非有限值，暂停预测")
                    QThread.msleep(1000)
                    continue
                
                X = scaled_data[-60:].reshape(1, 60, -1)
                y_pred = self.parent.model.predict(X)
                if np.isnan(y_pred) or not np.isfinite(y_pred):
                    self.parent.ai_log_signal.emit("模型预测值无效（NaN或非有限），暂停预测")
                    QThread.msleep(1000)
                    continue
                pred_vector = np.hstack([y_pred, np.zeros((1, len(feature_columns)-1))])
                pred_price = self.parent.scaler.inverse_transform(pred_vector)[0, 0]
                if np.isnan(pred_price) or not np.isfinite(pred_price):
                    self.parent.ai_log_signal.emit("逆缩放后的价格值无效（NaN或非有限），暂停预测")
                    QThread.msleep(1000)
                    continue
                
                current_price = data['close'].iloc[-1]
                price_change = (pred_price - current_price) / current_price if current_price != 0 else 0.0
                error = abs(pred_price - current_price)

                self.parent.ai_log_signal.emit(
                    f"预测价格：{pred_price:.2f}, 当前价格：{current_price:.2f}, "
                    f"预测变化：{price_change*100:.2f}%, 误差：{error:.2f}"
                )

                signal = None
                if price_change > self.price_change_threshold:
                    signal = 'buy'
                    self.parent.ai_log_signal.emit(f"触发买入信号：预测价格 {pred_price:.2f}，变化 {price_change*100:.2f}% > {self.price_change_threshold*100:.2f}%")
                elif price_change < -self.price_change_threshold:
                    signal = 'sell'
                    self.parent.ai_log_signal.emit(f"触发卖出信号：预测价格 {pred_price:.2f}，变化 {price_change*100:.2f}% < {-self.price_change_threshold*100:.2f}%")
                else:
                    self.parent.ai_log_signal.emit(f"未触发交易：预测价格变化 {price_change*100:.2f}% 未达到阈值 ±{self.price_change_threshold*100:.2f}%")

                if signal:
                    adjusted_sl = self.sl
                    adjusted_tp = self.tp
                    if self.dynamic_sl or self.dynamic_tp:
                        atr = self.parent.calculate_atr(data)
                        if self.dynamic_sl:
                            adjusted_sl = atr * 1.5
                        if self.dynamic_tp:
                            adjusted_tp = atr * 3.0
                        self.parent.ai_log_signal.emit(f"动态参数 - ATR: {atr:.2f}, SL: {adjusted_sl:.2f}, TP: {adjusted_tp:.2f}")

                    if self.parent.mt5_handler.execute_trade(self.symbol, self.volume, adjusted_sl, adjusted_tp, signal, self.dynamic_sl, self.dynamic_tp):
                        self.parent.ai_log_signal.emit(f"执行{signal.upper()}交易 - 品种: {self.symbol}, 交易量: {self.volume}, SL: {adjusted_sl:.2f}, TP: {adjusted_tp:.2f}")
                        open_time = datetime.now()
                        time.sleep(self.trade_time_gap)
                        account_info = self.parent.mt5_handler.get_account_info()
                        if account_info:
                            self.parent.ai_log_signal.emit(f"账户状态 - 余额: {account_info['balance']:.2f}, 浮动盈亏: {account_info['profit']:.2f}")
                    else:
                        self.parent.ai_log_signal.emit(f"执行{signal.upper()}交易失败 - 品种: {self.symbol}")

                # QThread.msleep(1000)
            except Exception as e:
                self.parent.ai_log_signal.emit(f"交易错误：{str(e)}")
                QThread.msleep(1000)

        self.finished.emit()