import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
import joblib
import pandas_ta as ta
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from mt5.mt5_handler import MT5Handler
import matplotlib.pyplot as plt


# PatchMixer Layers
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        # 修改处：将 **kwargs 传给父类
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Dense(embed_dim)
    
    # 新增方法（在 __init__ 后，call 前）
    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim
        })
        return super.get_config()

    @classmethod
    def from_config(cls, config):
        patch_size = config.pop('patch_size')
        embed_dim = config.pop('embed_dim')
        return cls(patch_size, embed_dim, **config)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        num_features = x.shape[-1]
        num_patches = seq_len // self.patch_size
        x = tf.reshape(x, (batch_size, num_patches, self.patch_size * num_features))
        x = self.proj(x)
        return x

class PatchMixerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, kernel_size, **kwargs):
        # 修改处：将 **kwargs 传给父类
        super(PatchMixerBlock, self).__init__(**kwargs)
        self.depthwise = tf.keras.layers.DepthwiseConv1D(kernel_size, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.gelu = tf.keras.layers.Activation('gelu')
        self.pointwise = tf.keras.layers.Conv1D(embed_dim, 1)
        self.add = tf.keras.layers.Add()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
            
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,  # 注意：需在 __init__ 中添加 self.embed_dim = embed_dim
            'kernel_size': self.kernel_size  # 同上，需添加 self.kernel_size = kernel_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        embed_dim = config.pop('embed_dim')
        kernel_size = config.pop('kernel_size')
        return cls(embed_dim, kernel_size, **config)

    def call(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.pointwise(x)
        x = self.add([x, residual])
        return x

class PatchMixerModel(tf.keras.Model):
    def __init__(self, patch_size, embed_dim, num_blocks, kernel_size, mlp_dim, **kwargs):
        # 修改处：将 **kwargs 传给父类
        super(PatchMixerModel, self).__init__(**kwargs)
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        # 注意：这里传递参数给 PatchMixerBlock 时不需要改，因为它是内部调用的
        self.patch_mixer_blocks = [PatchMixerBlock(embed_dim, kernel_size) for _ in range(num_blocks)]
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.linear_head = tf.keras.layers.Dense(1)
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.mlp_dim = mlp_dim
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_blocks': self.num_blocks,
            'kernel_size': self.kernel_size,
            'mlp_dim': self.mlp_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        patch_size = config.pop('patch_size')
        embed_dim = config.pop('embed_dim')
        num_blocks = config.pop('num_blocks')
        kernel_size = config.pop('kernel_size')
        mlp_dim = config.pop('mlp_dim')
        return cls(patch_size, embed_dim, num_blocks, kernel_size, mlp_dim, **config)

    def call(self, x):
        x = self.patch_embedding(x)
        for block in self.patch_mixer_blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        linear_pred = self.linear_head(x)
        mlp_pred = self.mlp_head(x)
        return (linear_pred + mlp_pred) / 2


def calculate_technical_indicators(df, drop_nan=True):
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    bollinger = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bollinger['BBU_20_2.0_2.0']
    df['bb_middle'] = bollinger['BBM_20_2.0_2.0']
    df['bb_lower'] = bollinger['BBL_20_2.0_2.0']
    df['obv'] = ta.obv(df['close'], df['tick_volume'])
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['tick_volume'])
    if drop_nan:
        df = df.dropna()
        return df
    
    # df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.ffill().bfill()
    if df.isnull().values.any():
        print(f"填充后仍有{df.isnull().sum().sum()}个NaN，执行最终清理")
        df = df.dropna()
    return df

def plot_and_save_loss(history, filename='loss_curve.png', dpi=300):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss, 'r', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xlim(left=1)
    plt.ylim(bottom=0)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ 损失曲线已保存至 {filename} (DPI={dpi})")

def load_model_and_scaler(args, model_path="patchmixer_model.keras", scaler_path="patchmixer_scaler.joblib"):
    model = load_model(model_path, custom_objects={
        'PatchEmbedding': PatchEmbedding,
        'PatchMixerBlock': PatchMixerBlock,
        'PatchMixerModel': PatchMixerModel
    })
    # model = PatchMixerModel(patch_size=args.patch_size, embed_dim=args.embed_dim, num_blocks=args.num_blocks, kernel_size=args.kernel_size, mlp_dim=args.mlp_dim)
    # model.load_weights(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# 新增：辅助函数用于生成序列
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, 0]) # 假设目标变量（close）在第0列
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description="Train PatchMixer model for price prediction")
    parser.add_argument("--symbol", type=str, default="BTCUSDm", help="Trading symbol")
    parser.add_argument("--days", type=int, default=24*30, help="Number of days for historical data")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs") # 200
    parser.add_argument("--timeframe", type=str, default="M5")
    parser.add_argument("--save_folder", type=str, default=r"D:\aHYC\mt5\data\20260216")
    parser.add_argument("--model_save_folder", type=str, default=r"D:\aHYC\mt5\deepLearning\v23_mt5_patchMixer_260216\patchMixer_model_260216")
    parser.add_argument("--is_download_data", type=int, default=0)
    parser.add_argument("--is_only_test", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=10, help="Patch size for PatchMixer")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for PatchMixer")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of PatchMixer blocks")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for depthwise convolution")
    parser.add_argument("--mlp_dim", type=int, default=64, help="Dimension for MLP head")
    args = parser.parse_args()

    symbol = args.symbol
    days = args.days
    epochs = args.epochs
    timeframe = args.timeframe
    save_folder = args.save_folder
    model_save_folder = args.model_save_folder
    save_path = save_folder
    is_download_data = args.is_download_data
    is_only_test = args.is_only_test
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f"data_{days}_days.csv")
    if model_save_folder:
        os.makedirs(model_save_folder, exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    if tf.config.list_physical_devices('GPU'):
        print("GPU 可用，将用于训练。")
    else:
        print("GPU 不可用，训练将使用 CPU。")

    mt5_handler = MT5Handler()

    if not mt5.initialize():
        print("MT5初始化失败，请检查MT5终端")
        sys.exit(1)

    print(f"开始训练 - 品种: {symbol}, 时间: {pd.Timestamp.now()}")
    to_date = pd.Timestamp.now()
    from_date = to_date - pd.Timedelta(days=days)
    data_num_one_day = 0
    time_frame = None
    if timeframe == "M1":
        time_frame = mt5.TIMEFRAME_M1
        data_num_one_day = 60*24
    elif timeframe == "M5":
        time_frame = mt5.TIMEFRAME_M5
        data_num_one_day = 60/5 * 24
    else:
        raise ValueError(f"unknown timeframe：{timeframe}, please optimize code.")

    scaler = None
    X_train, X_test = None, None
    y_train, y_test = None, None

    if is_download_data:
        data = mt5_handler.download_data(symbol, timeframe=time_frame, days=days, save_path=save_path)

        if data.empty or len(data) < 60:
            print(f"无法获取足够数据（行数: {len(data)}），请检查品种或MT5连接")
            sys.exit(1)

        print(f"获取数据完成，行数: {len(data)}")
        data = calculate_technical_indicators(data, drop_nan=True)
        # feature_columns = ['close', 'tick_volume', 'rsi', 'atr', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower', 'typical_price', 'obv']
        # features = data[feature_columns].dropna()
        features = data.dropna()
        
        target_col = data.pop('close')
        data.insert(0, 'close', target_col)

        feature_columns = list(data.columns)
        np.save(os.path.join(save_folder, "feature_columns.npy"), feature_columns)

        features = data

        if features.isnull().any().any() or not np.all(np.isfinite(features.values)):
            print("特征数据包含NaN或非有限值，尝试填补")
            # features = features.fillna(method='ffill').fillna(method='bfill')
            features = features.ffill().bfill()
            if features.isnull().any().any():
                print("特征数据填补失败，训练中止")
                sys.exit(1)

        scaler = RobustScaler()
        # scaled_features = scaler.fit_transform(features)
        train_size = int(len(features) * 0.8)
        features_train = features.iloc[:train_size]
        features_test = features.iloc[train_size:]

        scaler = RobustScaler()
        # 仅在训练集上 fit
        features_train_scaled = scaler.fit_transform(features_train)
        # 在测试集上仅 transform
        features_test_scaled = scaler.transform(features_test)
        joblib.dump(scaler, os.path.join(save_folder, "patchmixer_scaler.joblib"))
        joblib.dump(scaler, os.path.join(model_save_folder, "patchmixer_scaler.joblib"))

        # 分别生成序列
        time_steps = 60
        X_train, y_train = create_sequences(features_train_scaled, time_steps)
        X_test, y_test = create_sequences(features_test_scaled, time_steps)

        if save_folder:
            np.save(os.path.join(save_folder, "X_train.npy"), X_train)
            np.save(os.path.join(save_folder, "X_test.npy"), X_test)
            np.save(os.path.join(save_folder, "y_train.npy"), y_train)
            np.save(os.path.join(save_folder, "y_test.npy"), y_test)
    else:
        # 加载数据
        if os.path.exists(os.path.join(save_folder, "X_train.npy")):
            X_train = np.load(os.path.join(save_folder, "X_train.npy"))
            X_test = np.load(os.path.join(save_folder, "X_test.npy"))
            y_train = np.load(os.path.join(save_folder, "y_train.npy"))
            y_test = np.load(os.path.join(save_folder, "y_test.npy"))
            scaler = joblib.load(os.path.join(save_folder, "patchmixer_scaler.joblib"))
            feature_columns = np.load(os.path.join(save_folder, "feature_columns.npy")).tolist()
        else:
            print("错误：未找到保存的训练数据文件，请先运行 is_download_data=1")
            sys.exit(1)

    print(f"训练集样本: {len(X_train)}, 验证集样本: {len(X_test)}")
    if not is_only_test:
        patch_size = args.patch_size
        embed_dim = args.embed_dim
        num_blocks = args.num_blocks
        kernel_size = args.kernel_size
        mlp_dim = args.mlp_dim
        time_steps = 60

        if time_steps % patch_size != 0:
            print(f"Error: time_steps {time_steps} must be divisible by patch_size {patch_size}")
            sys.exit(1)

        model = PatchMixerModel(patch_size, embed_dim, num_blocks, kernel_size, mlp_dim)
        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss='mse')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test), 
                            callbacks=[early_stopping, lr_scheduler, tensorboard_callback], verbose=1)
        
        plot_and_save_loss(history, os.path.join(model_save_folder, 'patchmixer_loss_curve.png'))

        y_pred = model.predict(X_test)
        y_pred_price = scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), len(feature_columns)-1))]))[:, 0]
        y_test_price = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), len(feature_columns)-1))]))[:, 0]
        mse = np.mean((y_pred_price - y_test_price) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_price - y_test_price))
        print(f"最终验证集 MSE: {mse:.2f}")
        print(f"最终验证集 RMSE: {rmse:.2f}")
        print(f"最终验证集 MAE: {mae:.2f}")

        # model.save(os.path.join(model_save_folder, "patchmixer_model.h5"))
        # model.save_weights(os.path.join(model_save_folder, "patchmixer_weights.h5"))
        model.save(os.path.join(model_save_folder, "patchmixer_model.keras"))
        print("模型和Scaler已保存")
    else:
        model_path = os.path.join(model_save_folder, "patchmixer_model.keras")
        scaler_path = os.path.join(save_folder, "patchmixer_scaler.joblib")
        if not os.path.exists(model_path):
             print(f"模型文件不存在: {model_path}")
             sys.exit(1)
        model, scaler = load_model_and_scaler(args, model_path, scaler_path)
        y_pred = model.predict(X_test)
        y_pred_price = scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), len(feature_columns)-1))]))[:, 0]
        y_test_price = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), len(feature_columns)-1))]))[:, 0]
        mse = np.mean((y_pred_price - y_test_price) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_price - y_test_price))
        print(f"最终验证集 MSE: {mse:.2f}")
        print(f"最终验证集 RMSE: {rmse:.2f}")
        print(f"最终验证集 MAE: {mae:.2f}")

if __name__ == "__main__":
    main()