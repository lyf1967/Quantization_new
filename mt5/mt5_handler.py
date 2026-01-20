import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta
import importlib.util
import threading
import numpy as np
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from strategies.v7_gallon_strategy import *


class MT5Handler:
    def __init__(self):
        if not mt5.initialize():  # 初始化
            print(f"{datetime.now()}: MT5初始化失败，请检查MT5终端是否运行或网络连接是否正常")
            raise Exception("MT5初始化失败")
        self.trading_thread = None  # 交易线程
        self.stop_trading_flag = False  # 停止交易flag
        self.strategy_instance = None   # 交易策略实例

    # 获取账户信息
    def get_account_info(self):
        account_info = mt5.account_info()  # 获取mt5账号信息
        if account_info:
            result = {
                'login': account_info.login,   # 账户登录ID
                'balance': account_info.balance,  # 账户余额（初始资金）
                'profit': account_info.profit   # 当前浮动盈亏
            }
            print(f"{datetime.now()}: 获取账户信息 - 账户ID: {result['login']}, 余额: {result['balance']}, 盈亏: {result['profit']}")
            return result
        print(f"{datetime.now()}: 无法获取账户信息，MT5可能未正确初始化")
        return None

    # 获取初始余额
    def get_initial_balance(self):
        """从历史记录中计算账户的总入金作为初始余额"""
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取历史记录")
            return 0.0

        to_date = datetime.now()  # 当前时间
        # 计算一个日期（to_date）减去 3650 天（约10年）后的新日期（from_date）
        from_date = to_date - timedelta(days=3650)

        deals = mt5.history_deals_get(from_date, to_date)  # 获取历史交易信息
        if not deals:
            print(f"{datetime.now()}: 未找到历史交易记录，时间范围：{from_date} 至 {to_date}")
            return 0.0

        initial_balance = 0.0
        for deal in deals:  # 遍历所有交易单子
            # 账户出入金操作（DEAL_TYPE_BALANCE） # deal.profit > 0：过滤入金操作（出金时 profit 为负值）
            if deal.type == mt5.DEAL_TYPE_BALANCE and deal.profit > 0:
                initial_balance += deal.profit  # 入金金额
                print(f"{datetime.now()}: 检测到入金记录 - 时间: {datetime.fromtimestamp(deal.time).isoformat()}, 金额: {deal.profit}")

        if initial_balance == 0.0:
            print(f"{datetime.now()}: 未找到入金记录，初始余额设为 0")
        else:
            print(f"{datetime.now()}: 计算初始余额为 {initial_balance}")
        return initial_balance

    # 获取交易品种
    def get_symbols(self):
        symbols = mt5.symbols_get() # 获取当前交易账户可访问的全部金融品种列表
        return [symbol.name for symbol in symbols]  # 仅保留名称信息

    # 获取当前价格
    def get_current_price(self, symbol):
        tick = mt5.symbol_info_tick(symbol)  # 获取指定交易品种的最新报价（Tick 数据）
        # bid（买入价）：做空时的入场价/做多时的离场参考价
        return tick.bid if tick else 0.0
    
    def download_data(self, symbol, timeframe=mt5.TIMEFRAME_M5, days=365, save_path=None):
        # 初始化 MT5
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取K线数据")
            return pd.DataFrame()

        # 设置下载的时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # 转换为 MT5 可识别的时间戳格式
        start_time = int(start_time.timestamp())
        end_time = int(end_time.timestamp())

        # 分批下载数据
        batch_size = 30  # 每次下载 11*30 天的数据
        all_data = []

        current_start = start_time
        while current_start < end_time:
            # 计算本次下载的结束时间
            batch_end_time = min(current_start + batch_size * 86400, end_time) # 一天86400s
            current_start_dt = datetime.fromtimestamp(current_start)
            batch_end_time_dt = datetime.fromtimestamp(batch_end_time)

            # 下载指定时间范围内的 K 线数据
            rates = mt5.copy_rates_range(symbol, timeframe, current_start_dt, batch_end_time_dt)
            # rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(count))
            # 检查数据是否成功下载
            if rates is None or len(rates) == 0:
                print(f"{datetime.now()}: 无法获取K线数据 - 品种: {symbol}, 时间: {datetime.fromtimestamp(current_start)} - {datetime.fromtimestamp(batch_end_time)}")
                # break
                current_start = batch_end_time
                continue
            
            if len(rates) < 5:
                print(f"len(rates): {len(rates)}")
                current_start = batch_end_time
                continue

            # 将数据转换为 DataFrame 并设置时间索引
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            all_data.append(df)

            # 更新下一次下载的起始时间
            current_start = batch_end_time

        # 合并所有分批数据并去重
        if all_data:
            final_df = pd.concat(all_data)
            # 移除基于时间索引的重复行，保留第一次出现的数据
            final_df = final_df[~final_df.index.duplicated(keep='first')]
            # 按时间排序
            final_df.sort_index(inplace=True)
            print(f"下载完成，数据行数: {len(final_df)}")
            # 如果指定了保存路径，则保存数据
            if save_path:
                final_df.to_csv(save_path)
                print(f"数据已保存至: {save_path}")
            return final_df
        else:
            print(f"{datetime.now()}: 无法获取任何K线数据 - 品种: {symbol}")
            return pd.DataFrame()



    # 获取历史数据
    def get_ohlc_data(self, symbol, timeframe=mt5.TIMEFRAME_M5, count=50):
        print(f"timeframe:{timeframe}")
        # time.sleep(100)
        # mt5.TIMEFRAME_M1 表示1分钟线
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取K线数据")
            return pd.DataFrame()

        # 0：从当前最新K线开始获取（0表示最新位置）； count：获取的K线数量（默认为50条）
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count) # 获取历史K线数据，返回为numpy.array
        if rates is None or len(rates) < 31:
            print(f"{datetime.now()}: 无法获取足够K线数据 - 品种: {symbol}, 获取行数: {len(rates) if rates is not None else 0}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        # 将时间戳（Unix秒）转为可读时间（unit='s'）
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # 设置时间为索引
        df.set_index('time', inplace=True)
        
        # 验证 close 列
        # open（开盘价）、high（最高价）、low（最低价）、close（收盘价）
        # tick_volume 记录的是特定时间段内价格跳动的次数，而非成交量（如合约手数）;Tick是价格的最小变动单位，例如期货合约价格从 1000.5 变为 1000.6 即构成一次跳动（tick）
        # real_volume表示真正的成交量
        # all(...)：当所有必需列均存在时返回True，否则返回False
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_columns):
            print(f"{datetime.now()}: K线数据缺少必要列 - 品种: {symbol}, 现有列: {list(df.columns)}")
            return pd.DataFrame()
        
        # 清洗数据
        # 转化为数值类型； errors='coerce'：非数值内容转为NaN（如空字符串、文本）
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        # .isnull().any()：检查是否存在NaN ； np.isfinite()：排除±inf（无穷大）或NaN
        if df['close'].isnull().any() or not np.all(np.isfinite(df['close'])):
            print(f"{datetime.now()}: K线数据无效 - 品种: {symbol}, close列包含NaN或非有限值: {df['close'].tail(5).to_dict()}")
            df['close'] = df['close'].fillna(method='ffill')  # ffill（前向填充）：用最近有效值覆盖NaN
            if df['close'].isnull().any():
                print(f"{datetime.now()}: 清洗后仍包含NaN - 品种: {symbol}")
                return pd.DataFrame()
        
        print(f"{datetime.now()}: 获取K线数据成功 - 品种: {symbol}, 行数: {len(df)}, 前两行: {df['close'].tail(2).to_dict()}")
        return df[required_columns]

    # 获取历史成交记录信息
    def get_history(self, from_date=None, to_date=None):
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取历史交易记录")
            return []

        if to_date is None:
            to_date = datetime.now() # 当前时间
        if from_date is None:
            from_date = to_date - timedelta(days=30)  # 30天前

        print(f"{datetime.now()}: 获取历史交易记录，时间范围：{from_date} 至 {to_date}")
        deals = mt5.history_deals_get(from_date, to_date) # 获取历史交易记录
        if not deals:
            print(f"{datetime.now()}: 未找到历史交易记录，时间范围：{from_date} 至 {to_date}")
            return []

        history = []
        for deal in deals:
            history.append({
                # datetime.fromtimestamp()：将时间戳转为本地时区的datetime对象
                # .isoformat()：生成ISO 8601标准时间字符串（如 "2025-06-08T14:30:45"）
                'time': datetime.fromtimestamp(deal.time).isoformat(),  # deal.time：MT5返回的Unix时间戳（秒级整数）
                'symbol': deal.symbol,  # deal.symbol：交易品种代码（如 "EURUSD"）
                'type': 'buy' if deal.type == mt5.DEAL_TYPE_BUY else 'sell',  # mt5.DEAL_TYPE_BUY = 0（买入）
                'volume': deal.volume, # volume：成交手数（浮点数，如 0.1 手）
                'profit': deal.profit  # profit：该笔成交的盈亏金额（含手续费和库存费）
            })
        print(f"{datetime.now()}: 成功获取 {len(history)} 条历史交易记录")
        return history

    # 计算金融品种的平均真实波幅（ATR），是衡量市场波动性的核心指标; 用于衡量市场风险
    def calculate_atr(self, symbol, timeframe=mt5.TIMEFRAME_M1, period=14):
        # period + 1 表示K线数量
        data = self.get_ohlc_data(symbol, timeframe, period + 1) # 获取历史数据 # period + 1：需多取1根K线（因TR计算依赖前一日收盘价）
        if data.empty:
            return 0.0
        data['tr'] = pd.concat([
            data['high'] - data['low'],  # 当日最高价-最低价波幅
            (data['high'] - data['close'].shift()).abs(), # 当日最高价-前日收盘价波幅  # shift() 获取前一日收盘价（close列向下平移1位)
            (data['low'] - data['close'].shift()).abs()  # 当日最低-前日收盘价波幅
        ], axis=1).max(axis=1) # 取三者最大值
        # .rolling(window=period)创建滑动窗口对象,window=period：窗口大小（如 period=14 表示14个数据的窗口）
        # .mean()对窗口内数据计算算术平均值
        # .iloc[-1] → 取最新结果
        return data['tr'].rolling(window=period).mean().iloc[-1]
        
    def symbol_info(self, symbol):
        """验证交易品种是否有效"""
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法验证交易品种")
            return None
        symbol_info = mt5.symbol_info(symbol)  # 用于获取指定交易品种（如 EURUSD、XAUUSD）的详细规格和属性信息
        if symbol_info is None:
            print(f"{datetime.now()}: 无效交易品种: {symbol}")
        return symbol_info

    # 执行MT5市价单交易（开仓）
    def execute_trade(self, symbol, volume, sl, tp, trade_type, dynamic_sl=False, dynamic_tp=False):
        # symbol：交易品种（如EURUSD）
        # volume：交易手数
        # sl / tp：止损 / 止盈点数（非价格）
        # trade_type：方向（buy / sell）
        # dynamic_sl / dynamic_tp：是否动态设置止损止盈（默认关闭）

        symbol_info = self.symbol_info(symbol)
        if not symbol_info:
            print(f"{datetime.now()}: 无效交易品种 {symbol}")
            return False
            
        point = symbol_info.point  # 品种价格的最小变动单位（如 EURUSD 为 0.0001）。
        print(f"{datetime.now()}: 品种 {symbol} 的点值: {point}")
        
        tick = mt5.symbol_info_tick(symbol) # 用于获取指定交易品种的最新报价（Tick 数据），即市场实时买卖价格和交易量等动态信息。
        if tick is None:
            print(f"{datetime.now()}: 无法获取当前价格 - 品种: {symbol}, 请检查MT5连接或市场状态")
            return False

        # 买单用卖价(ask)，卖单用买价(bid)（遵循做市商规则）
        price = tick.ask if trade_type == 'buy' else tick.bid
        if price == 0.0:
            print(f"{datetime.now()}: 当前价格无效 - 品种: {symbol}, 价格: {price}")
            return False

        # 获取交易品种（如 EURUSD）的订单执行填充模式（Order Filling Mode），决定订单成交规则
        filling_mode = symbol_info.filling_mode
        if filling_mode & mt5.ORDER_FILLING_FOK:  # mt5.ORDER_FILLING_FOK: 全部成交或取消 --- 要求订单立即全部成交，否则取消
            type_filling = mt5.ORDER_FILLING_FOK
        elif filling_mode & mt5.ORDER_FILLING_IOC: # mt5.ORDER_FILLING_IOC: 立即成交或取消 --- 允许部分成交，剩余部分取消
            type_filling = mt5.ORDER_FILLING_IOC
        else:
            print(f"{datetime.now()}: 品种 {symbol} 不支持FOK或IOC填充类型，使用RETURN")
            type_filling = mt5.ORDER_FILLING_RETURN  # mt5.ORDER_FILLING_RETURN: 返回剩余订单 --- 部分成交后，剩余订单保留在订单簿（交易所市场常用
        
        adjusted_tp = 0.0
        adjusted_sl = 0.0
        
        if not dynamic_tp and tp > 0.0: # 非动态止盈 且 止盈点数>0
            # # 计算止盈价：buy=开仓价+点数×点值, sell=开仓价-点数×点值
            adjusted_tp = price + (tp * point) if trade_type == 'buy' else price - (tp * point)
            # 方向验证：止盈需高于开仓价（buy）或低于开仓价（sell）
            if trade_type == 'buy' and adjusted_tp <= price:
                print(f"{datetime.now()}: 止盈点位 {adjusted_tp:.2f} 小于等于开仓价格 {price:.2f}，设置无效")
                adjusted_tp = 0.0
            elif trade_type == 'sell' and adjusted_tp >= price:
                print(f"{datetime.now()}: 止盈点位 {adjusted_tp:.2f} 大于等于开仓价格 {price:.2f}，设置无效")
                adjusted_tp = 0.0
            print(f"{datetime.now()}: 计算止盈 - 开仓价格: {price:.2f}, 用户设置TP: {tp}, 点值: {point}, 调整后TP: {adjusted_tp:.2f}")
        
        if not dynamic_sl and sl > 0.0:
            adjusted_sl = price - (sl * point) if trade_type == 'buy' else price + (sl * point)
            if trade_type == 'buy' and adjusted_sl >= price:
                print(f"{datetime.now()}: 止损点位 {adjusted_sl:.2f} 大于等于开仓价格 {price:.2f}，设置无效")
                adjusted_sl = 0.0
            elif trade_type == 'sell' and adjusted_sl <= price:
                print(f"{datetime.now()}: 止损点位 {adjusted_sl:.2f} 小于等于开仓价格 {price:.2f}，设置无效")
                adjusted_sl = 0.0
            print(f"{datetime.now()}: 计算止损 - 开仓价格: {price:.2f}, 用户设置SL: {sl}, 点值: {point}, 调整后SL: {adjusted_sl:.2f}")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if trade_type == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": adjusted_sl,
            "tp": adjusted_tp,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"{datetime.now()}: 下单成功 - {trade_type.upper()} {symbol}, 价格: {price:.2f}, 止损: {adjusted_sl:.2f}, 止盈: {adjusted_tp:.2f}")
            # 记录开仓时间供动态止损使用
            if self.strategy_instance and hasattr(self.strategy_instance, 'position_open_times'):
                ticket = result.order
                self.strategy_instance.position_open_times[ticket] = {'open_time': datetime.now(), 'prev_price': price}
                print(f"{datetime.now()}: 记录订单 {ticket} 开仓时间和初始价格")
            return True
        else:
            print(f"{datetime.now()}: 下单失败 - {trade_type.upper()} {symbol}, 错误代码: {result.retcode}")
            return False
            
    def close_specific_position(self, symbol, ticket):
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            print(f"{datetime.now()}: 无持仓可平仓 - {symbol}")
            return False
            
        for pos in positions:
            if pos.ticket == ticket:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"{datetime.now()}: 平仓成功 - {symbol}, 票号: {pos.ticket}")
                    # 移除动态止损的开仓时间记录
                    if self.strategy_instance and hasattr(self.strategy_instance, 'position_open_times'):
                        if pos.ticket in self.strategy_instance.position_open_times:
                            del self.strategy_instance.position_open_times[pos.ticket]
                            print(f"{datetime.now()}: 移除订单 {pos.ticket} 的开仓时间记录")
                    return True
                else:
                    print(f"{datetime.now()}: 平仓失败 - {symbol}, 票号: {pos.ticket}, 错误代码: {result.retcode}")
                    return False
        print(f"{datetime.now()}: 未找到指定订单 - 票号: {ticket}")
        return False
        
    def close_position(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            print(f"{datetime.now()}: 无持仓可平仓 - {symbol}")
            return False
            
        for pos in positions:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"{datetime.now()}: 平仓成功 - {symbol}, 票号: {pos.ticket}")
                # 移除动态止损的开仓时间记录
                if self.strategy_instance and hasattr(self.strategy_instance, 'position_open_times'):
                    if pos.ticket in self.strategy_instance.position_open_times:
                        del self.strategy_instance.position_open_times[pos.ticket]
                        print(f"{datetime.now()}: 移除订单 {pos.ticket} 的开仓时间记录")
            else:
                print(f"{datetime.now()}: 平仓失败 - {symbol}, 错误代码: {result.retcode}")
        return True
        
    def get_open_positions(self):
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取持仓信息")
            return []
        positions = mt5.positions_get()
        if not positions:
            return []
        return [{
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
            "open_price": pos.price_open,
            "current_price": pos.price_current,
            "profit": pos.profit,
            "sl": pos.sl,
            "tp": pos.tp,
            "volume": pos.volume
        } for pos in positions]
        
    def start_auto_trading(self, strategy_path, symbol, volume, sl, tp, dynamic_sl=False, dynamic_tp=False):
        self.stop_trading_flag = False
        
        if not self.symbol_info(symbol):
            print(f"{datetime.now()}: 自动交易启动失败 - 无效交易品种: {symbol}")
            return False
            
        try:
            # if not os.path.exists(strategy_path):
            #     print(f"{datetime.now()}: 自动交易启动失败 - 策略文件 {strategy_path} 不存在")
            #     return False
            #
            # spec = importlib.util.spec_from_file_location("strategy", strategy_path)
            # if spec is None:
            #     print(f"{datetime.now()}: 自动交易启动失败 - 无法加载策略文件")
            #     return False
                
            # strategy_module = importlib.util.module_from_spec(spec)
            # spec.loader.exec_module(strategy_module)

            # if not hasattr(strategy_module, 'get_signal') and not hasattr(strategy_module, 'UltraHighFreqXAUUSD') and not hasattr(strategy_module, 'HighFreqXAUUSD'):
            #     print(f"{datetime.now()}: 自动交易启动失败 - 策略文件缺少get_signal函数或支持的类")
            #     return False
                
            def trading_loop():
                # self.strategy_instance = None
                self.strategy_instance = RSIHighFreqXAUUSD(handler=self, dynamic_sl_enabled=dynamic_sl,
                                                                           dynamic_tp_enabled=dynamic_tp,
                                                                           current_initial_volume=volume)

                # if hasattr(strategy_module, 'UltraHighFreqXAUUSD'):
                #     self.strategy_instance = strategy_module.UltraHighFreqXAUUSD(handler=self, dynamic_sl_enabled=dynamic_sl, dynamic_tp_enabled=dynamic_tp, current_initial_volume=volume)
                # elif hasattr(strategy_module, 'HighFreqXAUUSD'):
                #     self.strategy_instance = strategy_module.HighFreqXAUUSD(handler=self, dynamic_sl_enabled=dynamic_sl, dynamic_tp_enabled=dynamic_tp, current_initial_volume=volume)
                # elif hasattr(strategy_module, 'RSIHighFreqXAUUSD'):
                #     self.strategy_instance = strategy_module.RSIHighFreqXAUUSD(handler=self, dynamic_sl_enabled=dynamic_sl, dynamic_tp_enabled=dynamic_tp, current_initial_volume=volume)
                
                # 如果启用动态止损或动态止盈，启动监控
                if (dynamic_sl or dynamic_tp) and self.strategy_instance and hasattr(self.strategy_instance, 'start_dynamic_sl_monitor'):
                    self.strategy_instance.start_dynamic_sl_monitor(symbol, current_initial_volume=volume)
                
                while not self.stop_trading_flag:
                    try:
                        data = self.get_ohlc_data(symbol)
                        if data.empty:
                            print(f"{datetime.now()}: 无法获取K线数据 - 品种: {symbol}")
                            threading.Event().wait(1)
                            continue
                        print(f"{datetime.now()}: 调用策略 - 品种: {symbol}, 数据行数: {len(data)}")
                        if self.strategy_instance:
                            signal = self.strategy_instance.get_signal(data, symbol)
                        # else:
                        #     signal = strategy_module.get_signal(data, symbol, handler=self, dynamic_sl_enabled=dynamic_sl, dynamic_tp_enabled=dynamic_tp, current_initial_volume=volume)
                            
                        if signal == 'buy':
                            if self.execute_trade(symbol, volume, sl, tp, 'buy', dynamic_sl=dynamic_sl, dynamic_tp=dynamic_tp):
                                account_info = self.get_account_info()
                                print(f"{datetime.now()}: 账户状态 - 余额: {account_info['balance']:.2f}, 浮动盈亏: {account_info['profit']:.2f}")
                        elif signal == 'sell':
                            if self.execute_trade(symbol, volume, sl, tp, 'sell', dynamic_sl=dynamic_sl, dynamic_tp=dynamic_tp):
                                account_info = self.get_account_info()
                                print(f"{datetime.now()}: 账户状态 - 余额: {account_info['balance']:.2f}, 浮动盈亏: {account_info['profit']:.2f}")
                        
                        threading.Event().wait(1)
                    except Exception as e:
                        print(f"{datetime.now()}: 交易错误：{e}")
            
            self.trading_thread = threading.Thread(target=trading_loop)
            self.trading_thread.start()
            print(f"{datetime.now()}: 自动交易启动成功 - 品种: {symbol}, 策略: {os.path.basename(strategy_path)}")
            return True
            
        except Exception as e:
            print(f"{datetime.now()}: 自动交易启动失败 - 错误: {str(e)}")
            return False
        
    def stop_auto_trading(self):
        try:
            self.stop_trading_flag = True
            if self.trading_thread:
                self.trading_thread.join()
                self.trading_thread = None
                # 停止动态止损监控
                if self.strategy_instance and hasattr(self.strategy_instance, 'stop_dynamic_sl_monitor'):
                    self.strategy_instance.stop_dynamic_sl_monitor()
                self.strategy_instance = None
                print(f"{datetime.now()}: 自动交易停止成功")
                return True
            else:
                print(f"{datetime.now()}: 自动交易停止成功 - 无运行中的交易线程")
                return True
        except Exception as e:
            print(f"{datetime.now()}: 自动交易停止失败 - 错误: {str(e)}")
            return False