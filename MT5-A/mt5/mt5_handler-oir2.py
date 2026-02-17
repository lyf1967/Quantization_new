import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta
import importlib.util
import threading
import numpy as np

class MT5Handler:
    def __init__(self):
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，请检查MT5终端是否运行或网络连接是否正常")
            raise Exception("MT5初始化失败")
        self.trading_thread = None
        self.stop_trading_flag = False
        self.strategy_instance = None
        
    def get_account_info(self):
        account_info = mt5.account_info()
        if account_info:
            result = {
                'login': account_info.login,
                'balance': account_info.balance,
                'profit': account_info.profit
            }
            print(f"{datetime.now()}: 获取账户信息 - 账户ID: {result['login']}, 余额: {result['balance']}, 盈亏: {result['profit']}")
            return result
        print(f"{datetime.now()}: 无法获取账户信息，MT5可能未正确初始化")
        return None

    def get_initial_balance(self):
        """从历史记录中计算账户的总入金作为初始余额"""
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取历史记录")
            return 0.0

        to_date = datetime.now()
        from_date = to_date - timedelta(days=3650)

        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            print(f"{datetime.now()}: 未找到历史交易记录，时间范围：{from_date} 至 {to_date}")
            return 0.0

        initial_balance = 0.0
        for deal in deals:
            if deal.type == mt5.DEAL_TYPE_BALANCE and deal.profit > 0:
                initial_balance += deal.profit
                print(f"{datetime.now()}: 检测到入金记录 - 时间: {datetime.fromtimestamp(deal.time).isoformat()}, 金额: {deal.profit}")

        if initial_balance == 0.0:
            print(f"{datetime.now()}: 未找到入金记录，初始余额设为 0")
        else:
            print(f"{datetime.now()}: 计算初始余额为 {initial_balance}")
        return initial_balance

    def get_symbols(self):
        symbols = mt5.symbols_get()
        return [symbol.name for symbol in symbols]
        
    def get_current_price(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else 0.0
        
    def get_ohlc_data(self, symbol, timeframe=mt5.TIMEFRAME_M1, count=50):
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取K线数据")
            return pd.DataFrame()
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        # if rates is None or len(rates) < 31:
        if rates is None or len(rates) < 60:
            print(f"{datetime.now()}: 无法获取足够K线数据 - 品种: {symbol}, 获取行数: {len(rates) if rates is not None else 0}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']
        if not all(col in df.columns for col in required_columns):
            print(f"{datetime.now()}: K线数据缺少必要列 - 品种: {symbol}, 现有列: {list(df.columns)}")
            return pd.DataFrame()
        
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if df['close'].isnull().any() or not np.all(np.isfinite(df['close'])):
            print(f"{datetime.now()}: K线数据无效 - 品种: {symbol}, close列包含NaN或非有限值: {df['close'].tail(5).to_dict()}")
            df['close'] = df['close'].fillna(method='ffill')
            if df['close'].isnull().any():
                print(f"{datetime.now()}: 清洗后仍包含NaN - 品种: {symbol}")
                return pd.DataFrame()
        
        print(f"{datetime.now()}: 获取K线数据成功 - 品种: {symbol}, 行数: {len(df)}, 前两行: {df['close'].tail(2).to_dict()}")
        return df[required_columns]
        
    def get_history(self, from_date=None, to_date=None):
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取历史交易记录")
            return []

        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=30)

        print(f"{datetime.now()}: 获取历史交易记录，时间范围：{from_date} 至 {to_date}")
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            print(f"{datetime.now()}: 未找到历史交易记录，时间范围：{from_date} 至 {to_date}")
            return []

        history = []
        for deal in deals:
            history.append({
                'time': datetime.fromtimestamp(deal.time).isoformat(),
                'symbol': deal.symbol,
                'type': 'buy' if deal.type == mt5.DEAL_TYPE_BUY else 'sell',
                'volume': deal.volume,
                'profit': deal.profit
            })
        print(f"{datetime.now()}: 成功获取 {len(history)} 条历史交易记录")
        return history
        
    def calculate_atr(self, symbol, timeframe=mt5.TIMEFRAME_M1, period=14):
        data = self.get_ohlc_data(symbol, timeframe, period + 1)
        if data.empty:
            return 0.0
        data['tr'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low'] - data['close'].shift()).abs()
        ], axis=1).max(axis=1)
        return data['tr'].rolling(window=period).mean().iloc[-1]
        
    def symbol_info(self, symbol):
        """验证交易品种是否有效"""
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法验证交易品种")
            return None
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"{datetime.now()}: 无效交易品种: {symbol}")
        return symbol_info
        
    def execute_trade(self, symbol, volume, sl, tp, trade_type, dynamic_sl=False, dynamic_tp=False):
        symbol_info = self.symbol_info(symbol)
        if not symbol_info:
            print(f"{datetime.now()}: 无效交易品种 {symbol}")
            return False
            
        point = symbol_info.point
        print(f"{datetime.now()}: 品种 {symbol} 的点值: {point}")
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"{datetime.now()}: 无法获取当前价格 - 品种: {symbol}, 请检查MT5连接或市场状态")
            return False
            
        price = tick.ask if trade_type == 'buy' else tick.bid
        if price == 0.0:
            print(f"{datetime.now()}: 当前价格无效 - 品种: {symbol}, 价格: {price}")
            return False
        
        filling_mode = symbol_info.filling_mode
        if filling_mode & mt5.ORDER_FILLING_FOK:
            type_filling = mt5.ORDER_FILLING_FOK
        elif filling_mode & mt5.ORDER_FILLING_IOC:
            type_filling = mt5.ORDER_FILLING_IOC
        else:
            print(f"{datetime.now()}: 品种 {symbol} 不支持FOK或IOC填充类型，使用RETURN")
            type_filling = mt5.ORDER_FILLING_RETURN
        
        adjusted_tp = 0.0
        adjusted_sl = 0.0
        
        if not dynamic_tp and tp > 0.0:
            adjusted_tp = price + (tp * point) if trade_type == 'buy' else price - (tp * point)
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
            if self.strategy_instance and hasattr(self.strategy_instance, 'position_open_times'):
                ticket = result.order
                self.strategy_instance.position_open_times[ticket] = {
                    'open_time': datetime.now(),
                    'prev_price': price,
                    'entry_price': price,
                    'peak_price': price,
                    'trough_price': price
                }
                print(f"{datetime.now()}: 记录订单 {ticket} 开仓时间和初始价格")
                if hasattr(self.strategy_instance, 'update_position_entry'):
                    self.strategy_instance.update_position_entry(ticket, price)
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
            "tp": pos.tp
        } for pos in positions]
        
    def start_auto_trading(self, strategy_path, symbol, volume, sl, tp, dynamic_sl=False, dynamic_tp=False):
        self.stop_trading_flag = False
        
        if not self.symbol_info(symbol):
            print(f"{datetime.now()}: 自动交易启动失败 - 无效交易品种: {symbol}")
            return False
            
        try:
            if not os.path.exists(strategy_path):
                print(f"{datetime.now()}: 自动交易启动失败 - 策略文件 {strategy_path} 不存在")
                return False
                
            spec = importlib.util.spec_from_file_location("strategy", strategy_path)
            if spec is None:
                print(f"{datetime.now()}: 自动交易启动失败 - 无法加载策略文件")
                return False
                
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            
            if not hasattr(strategy_module, 'get_signal') and not hasattr(strategy_module, 'UltraHighFreqXAUUSD') and not hasattr(strategy_module, 'HighFreqXAUUSD') and not hasattr(strategy_module, 'RSIHighFreqXAUUSD'):
                print(f"{datetime.now()}: 自动交易启动失败 - 策略文件缺少get_signal函数或支持的类")
                return False
                
            def trading_loop():
                self.strategy_instance = None
                if hasattr(strategy_module, 'UltraHighFreqXAUUSD'):
                    self.strategy_instance = strategy_module.UltraHighFreqXAUUSD(handler=self, dynamic_sl_enabled=dynamic_sl, dynamic_tp_enabled=dynamic_tp)
                elif hasattr(strategy_module, 'HighFreqXAUUSD'):
                    self.strategy_instance = strategy_module.HighFreqXAUUSD(handler=self, dynamic_sl_enabled=dynamic_sl, dynamic_tp_enabled=dynamic_tp)
                elif hasattr(strategy_module, 'RSIHighFreqXAUUSD'):
                    self.strategy_instance = strategy_module.RSIHighFreqXAUUSD(handler=self, dynamic_sl_enabled=dynamic_sl, dynamic_tp_enabled=dynamic_tp)
                
                if (dynamic_sl or dynamic_tp) and self.strategy_instance and hasattr(self.strategy_instance, 'start_dynamic_sl_monitor'):
                    self.strategy_instance.start_dynamic_sl_monitor(symbol)
                
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
                        else:
                            signal = strategy_module.get_signal(data, symbol, handler=self, dynamic_sl_enabled=dynamic_sl, dynamic_tp_enabled=dynamic_tp)
                            
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