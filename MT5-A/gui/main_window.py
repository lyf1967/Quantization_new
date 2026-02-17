from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from .account_tab import AccountTab
from .trading_tab import TradingTab
from .history_tab import HistoryTab
from .chart_tab import ChartTab
from mt5.mt5_handler import MT5Handler

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MT5量化交易系统")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QTabWidget::pane {
                border: 1px solid #2e2e3e;
                background: #252537;
            }
            QTabBar::tab {
                background: #2e2e3e;
                color: #cdd6f4;
                padding: 10px;
                border: none;
            }
            QTabBar::tab:selected {
                background: #45475a;
                color: #89b4fa;
            }
        """)
        
        self.mt5_handler = MT5Handler()
        
        self.tabs = QTabWidget()
        self.account_tab = AccountTab(self.mt5_handler)
        self.trading_tab = TradingTab(self.mt5_handler)
        self.history_tab = HistoryTab(self.mt5_handler)
        self.chart_tab = ChartTab(self.mt5_handler)
        
        self.tabs.addTab(self.account_tab, "账户信息")
        self.tabs.addTab(self.trading_tab, "自动交易")
        self.tabs.addTab(self.history_tab, "交易历史")
        self.tabs.addTab(self.chart_tab, "AI交易")
        
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 初始化交易统计
        self.trading_tab.update_trade_stats()