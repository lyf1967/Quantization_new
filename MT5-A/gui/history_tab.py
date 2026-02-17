from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from mt5.mt5_handler import MT5Handler

class HistoryTab(QWidget):
    def __init__(self, mt5_handler: MT5Handler):
        super().__init__()
        self.mt5_handler = mt5_handler
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 设置暗黑风格
        self.setStyleSheet("""
            QWidget {
                background-color: #252537;
                color: #cdd6f4;
            }
            QTableWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                gridline-color: #45475a;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #45475a;
                color: #89b4fa;
                padding: 5px;
                border: 1px solid #585b70;
            }
        """)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["时间", "交易品种", "交易类型", "交易量", "盈亏"])
        
        layout.addWidget(self.history_table)
        self.setLayout(layout)
        self.update_history()
        
    def update_history(self):
        history = self.mt5_handler.get_history()
        self.history_table.setRowCount(len(history))
        
        for i, trade in enumerate(history):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(trade['time'])))
            self.history_table.setItem(i, 1, QTableWidgetItem(trade['symbol']))
            self.history_table.setItem(i, 2, QTableWidgetItem('买入' if trade['type'] == 'buy' else '卖出'))
            self.history_table.setItem(i, 3, QTableWidgetItem(str(trade['volume'])))
            self.history_table.setItem(i, 4, QTableWidgetItem(f"{trade['profit']:.2f}"))