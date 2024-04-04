import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QProgressBar, QLabel, QLineEdit, QHBoxLayout, QCheckBox, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    def __init__(self, model, dataset, hyperparams):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.hyperparams = hyperparams

    def run(self):
        epochs = self.hyperparams['num_epochs']
        sleep_time = 100 if self.model == 'AlexNet' else 200 if self.model == 'ResNet' else 300
        for i in range(epochs):
            self.progress.emit(int((i + 1) / epochs * 100))
            QThread.msleep(sleep_time)

class TrainingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.default_hyperparams = {
            'AlexNet': {'batch_size': 64, 'learning_rate': 0.001, 'momentum': 0.9, 'weight_decay': 0.005, 'num_epochs': 20},
            'ResNet': {'batch_size': 64, 'learning_rate': 0.001, 'momentum': 0.9, 'weight_decay': 0.001, 'num_epochs': 20},
            'VGG16': {'batch_size': 16, 'learning_rate': 0.005, 'momentum': 0.9, 'weight_decay': 0.0, 'num_epochs': 20}
        }
        self.hyperparamsInputs = {}  # Keep track of current hyperparameter inputs
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Model Training GUI')
        self.setGeometry(100, 100, 800, 400)
        self.mainLayout = QVBoxLayout()  # Store the main layout as an attribute

        self.modelLabel = QLabel("Select Model:")
        self.modelSelect = QComboBox()
        self.modelSelect.addItems(['AlexNet', 'ResNet', 'VGG16'])
        self.datasetLabel = QLabel("Select Dataset:")
        self.datasetSelect = QComboBox()
        self.datasetSelect.addItems(['EcoNet', 'ImageNet2012'])

        self.mainLayout.addWidget(self.modelLabel)
        self.mainLayout.addWidget(self.modelSelect)
        self.mainLayout.addWidget(self.datasetLabel)
        self.mainLayout.addWidget(self.datasetSelect)

        self.updateHyperparamsInputs('AlexNet')  # Initialize with AlexNet

        self.useDefaultsCheckbox = QCheckBox("Use Default Hyperparameters")
        self.useDefaultsCheckbox.setChecked(True)
        self.useDefaultsCheckbox.toggled.connect(self.toggleHyperparameters)
        self.mainLayout.addWidget(self.useDefaultsCheckbox)

        self.progressBar = QProgressBar()
        self.mainLayout.addWidget(self.progressBar)

        self.startButton = QPushButton("Start Training")
        self.startButton.clicked.connect(self.startTraining)
        self.mainLayout.addWidget(self.startButton)

        centralWidget = QWidget()
        centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(centralWidget)

        self.modelSelect.currentTextChanged.connect(self.modelChanged)

    def updateHyperparamsInputs(self, model):
        # Remove old hyperparameter inputs
        for layout in self.hyperparamsInputs.values():
            for i in reversed(range(layout.count())): 
                layout.itemAt(i).widget().setParent(None)
        # Create and add new hyperparameter inputs
        self.hyperparamsInputs = self.createHyperparamsInputs(model)
        for layout in self.hyperparamsInputs.values():
            self.mainLayout.addLayout(layout)

    def createHyperparamsInputs(self, model):
        hyperparamsLayouts = {}
        for hp, default_value in self.default_hyperparams[model].items():
            label = QLabel(f"{hp.replace('_', ' ').capitalize()}:")
            inputField = QLineEdit(str(default_value))
            hbox = QHBoxLayout()
            hbox.addWidget(label)
            hbox.addWidget(inputField)
            hyperparamsLayouts[hp] = hbox
        return hyperparamsLayouts

    def modelChanged(self, model):
        self.updateHyperparamsInputs(model)
        self.toggleHyperparameters(self.useDefaultsCheckbox.isChecked())

    def toggleHyperparameters(self, checked):
        for layout in self.hyperparamsInputs.values():
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                widget.setDisabled(checked)

    def startTraining(self):
        model = self.modelSelect.currentText()
        dataset = self.datasetSelect.currentText()
        hyperparams = {hp: float(layout.itemAt(1).widget().text()) if hp not in ['batch_size', 'num_epochs'] else int(layout.itemAt(1).widget().text()) for hp, layout in self.hyperparamsInputs.items()}
        if self.useDefaultsCheckbox.isChecked():
            hyperparams = self.default_hyperparams[model]
        self.trainingThread = TrainingThread(model, dataset, hyperparams)
        self.trainingThread.progress.connect(self.progressBar.setValue)
        self.trainingThread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = TrainingGUI()
    gui.show()
    sys.exit(app.exec_())





