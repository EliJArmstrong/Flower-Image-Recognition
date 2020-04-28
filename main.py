""" ----------------------------------------------------------------------------
                            Student Name: Eli Armstrong
                            StudentID:    000930363
---------------------------------------------------------------------------- """

#   _____                           _           _
#  |_   _|                         | |         | |
#    | | _ __ ___  _ __   ___  _ __| |_ ___  __| |
#    | || '_ ` _ \| '_ \ / _ \| '__| __/ _ \/ _` |
#   _| || | | | | | |_) | (_) | |  | ||  __/ (_| |
#   \___/_| |_| |_| .__/ \___/|_|   \__\___|\__,_|
#                 | |
#                 |_|
#    _ _ _                    _
#   | (_) |                  (_)
#   | |_| |__  _ __ __ _ _ __ _  ___  ___
#   | | | '_ \| '__/ _` | '__| |/ _ \/ __|
#   | | | |_) | | | (_| | |  | |  __/\__ \
#   |_|_|_.__/|_|  \__,_|_|  |_|\___||___/
#
#

# PyQt5 IS what is used to create the GUI
from PyQt5 import QtCore, QtGui, QtWidgets

# Used to create the bar graph
from PyQt5.QtWidgets import QMainWindow

# PyQt5 Chart to create the bar graph for visualization
from PyQt5.QtChart import QChart, QChartView, QValueAxis, QBarCategoryAxis, \
    QBarSet, QBarSeries
from PyQt5.Qt import Qt
from PyQt5.QtGui import QPainter

# imageAI is what is used to prediction the uploaded image from training data.
# Model Training commented out to save on some overhead
from imageai.Prediction.Custom import CustomImagePrediction  # , ModelTraining

# This TensorFlow library (deprecation) is to silence some warnings in the shell
from tensorflow.python.util import deprecation

# The os is to get file paths mostly to get the model file and jason file with
# Prediction Flower names
import os

# This is used to keep the shell clean from some warnings from the TensorFlow
# library.
import logging

# This gives to get command-line arguments via the sys.argv property
import sys, random

# To check the prediction time of the predicted image
import time


#  ______             _ _      _   _               _____ _   _ _____
#  | ___ \           | (_)    | | (_)             |  __ \ | | |_   _|
#  | |_/ / __ ___  __| |_  ___| |_ _  ___  _ __   | |  \/ | | | | |
#  |  __/ '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \  | | __| | | | | |
#  | |  | | |  __/ (_| | | (__| |_| | (_) | | | | | |_\ \ |_| |_| |_
#  \_|  |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_|  \____/\___/ \___/
#
#   _____ _
#  /  __ \ |
#  | /  \/ | __ _ ___ ___
#  | |   | |/ _` / __/ __|
#  | \__/\ | (_| \__ \__ \
#   \____/_|\__,_|___/___/
#
# This class creates a GUI so the user can easy import a image of a flower
# and get a prediction based on the .h5 model file.
class PredictionGUI(object):
    #   _____ _                 _   _            _       _     _
    #  /  __ \ |               | | | |          (_)     | |   | |
    #  | /  \/ | __ _ ___ ___  | | | | __ _ _ __ _  __ _| |__ | | ___  ___
    #  | |   | |/ _` / __/ __| | | | |/ _` | '__| |/ _` | '_ \| |/ _ \/ __|
    #  | \__/\ | (_| \__ \__ \ \ \_/ / (_| | |  | | (_| | |_) | |  __/\__ \
    #   \____/_|\__,_|___/___/  \___/ \__,_|_|  |_|\__,_|_.__/|_|\___||___/
    #
    #

    # This variable is to hold the file path of the users selected flower image
    fileName = ""

    # Keeps track of the number of predictions in a session
    predictionCount = 0

    # Keeps a count of the number of roses the program predicts in a session.
    roseCount = 0

    # Keeps track of the count of sunflowers in a session
    sunflowerCount = 0

    # Keeps track of the number of tulips predicted in a session
    tulipCount = 0

    # Keeps track of the number of daisies predicted in a session
    daisyCount = 0

    # Keeps track of the number of dandelions predicted in a session.
    dandelionCount = 0

    #   _____ _                ______                _   _
    #  /  __ \ |               |  ___|              | | (_)
    #  | /  \/ | __ _ ___ ___  | |_ _   _ _ __   ___| |_ _  ___  _ __
    #  | |   | |/ _` / __/ __| |  _| | | | '_ \ / __| __| |/ _ \| '_ \
    #  | \__/\ | (_| \__ \__ \ | | | |_| | | | | (__| |_| | (_) | | | |
    #   \____/_|\__,_|___/___/ \_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|
    #
    #

    """
     This method sets up the UI's elements 

     :param self: this is a reference to this object.
     :param mainScene: this is a Q Main Window object.
    """

    def setupUI(self, mainScene):

        # Showing the GUI is being set up
        print("Setting up GUI program please wait.")
        self.setupMainWindow(mainScene)

        # A widget is creates and connect UI elements to the main scene
        self.setupWidget(mainScene)

        # Create the tab window
        self.setUptabs()

        # sets up the vertical layout box
        self.setUpVerticalLayout()

        # Sets up the buttons
        self.setupButtons()

        # Sets up the labels
        self.setupLabels()

        # Adds the tab with the objects to the tab view
        self.addTabsToTabView()

        # Connects the widget created above as the central (main) widget
        # of the GUI. Basically placed the UI elements in the gui window.
        mainScene.setCentralWidget(self.widget)

        # Set the current tab to the prediction tab.
        self.tabView.setCurrentIndex(0)

        # In the setup functions the UI object will get names by the set
        # object name functions. This connect the object via the given names
        QtCore.QMetaObject.connectSlotsByName(mainScene)

        # Connects the buttons to functions
        self.connectButtonsToFunctions()

        # The program can take a few seconds to set up the model for prediction
        # So, a message to tell the user that something is happening
        print("Setting up prediction model.")
        self.setupPredictionFlowerModel()

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method sets up the mainWindow. 

     :param self: this is a reference to this object.
     :param mainScene: this is a Q Main Window object.
    """

    def setupMainWindow(self, mainScene):
        # This gives the window is object name.
        mainScene.setObjectName("mainScene")

        # This sizes the window to 614 x 450
        mainScene.resize(614, 450)

        # This sets the label at the top window
        mainScene.setWindowTitle("Flower Recognition")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method sets up a QtWidget. 

     :param self: this is a reference to this object.
     :param mainScene: this is a Q Main Window object.
    """

    def setupWidget(self, mainScene):
        # Grabs the QWidget reference from the manScene
        self.widget = QtWidgets.QWidget(mainScene)

        # Gives the widget it object name
        self.widget.setObjectName("widget")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method sets up a the tabs. 

     :param self: this is a reference to this object.
     :param mainScene: this is a Q Main Window object.
    """

    def setUptabs(self):

        # This sets up the main container for the tabs
        self.tabView = QtWidgets.QTabWidget(self.widget)
        self.tabView.setGeometry(QtCore.QRect(0, 0, 611, 455))
        self.tabView.setObjectName("tabView")

        # The main tab the user first sees
        self.predictionTab = QtWidgets.QWidget()
        self.predictionTab.setObjectName("predictionTab")

        # The tab with the bar graph
        self.graphTab = QtWidgets.QWidget()
        self.graphTab.setObjectName("graphTab")

        # The table will session data
        self.sessionDataTab = QtWidgets.QWidget()
        self.sessionDataTab.setObjectName("sessionDataTab")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method added the tabs to the tab view. 

     :param self: this is a reference to this object.
    """

    def addTabsToTabView(self):
        # Adds the tabs to the tab view
        self.tabView.addTab(self.predictionTab, "")
        self.tabView.addTab(self.graphTab, "")
        self.tabView.addTab(self.sessionDataTab, "")

        # Sets the labels of the tabs
        self.tabView.setTabText(self.tabView.indexOf(self.predictionTab),
                                "Prediction Application")
        self.tabView.setTabText(self.tabView.indexOf(self.graphTab),
                                "Prediction Graph")
        self.tabView.setTabText(self.tabView.indexOf(self.sessionDataTab),
                                "Session Data")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method added the an Vbox to the sessions tab. 

     :param self: this is a reference to this object.
    """

    def setUpVerticalLayout(self):
        # creates a vertical layout in the sessions tab
        self.verticalLayoutWidget = QtWidgets.QWidget(self.sessionDataTab)

        # Where the vertical layout will be placed and the height and width
        # of the vertical layout.
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(5, 5, 591, 421))

        # Sets the vertical layout's object name
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        # Create the Q V Box Layout object
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)

        # Sets the margins of the vertical layout
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        # Sets the object name.
        self.verticalLayout.setObjectName("verticalLayout")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method sets up the GUI buttons. 

     :param self: this is a reference to this object.
    """

    def setupButtons(self):

        # These lines of code creates a font that the buttons will use for there
        # labels
        buttonFont = QtGui.QFont()
        # buttonFont.setFamily("MS Serif")
        # buttonFont.setFamily()
        buttonFont.setPointSize(10)
        buttonFont.setBold(True)
        buttonFont.setWeight(75)

        # The buttons height and width.
        buttonHeight = 101
        buttonWidth = 121

        # ----------------------------------------------------------------------
        # Select image button
        # ----------------------------------------------------------------------

        # Creates a push button from the class widget
        self.selectImage = QtWidgets.QPushButton(self.predictionTab)
        # Sets where the button will be place in the window and set the buttons
        # height and width
        self.selectImage.setGeometry(
            QtCore.QRect(160, 310, buttonWidth, buttonHeight))

        # Sets the font from above to this button
        self.selectImage.setFont(buttonFont)

        # Sets the objects name
        self.selectImage.setObjectName("selectImage")

        # Set the label text in the button
        self.selectImage.setText("Select an Image")

        # ----------------------------------------------------------------------
        # Image prediction button
        # ----------------------------------------------------------------------

        # Creates a push button from the class widget
        self.PredictBtn = QtWidgets.QPushButton(self.predictionTab)

        # Makes it so that the button is not enabled at start up
        self.PredictBtn.setEnabled(False)

        # Sets where the button will be place in the window and set the buttons
        # height and width
        self.PredictBtn.setGeometry(
            QtCore.QRect(330, 310, buttonWidth, buttonHeight))

        # Sets the font from above to this button
        self.PredictBtn.setFont(buttonFont)

        # Sets the objects name
        self.PredictBtn.setObjectName("PredictBtn")

        # Set the label text in the button
        self.PredictBtn.setText("Predict Image")

        # ----------------------------------------------------------------------
        # Graph button: Shows a graph
        # ----------------------------------------------------------------------

        # Creates the  graph button as a Q push button and adds it to the graph
        # tab.
        self.graphButton = QtWidgets.QPushButton(self.graphTab)

        # Where is 2D space the button will be in the tab. Also, the height and
        # width is set
        self.graphButton.setGeometry(QtCore.QRect(10, 10, 350, 61))

        # Sets the object name.
        self.graphButton.setObjectName("graphButton")

        # Makes the button non enabled at the start of the gui
        self.graphButton.setEnabled(False)

        # the text of the button
        self.graphButton.setText("Click Here for a graph of confidence scores")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method sets up the GUI labels. 

     :param self: this is a reference to this object.
    """

    def setupLabels(self):

        # Sets the font that will be used for the label
        LabelFont = QtGui.QFont()
        LabelFont.setPointSize(12)
        LabelFont.setBold(True)
        LabelFont.setWeight(75)

        # A basic font
        BasicFont = QtGui.QFont()
        BasicFont.setPointSize(10)

        # ----------------------------------------------------------------------
        # Image label
        # Note: Labels in PyQt5 have the ability to show images this is what is
        #       used to show the user the image that they selected.
        # ----------------------------------------------------------------------

        # Creates a Q Label
        self.imageLbl = QtWidgets.QLabel(self.predictionTab)

        # Sets the position of the label. The height and width as well.
        self.imageLbl.setGeometry(QtCore.QRect(30, 10, 551, 241))

        # This creates a black outline around the labels box.
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)

        # This sets the text of the label to empty.
        self.imageLbl.setText("")

        # This gives the object it name.
        self.imageLbl.setObjectName("imageLbl")

        # Sets the labels contents to the center.
        self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)

        # ----------------------------------------------------------------------
        # prediction label
        # Note: This is the main way the program communicate to the user.
        # ----------------------------------------------------------------------

        # creates a Q Label
        self.predictionLbl = QtWidgets.QLabel(self.predictionTab)

        # Sets the position of the label. The height and width as well.
        self.predictionLbl.setGeometry(QtCore.QRect(30, 250, 551, 51))

        # Sets the font for the label
        self.predictionLbl.setFont(LabelFont)

        # This sets the alignment of the text to the center of the labels box
        self.predictionLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.predictionLbl.setObjectName("predictionLbl")

        # Sets the initial text of the label
        self.predictionLbl.setText("Click the select an image button to start")

        # ----------------------------------------------------------------------
        # time title label
        # ----------------------------------------------------------------------

        # creates and adds the label to the prediction tab
        self.timeTitleLbl = QtWidgets.QLabel(self.predictionTab)

        # Sets the position of the label. The height and width as well.
        self.timeTitleLbl.setGeometry(QtCore.QRect(10, 310, 141, 31))

        # Sets the font
        self.timeTitleLbl.setFont(BasicFont)

        # This sets the alignment of the text to the center of the labels box
        self.timeTitleLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.timeTitleLbl.setObjectName("timeTitleLbl")

        # Sets the initial text of the label
        self.timeTitleLbl.setText("Prediction Time")

        # ----------------------------------------------------------------------
        # time title label
        # ----------------------------------------------------------------------

        # creates and adds the label to the prediction tab
        self.timeLbl = QtWidgets.QLabel(self.predictionTab)

        # Sets the position of the label. The height and width as well.
        self.timeLbl.setGeometry(QtCore.QRect(6, 340, 141, 31))

        # Sets the font
        self.timeLbl.setFont(BasicFont)

        # This sets the alignment of the text to the center of the labels box
        self.timeLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.timeLbl.setObjectName("timeLbl")

        # Sets the initial text of the label
        self.timeLbl.setText("0.000 seconds")

        # ----------------------------------------------------------------------
        # Session title label
        # ----------------------------------------------------------------------

        # creates and adds the label to the vertical Layout
        self.sessionTitleLbl = QtWidgets.QLabel(self.verticalLayoutWidget)

        # Sets the position of the label. The height and width as well.
        self.sessionTitleLbl.setGeometry(QtCore.QRect(10, 80, 611, 81))

        # Sets the font
        largeFont = QtGui.QFont()
        largeFont.setPointSize(32)
        self.sessionTitleLbl.setFont(largeFont)

        # This sets the alignment of the text to the center of the labels box
        self.sessionTitleLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.sessionTitleLbl.setObjectName("sessionTitleLbl")

        # Sets the initial text of the label
        self.sessionTitleLbl.setText("Session Data")

        # Adds the label to the vertical layout
        self.verticalLayout.addWidget(self.sessionTitleLbl)

        # ----------------------------------------------------------------------
        # Number of predictions label
        # ----------------------------------------------------------------------

        # Creates and adds the label to the vertical Layout widget
        self.numOfPredictionsMadeLbl = QtWidgets.QLabel(
            self.verticalLayoutWidget)

        # Sets the font
        self.numOfPredictionsMadeLbl.setFont(BasicFont)

        # This sets the alignment of the text to the center of the labels box
        self.numOfPredictionsMadeLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.numOfPredictionsMadeLbl.setObjectName("numOfPredictionsMadeLbl")

        # Sets the initial text of the label
        self.numOfPredictionsMadeLbl.setText("Number of predictions: 0")

        # Adds the label to the vertical layout
        self.verticalLayout.addWidget(self.numOfPredictionsMadeLbl)

        # ----------------------------------------------------------------------
        # Rose label
        # ----------------------------------------------------------------------

        # Creates and adds the label to the vertical Layout widget
        self.roseLbl = QtWidgets.QLabel(self.verticalLayoutWidget)

        # Sets the font
        self.roseLbl.setFont(BasicFont)

        # This sets the alignment of the text to the center of the labels box
        self.roseLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.roseLbl.setObjectName("roseLbl")

        # Sets the initial text of the label
        self.roseLbl.setText("Number of predicted roses: 0")

        # Adds the label to the vertical layout
        self.verticalLayout.addWidget(self.roseLbl)

        # ----------------------------------------------------------------------
        # Tulip label
        # ----------------------------------------------------------------------

        # Creates and adds the label to the vertical Layout widget
        self.tulipLbl = QtWidgets.QLabel(self.verticalLayoutWidget)

        # Sets the font
        self.tulipLbl.setFont(BasicFont)

        # This sets the alignment of the text to the center of the labels box
        self.tulipLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.tulipLbl.setObjectName("tulipLbl")

        # Sets the initial text of the label
        self.tulipLbl.setText("Number of predicted tulips: 0")

        # Adds the label to the vertical layout
        self.verticalLayout.addWidget(self.tulipLbl)

        # ----------------------------------------------------------------------
        # Sunflower label
        # ----------------------------------------------------------------------

        # Creates and adds the label to the vertical Layout widget
        self.sunflowerLbl = QtWidgets.QLabel(self.verticalLayoutWidget)

        # Sets the font
        self.sunflowerLbl.setFont(BasicFont)

        # This sets the alignment of the text to the center of the labels box
        self.sunflowerLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.sunflowerLbl.setObjectName("sunflowerLbl")

        # Sets the initial text of the label
        self.sunflowerLbl.setText("Number of predicted sunflowers: 0")

        # Adds the label to the vertical layout
        self.verticalLayout.addWidget(self.sunflowerLbl)

        # ----------------------------------------------------------------------
        # Dandelion label
        # ----------------------------------------------------------------------

        # Creates and adds the label to the vertical Layout widget
        self.dandelionLbl = QtWidgets.QLabel(self.verticalLayoutWidget)

        # Sets the font
        self.dandelionLbl.setFont(BasicFont)

        # This sets the alignment of the text to the center of the labels box
        self.dandelionLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.dandelionLbl.setObjectName("dandelionLbl")

        # Sets the initial text of the label
        self.dandelionLbl.setText("Number of predicted dandelions: 0")

        # Adds the label to the vertical layout
        self.verticalLayout.addWidget(self.dandelionLbl)

        # ----------------------------------------------------------------------
        # Daisy label
        # ----------------------------------------------------------------------

        # Creates and adds the label to the vertical Layout widget
        self.daisyLbl = QtWidgets.QLabel(self.verticalLayoutWidget)

        # Sets the font
        self.daisyLbl.setFont(BasicFont)

        # This sets the alignment of the text to the center of the labels box
        self.daisyLbl.setAlignment(QtCore.Qt.AlignCenter)

        # Sets the objects name
        self.daisyLbl.setObjectName("daisyLbl")

        # Sets the initial text of the label
        self.daisyLbl.setText("Number of predicted daisies: 0")

        # Adds the label to the vertical layout
        self.verticalLayout.addWidget(self.daisyLbl)

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method connects the buttons to function when clicked. 

     :param self: this is a reference to this object.
    """

    def connectButtonsToFunctions(self):
        # Calls the place image function when the select image button is clicked
        self.selectImage.clicked.connect(self.placeImage)

        # Calls the get Prediction function when the select image button is clicked
        self.PredictBtn.clicked.connect(self.getPrediction)

        # calls the show graph function
        self.graphButton.clicked.connect(self.showGraph)

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method connects the buttons to function when clicked. 

     :param self: this is a reference to this object.
    """

    def placeImage(self):

        # Gives the user a prompt to help the user get started
        self.predictionLbl.setText("Click the select an image button to start")

        # This opens an file dialog window and will only show png, jpg, jpeg
        # and folders.
        # This function returns the file path and the 4 parameter of the
        # function this is being placed in _ because this program does not need it.
        fName, _ = QtWidgets.QFileDialog.getOpenFileName(None,
                                                         "Select an image", "",
                                                         "Image Files (*.png *.jpg *.jpeg)")

        # checks to see if a file was found from the above function.
        if fName:
            # When a new image is selected this makes the graph button
            # unclickable so the user will not be able to see old data.
            self.graphButton.setEnabled(False)

            # Creates a Q pix map with the selected file.
            pictureMapObject = QtGui.QPixmap(fName)

            # This scales the image to the the image label size.
            pictureMapObject = pictureMapObject.scaled(self.imageLbl.width(),
                                                       self.imageLbl.height(),
                                                       QtCore.Qt.KeepAspectRatio)

            # Set the image to the image label.
            self.imageLbl.setPixmap(pictureMapObject)

            # Sets the image path to the fileName variable.
            self.fileName = fName

            # Once an image is selected an loaded the prediction button will then be enabled.
            self.PredictBtn.setEnabled(True)

            # Changes the user prompt to show what action the user should take next.
            self.predictionLbl.setText("Click the Predict Image button.")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method sets up the prediction model that will be used to check user
     selected the image. 

     :param self: this is a reference to this object.
    """

    def setupPredictionFlowerModel(self):

        # This gets the path of the .py file or the exe file
        exPath = os.getcwd()

        # Creates a custom image prediction object.
        self.predict = CustomImagePrediction()

        # Tells the custom image prediction object that it will be using
        # self.predict.setModelTypeAsResNet()
        # self.predict.setModelTypeAsInceptionV3()
        self.predict.setModelTypeAsDenseNet()

        # Check to see if there is an .h5 file (The model file). If the file
        # is not found then a message is shown to the user saying that the
        # file is missing. Both buttons are disabled to prevent any
        # unexpected crash of the program.
        if (os.path.isfile(
                os.path.join(exPath, "DenseNet_flower_model_85.h5")) is False):
            print("Could not file a training model file (.h5)")
            self.PredictBtn.setEnabled(False)
            self.selectImage.setEnabled(False)
            self.predictionLbl.setText("Missing the model file (.h5)")

        else:
            # If the file the .h5 file exists then the .h5 model path will
            # be set to it.
            self.predict.setModelPath(
                os.path.join(exPath, "DenseNet_flower_model_85.h5"))
            # Checks to see if the .json model class file is found. If it
            # is found the json path is loaded with the file.
            # If the file is not found the user will be shown a message.
            if os.path.isfile(os.path.join(exPath, "flower_model_class.json")):
                self.predict.setJsonPath(
                    os.path.join(exPath, "flower_model_class.json"))
                self.predict.loadModel(num_objects=5)
            else:
                # The message that is shown if the json file is not found.
                # Both buttons are disabled to prevent any unexpected crash
                # of the program.
                print("Could not file a model class file (.jason)")
                self.PredictBtn.setEnabled(False)
                self.selectImage.setEnabled(False)
                self.predictionLbl.setText(
                    "Missing the model class file (.json)")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method takes the checks to see if the user selected an a file.
     If the user selected file is not empty Then the image is given a confidence
     and what flower the computer thinks it is. This is display to the user via
     a label on the GUI

     :param self: this is a reference to this object.
    """

    def getPrediction(self):

        # Checks to see if the user image if has been selected.
        if self.fileName != "":

            # Starts the prediction time counter
            start = time.time()

            # The predict image function returns to arrays one for the
            # prediction name and one for the probability of being correct
            # (I call it a confidence score). The result count being one means
            # that the function will only return one predict flower name and
            # the probability of the guess being correct.
            self.prediction, self.probability = self.predict.predictImage(
                self.fileName, result_count=5)

            # Ends the prediction time counter
            end = time.time()

            # time data
            print(time.asctime(time.localtime(start)))

            # Sets the time label to show the user how long it took to predict the image
            self.timeLbl.setText("{0:.3f} seconds".format((end - start)))

            # The prediction is be presented to the user via a label on the GUI.
            self.predictionLbl.setText(
                "With a confidence score of {0:.3f}%. This is a {1}.".format(
                    self.probability[0], self.prediction[0].capitalize()))

            # Makes sure the labels are updated
            self.tabView.setCurrentIndex(1)
            self.tabView.setCurrentIndex(0)

            # calls the update flower count method.
            self.updateFlowerCount(self.prediction[0])

            # Just a console print for debugging
            print(str(self.prediction[0]) + " : " + str(self.probability[0]))

            # enables the graphButton
            self.graphButton.setEnabled(True)

        # If the use did not select an image and some how the user was to by
        # pass the the button not being enabled. Then this message will show to the user.
        else:
            self.predictionLbl.setText("No Image file selected.")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method shows a bar graph based on model output data. 

     :param self: this is a reference to this object.
    """

    def showGraph(self):

        # Creates the bar graph object with prediction data
        self.flowerGraph = BarGraph(self.prediction, self.probability)

        # Shows the window and graph in a pop out.
        self.flowerGraph.show()

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method updates the flower counts based on the passed flower name.
     Then updates the label in the session data tab.

     :param self: this is a reference to this object.
     :param flowerName: the flower name to be updated.
    """

    def updateFlowerCount(self, flowerName):

        # Update the prediction count
        self.predictionCount = self.predictionCount + 1

        # updates the prediction count label in the session data tab
        self.numOfPredictionsMadeLbl.setText(
            "Number of predictions: " + str(self.predictionCount))

        # if-else to update the correct flower type label
        if flowerName == "daisy":

            # updates the daisy count and sets the daisy label in the sessions data tab
            self.daisyCount = self.daisyCount + 1
            self.daisyLbl.setText(
                "Number of predicted daisies: " + str(self.daisyCount))

        elif flowerName == "dandelion":

            # updates the dandelion count and sets the daisy label in the sessions data tab
            self.dandelionCount = self.dandelionCount + 1
            self.dandelionLbl.setText(
                "Number of predicted dandelions: " + str(self.dandelionCount))

        elif flowerName == "rose":

            # updates the rose count and sets the daisy label in the sessions data tab
            self.roseCount = self.roseCount + 1
            self.roseLbl.setText(
                "Number of predicted roses: " + str(self.roseCount))

        elif flowerName == "sunflower":

            # updates the sunflower count and sets the daisy label in the sessions data tab
            self.sunflowerCount = self.sunflowerCount + 1
            self.sunflowerLbl.setText(
                "Number of predicted sunflowers: " + str(self.sunflowerCount))

        elif flowerName == "tulip":

            # updates the tulip count and sets the daisy label in the sessions data tab
            self.tulipCount = self.tulipCount + 1
            self.tulipLbl.setText(
                "Number of predicted tulips: " + str(self.tulipCount))

        else:

            # If no flower name matches this will print
            print("no flower of that name.")

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------

    """
     This method returns a a string of text to be used for a log file.

     :param self: this is a reference to this object.
    """

    def logText(self):
        return "\n\n" + str(
            time.asctime(time.localtime(time.time()))) + "\n\nSession Data \n" + \
               "Number of predictions: " + str(self.predictionCount) + "\n" + \
               "Number of predicted daisies: " + str(self.daisyCount) + "\n" + \
               "Number of predicted dandelions: " + str(
            self.dandelionCount) + "\n" + \
               "Number of predicted roses: " + str(self.roseCount) + "\n" + \
               "Number of predicted sunflowers: " + str(
            self.sunflowerCount) + "\n" + \
               "Number of predicted tulips: " + str(self.tulipCount) + "\n\n"

    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------


#  ______              _____                 _
#  | ___ \            |  __ \               | |
#  | |_/ / __ _ _ __  | |  \/_ __ __ _ _ __ | |__
#  | ___ \/ _` | '__| | | __| '__/ _` | '_ \| '_ \
#  | |_/ / (_| | |    | |_\ \ | | (_| | |_) | | | |
#  \____/ \__,_|_|     \____/_|  \__,_| .__/|_| |_|
#                                     | |
#                                     |_|
#   _____ _
#  /  __ \ |
#  | /  \/ | __ _ ___ ___
#  | |   | |/ _` / __/ __|
#  | \__/\ | (_| \__ \__ \
#   \____/_|\__,_|___/___/
#
#


'''
This creates a window that will be a bar graph
'''


class BarGraph(QMainWindow):
    '''
        The parameterized constructor for the bar graph class

        :param self: this is a reference to this object.
        :param predictions: This is an array of flower names.
                            This will be the x axis
        :param probabilities: This is the array of percentages that is the
                              y axis
    '''

    def __init__(self, predictions, probabilities):
        # calls the parent (QMainWindow) constructor
        super().__init__()

        # Size of the window
        self.resize(800, 600)

        # The title of the window
        self.setWindowTitle("Flower Confidence score graph")

        # The array to hold flower sets
        self.flowerSets = []

        # A Q bar series object
        self.flowerSeries = QBarSeries()

        # The for loop add Q bar sets and probabilities to the Q bsr series object.
        for i in range(5):
            self.flowerSets.append(QBarSet(
                predictions[i] + " ({0:.3f}%)".format(probabilities[i])))
            self.flowerSets[i].append(probabilities[i])
            self.flowerSeries.append(self.flowerSets[i])

        # creates the char that will be the bar graph chart (also the base class of all qt charts)
        flowerChart = QChart()

        # added the series object to the chart object
        flowerChart.addSeries(self.flowerSeries)

        # Adds a title to the graph
        flowerChart.setTitle('Confidence score percentages by flower type')

        # Gives the graph a nice animation when opened
        flowerChart.setAnimationOptions(QChart.SeriesAnimations)

        # The next two line label the x axis
        axisX = QBarCategoryAxis()
        axisX.append(["Flower Types"])

        # Gives the y axis its values
        axisY = QValueAxis()
        axisY.setRange(0, probabilities[0])

        # added the axises to the chart
        flowerChart.addAxis(axisX, Qt.AlignBottom)
        flowerChart.addAxis(axisY, Qt.AlignLeft)

        # Makes sure teh legend is visible
        flowerChart.legend().setVisible(True)

        # Places the  legend to the bottom of the chart
        flowerChart.legend().setAlignment(Qt.AlignBottom)

        # places the chart in a chart view
        flowerChartView = QChartView(flowerChart)

        # Adds the chart to the main widget
        self.setCentralWidget(flowerChartView)


#   _____         _       _
#  |_   _|       (_)     (_)
#    | |_ __ __ _ _ _ __  _ _ __   __ _
#    | | '__/ _` | | '_ \| | '_ \ / _` |
#    | | | | (_| | | | | | | | | | (_| |
#    \_/_|  \__,_|_|_| |_|_|_| |_|\__, |
#                                  __/ |
#                                 |___/
#  ___  ___     _   _               _
#  |  \/  |    | | | |             | |
#  | .  . | ___| |_| |__   ___   __| |
#  | |\/| |/ _ \ __| '_ \ / _ \ / _` |
#  | |  | |  __/ |_| | | | (_) | (_| |
#  \_|  |_/\___|\__|_| |_|\___/ \__,_|
#
#

'''
This is the retraining part of the model training program.
commented out to save on some over head.
'''
# def training():
# The current dir path
#    exePath = os.getcwd()
# modeltraining class
#    mt = ModelTraining()
# set the model to be trained to denseNet
#    mt.setModelTypeAsDenseNet()
# The folder to contain the flower images within other folder that are the
# different types of flowers.
#    mt.setDataDirectory("flowers")
# Trains the model with 200 epochs and uses the last model to train future installation
# while keeping old training model data
#    mt.trainModel(num_objects=5, num_experiments=200, enhance_data=True, batch_size=16, show_network_summary=True,
#              transfer_from_model=os.path.join(exePath, "DenseNet_flower_model_85.h5"))

#   _____ _   _ _____
#  |  __ \ | | |_   _|
#  | |  \/ | | | | |
#  | | __| | | | | |
#  | |_\ \ |_| |_| |_
#   \____/\___/ \___/
#
#
#  ___  ___     _   _               _
#  |  \/  |    | | | |             | |
#  | .  . | ___| |_| |__   ___   __| |
#  | |\/| |/ _ \ __| '_ \ / _ \ / _` |
#  | |  | |  __/ |_| | | | (_) | (_| |
#  \_|  |_/\___|\__|_| |_|\___/ \__,_|
#
#
'''
This is to run the gui part of the application
'''


def gui():
    # This disables some warning to keep the shell as clean as possible.
    logging.disable(logging.WARNING)

    # This also disables some warning to keep the shell as clean as possible.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # This creates the Qt application an receives any command-line arguments
    # if need be.
    qTApp = QtWidgets.QApplication(sys.argv)

    # This creates a Qt Main Window for the GUI app.
    openingScene = QtWidgets.QMainWindow()

    # This creates a instance of the class created above.
    gui = PredictionGUI()

    # This sets up the GUI with the class above using the Qt window create above
    gui.setupUI(openingScene)

    # This presents the GUI to the User.
    openingScene.show()

    # This does any GUI clean up after the gui is shown.
    exe = qTApp.exec_()

    # Creates a log file will session data.
    logFile = open("log.txt", "a+")
    logFile.write(gui.logText())
    logFile.close()

    # clean up an close
    sys.exit(exe)


#  ___  ___      _
#  |  \/  |     (_)
#  | .  . | __ _ _ _ __
#  | |\/| |/ _` | | '_ \
#  | |  | | (_| | | | | |
#  \_|  |_/\__,_|_|_| |_|
#
# This is the start of the program.

if __name__ == "__main__":
    # run the prediction GUI app
    gui()

    # This is to run the training retraining program
    # its commented out to save on over head
    # training()
