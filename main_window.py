from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QTabWidget, QMenuBar, QWidget, QHBoxLayout, QGridLayout
from PySide6.QtWidgets import QLineEdit, QGroupBox, QCheckBox, QDialog, QErrorMessage, QPushButton
from PySide6.QtGui import QAction, QIntValidator, QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from detect_trigger import DetectTriggerWindow
from locate import LocateWindow
from projects import Project
from util import NonLinLoc
import os
from copy import copy, deepcopy

class CentralWindow(QWidget):
    def __init__(self, project):
        super().__init__()

        self.init_ui(project)
    
    def init_ui(self, project):
        layout = QVBoxLayout()

        # Add a tab widget
        tab_widget = QTabWidget(self)
        layout.addWidget(tab_widget)

        # Add tabs to the tab widget
        tab1 = DetectTriggerWindow(project)
        tab2 = LocateWindow(project)
        # tab1 = QWidget()
        # tab2 = QWidget()
        # tab3 = QWidget()
        # tab4 = QWidget()

        tab_widget.addTab(tab1, 'Detect/Trigger')
        tab_widget.addTab(tab2, 'Locate')
        # tab_widget.addTab(tab2, 'Locate')
        # tab_widget.addTab(tab3, 'NonLinLoc')
        # tab_widget.addTab(tab4, 'Fault Plan Solution')

        # # Add content to the tabs
        # self.add_tab_content(tab1, 'Content for Tab 1')
        # self.add_tab_content(tab2, 'Content for Tab 2')
        # self.add_tab_content(tab3, 'Content for Tab 3')
        # self.add_tab_content(tab4, 'Content for Tab 4')

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self, folder_path, file_browser_app):
        super().__init__()

        # self.file_browser_app = file_browser_app
        self.folder_path = folder_path

        self.read_project_settings()

        self.init_ui()

    def init_ui(self):
        self.create_actions()
        self.create_menu()
        self.central_window = CentralWindow(self.project)

        self.setCentralWidget(self.central_window)
        self.setWindowTitle('Main Window')
        self.setMinimumSize(800, 400)
        # self.showMaximized()

    def read_project_settings(self):
        self.project = Project(self.folder_path)

    # def createMenus(self):

    #     fileMenu = menuBar().addMenu(tr("File"))
    #     fileMenu.addAction(newAct)
    #     fileMenu.addAction(openAct)
    #     fileMenu.addAction(saveAct)

    def create_actions(self):
        print("MAKING ACTUIONS")
        self.define_nll_genericsettings_action = QAction("Generic Settings", self)
        self.define_nll_genericsettings_action.triggered.connect(self._define_nll_genericsettings)  

        self.define_nll_vmodel_action = QAction("Velocity Model", self)
        self.define_nll_vmodel_action.triggered.connect(self._define_nll_vmodel)  

        self.define_nll_ttimetables_action = QAction("Generate Travel Time Tables", self)
        self.define_nll_ttimetables_action.triggered.connect(self._generate_ttime_tables)  

        # self.define_nll_NLLoc_action = QAction("NLLoc Settings", self)
        # self.define_nll_NLLoc_action.triggered.connect(self._define_nll_NLLoc)  

        self.view_stations_action = QAction("View Stations")
        self.view_stations_action.triggered.connect(self._view_stations)

    def create_menu(self):
        print("MAKING MENU")

        nonlinloc_menu = self.menuBar().addMenu("NonLinLoc")
        nonlinloc_menu.addAction(self.define_nll_genericsettings_action)
        nonlinloc_menu.addAction(self.define_nll_vmodel_action)
        nonlinloc_menu.addAction(self.define_nll_ttimetables_action)
        # nonlinloc_menu.addAction(self.define_nll_NLLoc_action)

        stations_menu = self.menuBar().addMenu("Stations")
        stations_menu.addAction(self.view_stations_action)

    def _define_nll_genericsettings(self):
        print("NLL SETTIGNS", self.project.nll_settings.nll_locsearch["numScatter"])  

        self.nll_settings_window = NLLGenericSettings(self.project)
        self.nll_settings_window.accepted.connect(self.set_new_nll_settings)
        self.nll_settings_window.exec()
        
    def set_new_nll_settings(self):
        print("NEW NLL SETTINGS")
        self.project.nll_settings = deepcopy(self.nll_settings_window.nll_settings)
        self.nll_settings_window.close()
        self.project.nll_settings.write_yaml(self.folder_path)
    def _define_nll_vmodel(self):
        print("NLL VMODEL")   
    # def _define_nll_NLLoc(self):
    #     print("NLL NLLoc")     
    def _view_stations(self):
        print("VIEW STATIONS")
        print("NOT IMPLEMENTED")
        return
    def _generate_ttime_tables(self):
        self.project.nll_settings.make_traveltime_tables(self.project.filepath)


class NLLGenericSettings(QDialog):
    def __init__(self, project):
        super().__init__()

        self.project = project
        self.nll_settings = deepcopy(self.project.nll_settings)
        self.init_ui()
        self.setup_connections()

        self.setModal(True)


    
    def init_ui(self):
        # add a button for OK
        self.ok_button = QPushButton("OK")
        self.ok_button.setAutoDefault(True)
        self.ok_button.setDefault(True)

        generic_settings_box = QGroupBox("Generic Settings", self)
        generic_settings_box.setCheckable(False)
        vbox = QVBoxLayout()
        self.message_flag_widget = QLineEdit(str(self.nll_settings.message_flag), self, 
                                             validator=QIntValidator(0, 5))
        self.random_number_widget = QLineEdit(str(self.nll_settings.random_number), self,
                                              validator=QIntValidator())
        self.transform_widget = QLineEdit(self.nll_settings.transform, self)
        self.transform_widget.setReadOnly(True)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Message Flag"))
        hbox.addWidget(self.message_flag_widget)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Random Number"))
        hbox.addWidget(self.random_number_widget)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Transform"))
        hbox.addWidget(self.transform_widget)
        vbox.addLayout(hbox)
        generic_settings_box.setLayout(vbox)
        
        ## Vel2Grid Settings
        vel2grid_settings_box = QGroupBox("Vel2Grid Settings", self)
        vel2grid_settings_box.setCheckable(False)
        vbox = QVBoxLayout()
        self.vg_out_widget = QLineEdit(str(self.nll_settings.vg_out), self)
        box, self.vg_grid_widgets = self.showGrid("VGGRID", self.nll_settings.vg_grid)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("VGOUT"))
        hbox.addWidget(self.vg_out_widget)
        vbox.addLayout(hbox)
        vbox.addWidget(box)
        vel2grid_settings_box.setLayout(vbox)
        print("HELP")

        ## Grid2Time Settings
        grid2time_settings_box = QGroupBox("Grid2Time Settings", self)
        grid2time_settings_box.setCheckable(False)
        vbox = QVBoxLayout()
        self.gt_ttimeFileRoot_widget = QLineEdit(str(self.nll_settings.gt_ttimeFileRoot), self)
        self.gt_out_widget = QLineEdit(str(self.nll_settings.gt_out), self)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Model Directory"))
        hbox.addWidget(self.gt_ttimeFileRoot_widget)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Out Root"))
        hbox.addWidget(self.gt_out_widget)
        vbox.addLayout(hbox)
        grid2time_settings_box.setLayout(vbox)      

        print("HELP2")
        ## NonLinLoc Settings
        self.nll_signature_widget = QLineEdit(str(self.nll_settings.nll_signature), self)
        self.locsearch_widgets = {"type" : QLineEdit(str(self.nll_settings.nll_locsearch["type"]), self),
                                  "initNumCells_x" : QLineEdit(str(self.nll_settings.nll_locsearch["initNumCells_x"]), self, validator=QIntValidator(bottom=1)),
                                  "initNumCells_y" : QLineEdit(str(self.nll_settings.nll_locsearch["initNumCells_y"]), self, validator=QIntValidator(bottom=1)),
                                  "initNumCells_z" : QLineEdit(str(self.nll_settings.nll_locsearch["initNumCells_z"]), self, validator=QIntValidator(bottom=1)),
                                  "minNodeSize" : QLineEdit(str(self.nll_settings.nll_locsearch["minNodeSize"]), self, validator=QIntValidator(bottom=1)),
                                  "maxNumNodes" : QLineEdit(str(self.nll_settings.nll_locsearch["maxNumNodes"]), self, validator=QIntValidator(bottom=1)),
                                  "numScatter" : QLineEdit(str(self.nll_settings.nll_locsearch["numScatter"]), self, validator=QIntValidator(bottom=1)),
                                  "useStationsDensity" : QCheckBox("useStationsDensity", self),
                                  "stopOnMinNodeSize" : QCheckBox("stopOnMinNodeSize", self)}
        self.locsearch_widgets["type"].setReadOnly(True)
        if self.nll_settings.nll_locsearch["useStationsDensity"]:
            self.locsearch_widgets["useStationsDensity"].setChecked(True)
        if self.nll_settings.nll_locsearch["stopOnMinNodeSize"]:
            self.locsearch_widgets["stopOnMinNodeSize"].setChecked(True)
        locsearch_box = QGroupBox("LOCSEARCH", self)
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Type"))
        hbox.addWidget(self.locsearch_widgets["type"])
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        for key in ["initNumCells_x","initNumCells_y","initNumCells_z"]:
            hbox.addWidget(QLabel(key))
            hbox.addWidget(self.locsearch_widgets[key])
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        for key in ["minNodeSize","maxNumNodes","numScatter"]:
            hbox.addWidget(QLabel(key))
            hbox.addWidget(self.locsearch_widgets[key])
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.locsearch_widgets["useStationsDensity"])
        hbox.addWidget(self.locsearch_widgets["stopOnMinNodeSize"])
        vbox.addLayout(hbox)
        locsearch_box.setLayout(vbox)

        self.locmeth_widgets, self.locmeth_lineedit_widgets = self.get_locmeth_widgets()
        locmeth_box = QGroupBox("LOCMETH", self)
        locmeth_box_vbox = QGridLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Method"))
        hbox.addWidget(self.locmeth_widgets["method"])
        locmeth_box_vbox.addLayout(hbox, 0, 0)
        i, j, count = 1,0,1
        for key in ["maxDistStaGrid","minNumberPhases", "maxNumberPhases", 
                    "minNumberSphases","maxNum3DGridMemory", "VpVsRatio",
                    "minDistStaGrid", "iRejectDuplicateArrivals"]:
            print(key,i, j)
            locmeth_box_vbox.addWidget(self.locmeth_widgets[key], j,i)
            count += 1
            if count%3 == 0:
                j += 1
                i = 0
            else:
                i += 1
        locmeth_box.setLayout(locmeth_box_vbox)

        self.locgau_sigmatime_widget = QLineEdit(str(self.nll_settings.nll_locgau["SigmaTime"]), self, validator=QDoubleValidator(bottom=0.0))
        self.locgau_corrlen_widget = QLineEdit(str(self.nll_settings.nll_locgau["CorrLen"]), self, validator=QDoubleValidator(bottom=0.0))

        self.locgau2_sigmatfraction_widget = QLineEdit(str(self.nll_settings.nll_locgau2["SigmaTfraction"]), self, validator=QDoubleValidator(bottom=0.0))
        self.locgau2_sigmatmin_widget = QLineEdit(str(self.nll_settings.nll_locgau2["SigmaTmin"]), self, validator=QDoubleValidator(bottom=0.0))
        self.locgau2_sigmatmax_widget = QLineEdit(str(self.nll_settings.nll_locgau2["SigmaTmax"]), self, validator=QDoubleValidator(bottom=0.0))

        box, self.nll_grid_widgets = self.showGrid("LOCGRID", self.nll_settings.nll_locgrid)

        self.nll_locqual2err0_widget = QLineEdit(str(self.nll_settings.nll_locqual2err[0]), self, validator=QDoubleValidator(bottom=0.0))
        self.nll_locqual2err1_widget = QLineEdit(str(self.nll_settings.nll_locqual2err[1]), self, validator=QDoubleValidator(bottom=0.0))
        self.nll_locqual2err2_widget = QLineEdit(str(self.nll_settings.nll_locqual2err[2]), self, validator=QDoubleValidator(bottom=0.0))
        self.nll_locqual2err3_widget = QLineEdit(str(self.nll_settings.nll_locqual2err[3]), self, validator=QDoubleValidator(bottom=0.0))
        self.nll_locqual2err4_widget = QLineEdit(str(self.nll_settings.nll_locqual2err[4]), self, validator=QDoubleValidator(bottom=0.0))

        signature_layout = QHBoxLayout()
        signature_layout.addWidget(QLabel("Signature"))
        signature_layout.addWidget(self.nll_signature_widget)

        locgau_box = QGroupBox()
        locgau_layout = QHBoxLayout()
        locgau_layout.addWidget(QLabel("SigmaTime"))
        locgau_layout.addWidget(self.locgau_sigmatime_widget)
        locgau_layout.addWidget(QLabel("CorrLen"))
        locgau_layout.addWidget(self.locgau_corrlen_widget)
        locgau_layout.addStretch(1)
        locgau_box.setLayout(locgau_layout)

        locgau2_box = QGroupBox()
        locgau2_layout = QHBoxLayout()
        locgau2_layout.addWidget(QLabel("SigmaTfraction"))
        locgau2_layout.addWidget(self.locgau2_sigmatfraction_widget)
        locgau2_layout.addWidget(QLabel("SigmaTmin"))
        locgau2_layout.addWidget(self.locgau2_sigmatmin_widget)
        locgau2_layout.addWidget(QLabel("SigmaTmax"))
        locgau2_layout.addWidget(self.locgau2_sigmatmax_widget)
        locgau2_layout.addStretch(1)
        locgau2_box.setLayout(locgau2_layout)

        locqual2err_box = QGroupBox("LOCQUAL2ERR")
        locqual2err_box.setCheckable(False)
        locqual2err_layout = QHBoxLayout()
        locqual2err_layout.addWidget(QLabel("0"))
        locqual2err_layout.addWidget(self.nll_locqual2err0_widget)
        locqual2err_layout.addWidget(QLabel("1"))
        locqual2err_layout.addWidget(self.nll_locqual2err1_widget)
        locqual2err_layout.addWidget(QLabel("2"))
        locqual2err_layout.addWidget(self.nll_locqual2err2_widget)
        locqual2err_layout.addWidget(QLabel("3"))
        locqual2err_layout.addWidget(self.nll_locqual2err3_widget)
        locqual2err_layout.addWidget(QLabel("4"))
        locqual2err_layout.addWidget(self.nll_locqual2err4_widget)
        locqual2err_layout.addStretch(1)
        locqual2err_box.setLayout(locqual2err_layout)



        nonlinloc_settings_box_layout = QVBoxLayout()
        nonlinloc_settings_box_layout.addLayout(signature_layout)
        nonlinloc_settings_box_layout.addWidget(locsearch_box)
        nonlinloc_settings_box_layout.addWidget(locmeth_box)
        nonlinloc_settings_box_layout.addWidget(locgau_box)
        nonlinloc_settings_box_layout.addWidget(locgau2_box)
        nonlinloc_settings_box_layout.addWidget(locqual2err_box)
        nonlinloc_settings_box_layout.addWidget(box)

        
        nonlinloc_settings_box = QGroupBox("NonLinLoc Settings", self)
        nonlinloc_settings_box.setCheckable(False)
        nonlinloc_settings_box.setLayout(nonlinloc_settings_box_layout)


        left_layout = QVBoxLayout()
        right_layout = QHBoxLayout()
        left_layout.addWidget(generic_settings_box)
        left_layout.addWidget(vel2grid_settings_box)
        left_layout.addWidget(grid2time_settings_box)
        left_layout.addStretch(1)
        right_layout.addWidget(nonlinloc_settings_box)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        overall_layout = QVBoxLayout()
        overall_layout.addLayout(layout)
        
        overall_layout.addWidget(self.ok_button)

        self.setLayout(overall_layout)
    
    def showGrid(self, name, grid):
        grid_widgets = []
        grid_widgets.append(QLineEdit(str(grid.xnum), self, validator=QIntValidator(bottom=2)))
        if grid.dimension == 2:
            grid_widgets[0].setReadOnly(True)
        grid_widgets.append(QLineEdit(str(grid.ynum), self, validator=QIntValidator(bottom=2)))
        grid_widgets.append(QLineEdit(str(grid.znum), self, validator=QIntValidator(bottom=2)))
        grid_widgets.append(QLineEdit(str(grid.xorig), self, validator=QDoubleValidator()))
        grid_widgets.append(QLineEdit(str(grid.yorig), self, validator=QDoubleValidator()))
        grid_widgets.append(QLineEdit(str(grid.zorig), self, validator=QDoubleValidator()))
        if grid.dimension == 2:
            grid_widgets[3].setReadOnly(True)
            grid_widgets[4].setReadOnly(True)
            # grid_widgets[5].setReadOnly(True)
        grid_widgets.append(QLineEdit(str(grid.dx), self, validator=QDoubleValidator(bottom=0.)))
        grid_widgets.append(QLineEdit(str(grid.grid_type), self))
        grid_widgets.append(QLineEdit(str(grid.dimension), self, validator=QIntValidator(2, 3)))
        grid_box = QGroupBox(name, self)
        grid_box.setCheckable(False)
        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("xnum"))
        hbox.addWidget(grid_widgets[0])
        hbox.addWidget(QLabel("ynum"))
        hbox.addWidget(grid_widgets[1])
        hbox.addWidget(QLabel("znum"))
        hbox.addWidget(grid_widgets[2])
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("xorig"))
        hbox.addWidget(grid_widgets[3])
        hbox.addWidget(QLabel("yorig"))
        hbox.addWidget(grid_widgets[4])
        hbox.addWidget(QLabel("zorig"))
        hbox.addWidget(grid_widgets[5])
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("dx"))
        hbox.addWidget(grid_widgets[6])
        vbox.addLayout(hbox)
        hbox.addWidget(QLabel("Grid Type"))
        hbox.addWidget(grid_widgets[7])
        vbox.addLayout(hbox)
        hbox.addWidget(QLabel("Dimension"))
        hbox.addWidget(grid_widgets[8])
        vbox.addLayout(hbox)

        grid_box.setLayout(vbox)
        return grid_box, grid_widgets

    def get_locmeth_widgets(self):
        lineedit_widgets = {
            "maxDistStaGrid" : QLineEdit(str(self.nll_settings.nll_locmeth["maxDistStaGrid"]), self, validator=QIntValidator()),
            "minNumberPhases" : QLineEdit(str(self.nll_settings.nll_locmeth["minNumberPhases"]), self, validator=QIntValidator()),
            "maxNumberPhases" : QLineEdit(str(self.nll_settings.nll_locmeth["maxNumberPhases"]), self, validator=QIntValidator()),
            "minNumberSphases" : QLineEdit(str(self.nll_settings.nll_locmeth["minNumberSphases"]), self, validator=QIntValidator()),
            "maxNum3DGridMemory" : QLineEdit(str(self.nll_settings.nll_locmeth["maxNum3DGridMemory"]), self, validator=QIntValidator()),
            "VpVsRatio" : QLineEdit(str(self.nll_settings.nll_locmeth["VpVsRatio"]), self, validator=QDoubleValidator()),
            "minDistStaGrid" : QLineEdit(str(self.nll_settings.nll_locmeth["minDistStaGrid"]), self, validator=QDoubleValidator()),
        }
        widgets = {}
        widgets["method"] = QLineEdit(str(self.nll_settings.nll_locmeth["method"]), self)

        for key in ["maxDistStaGrid","minNumberPhases","maxNumberPhases",
                    "minNumberSphases","maxNum3DGridMemory","VpVsRatio","minDistStaGrid"]:
            widgets[key] = QGroupBox(key, self)
            widgets[key].setCheckable(True)
            hbox = QHBoxLayout()
            hbox.addWidget(lineedit_widgets[key])
            widgets[key].setLayout(hbox)
            if not self.nll_settings.nll_locmeth[key]:
                widgets[key].setChecked(False)
            else:
                widgets[key].setChecked(True)

        widgets["iRejectDuplicateArrivals"] = QCheckBox("iRejectDuplicateArrivals", self)

        return widgets, lineedit_widgets
        

    def setup_connections(self):

        self.ok_button.clicked.connect(self.accept)

        # generic
        self.message_flag_widget.editingFinished.connect(self._update_message_flag)
        self.random_number_widget.editingFinished.connect(self._update_random_number)
        self.transform_widget.editingFinished.connect(self._update_transform)

        # vel2grid
        self.vg_out_widget.editingFinished.connect(self._update_vg_out)
        self.vg_grid_widgets[0].editingFinished.connect(self._update_vggrid_xnum)
        self.vg_grid_widgets[1].editingFinished.connect(self._update_vggrid_ynum)
        self.vg_grid_widgets[2].editingFinished.connect(self._update_vggrid_znum)
        self.vg_grid_widgets[3].editingFinished.connect(self._update_vggrid_xorig)
        self.vg_grid_widgets[4].editingFinished.connect(self._update_vggrid_yorig)
        self.vg_grid_widgets[5].editingFinished.connect(self._update_vggrid_zorig)
        self.vg_grid_widgets[6].editingFinished.connect(self._update_vggrid_dx)
        self.vg_grid_widgets[7].editingFinished.connect(self._update_vggrid_gridtype)
        self.vg_grid_widgets[8].editingFinished.connect(self._update_vggrid_dimension)

        # grid2time
        self.gt_ttimeFileRoot_widget.editingFinished.connect(self._update_gt_ttimeFileRoot)
        self.gt_out_widget.editingFinished.connect(self._update_gt_out)

        # nonlinloc
        self.nll_signature_widget.editingFinished.connect(self._update_nll_signature)
        self.locsearch_widgets["type"].editingFinished.connect(self._update_nll_locsearch_type)
        self.locsearch_widgets["initNumCells_x"].editingFinished.connect(self._update_nll_locsearch_initNumCells_x)
        self.locsearch_widgets["initNumCells_y"].editingFinished.connect(self._update_nll_locsearch_initNumCells_y)
        self.locsearch_widgets["initNumCells_z"].editingFinished.connect(self._update_nll_locsearch_initNumCells_z)
        self.locsearch_widgets["minNodeSize"].editingFinished.connect(self._update_nll_locsearch_minNodeSize)
        self.locsearch_widgets["maxNumNodes"].editingFinished.connect(self._update_nll_locsearch_maxNumNodes)
        self.locsearch_widgets["numScatter"].editingFinished.connect(self._update_nll_locsearch_numScatter)
        self.locsearch_widgets["useStationsDensity"].stateChanged.connect(self._update_nll_locsearch_useStationsDensity)
        self.locsearch_widgets["stopOnMinNodeSize"].stateChanged.connect(self._update_nll_locsearch_stopOnMinNodeSize)
        # self.locmeth_widgets["method"].toggled.connect(self._update_method)
        # self.locmeth_widgets["maxDistStaGrid"].toggled.connect(self._update_maxDistStaGrid)
        # self.locmeth_widgets["minNumberPhases"].toggled.connect(self._update_minNumberPhases)
        # self.locmeth_widgets["maxNumberPhases"].toggled.connect(self._update_maxNumberPhases)
        # self.locmeth_widgets["minNumberSphases"].toggled.connect(self._update_minNumberSphases)
        # self.locmeth_widgets["maxNum3DGridMemory"].toggled.connect(self._update_maxNum3DGridMemory)
        # self.locmeth_widgets["VpVsRatio"].toggled.connect(self._update_VpVsRatio)
        # self.locmeth_widgets["minDistStaGrid"].toggled.connect(self._update_minDistStaGrid)
        # self.locmeth_widgets["iRejectDuplicateArrivals"].toggled.connect(self._update_iRejectDuplicateArrivals)
        self.locmeth_widgets["method"].editingFinished.connect(self._update_nll_locmeth_method)
        self.locmeth_lineedit_widgets["maxDistStaGrid"].editingFinished.connect(self._update_nll_locmeth_maxDistStaGrid)
        self.locmeth_lineedit_widgets["minNumberPhases"].editingFinished.connect(self._update_nll_locmeth_minNumberPhases)
        self.locmeth_lineedit_widgets["maxNumberPhases"].editingFinished.connect(self._update_nll_locmeth_maxNumberPhases)
        self.locmeth_lineedit_widgets["minNumberSphases"].editingFinished.connect(self._update_nll_locmeth_minNumberSphases)
        self.locmeth_lineedit_widgets["maxNum3DGridMemory"].editingFinished.connect(self._update_nll_locmeth_maxNum3DGridMemory)
        self.locmeth_lineedit_widgets["VpVsRatio"].editingFinished.connect(self._update_nll_locmeth_VpVsRatio)
        self.locmeth_lineedit_widgets["minDistStaGrid"].editingFinished.connect(self._update_nll_locmeth_minDistStaGrid)
        self.locmeth_widgets["iRejectDuplicateArrivals"].stateChanged.connect(self._update_nll_locmeth_iRejectDuplicateArrivals)
        self.nll_grid_widgets[0].editingFinished.connect(self._update_nll_locgrid_xnum)
        self.nll_grid_widgets[1].editingFinished.connect(self._update_nll_locgrid_ynum)
        self.nll_grid_widgets[2].editingFinished.connect(self._update_nll_locgrid_znum)
        self.nll_grid_widgets[3].editingFinished.connect(self._update_nll_locgrid_xorig)
        self.nll_grid_widgets[4].editingFinished.connect(self._update_nll_locgrid_yorig)
        self.nll_grid_widgets[5].editingFinished.connect(self._update_nll_locgrid_zorig)
        self.nll_grid_widgets[6].editingFinished.connect(self._update_nll_locgrid_dx)
        self.nll_grid_widgets[7].editingFinished.connect(self._update_nll_locgrid_gridtype)
        self.nll_grid_widgets[8].editingFinished.connect(self._update_nll_locgrid_dimension)
        self.locgau_sigmatime_widget.editingFinished.connect(self._update_nll_locgau_sigmatime)
        self.locgau_corrlen_widget.editingFinished.connect(self._update_nll_locgau_corrlen)
        self.locgau2_sigmatfraction_widget.editingFinished.connect(self._update_nll_locgau2_sigmatfraction)
        self.locgau2_sigmatmin_widget.editingFinished.connect(self._update_nll_locgau2_sigmatmin)
        self.locgau2_sigmatmax_widget.editingFinished.connect(self._update_nll_locgau2_sigmatmax)
        self.nll_locqual2err0_widget.editingFinished.connect(self._update_nll_locqual2err0)
        self.nll_locqual2err1_widget.editingFinished.connect(self._update_nll_locqual2err1)
        self.nll_locqual2err2_widget.editingFinished.connect(self._update_nll_locqual2err2)
        self.nll_locqual2err3_widget.editingFinished.connect(self._update_nll_locqual2err3)
        self.nll_locqual2err4_widget.editingFinished.connect(self._update_nll_locqual2err4)
    def _update_message_flag(self):
        self.nll_settings.message_flag = int(self.message_flag_widget.text())
    def _update_random_number(self):
        self.nll_settings.random_number = int(self.random_number_widget.text())
    def _update_transform(self):
        raise NotImplementedError
    def _update_vg_out(self):
        self.nll_settings.vg_out = self.vg_out_widget.text()
    def _update_vggrid_xnum(self):
        self.nll_settings.vg_grid.xnum = int(self.vg_grid_widgets[0].text())
    def _update_vggrid_ynum(self):
        self.nll_settings.vg_grid.ynum = int(self.vg_grid_widgets[1].text())
    def _update_vggrid_znum(self):
        self.nll_settings.vg_grid.znum = int(self.vg_grid_widgets[2].text())
    def _update_vggrid_xorig(self):
        self.nll_settings.vg_grid.xorig = int(self.vg_grid_widgets[3].text())
    def _update_vggrid_yorig(self):
        self.nll_settings.vg_grid.yorig = int(self.vg_grid_widgets[4].text())
    def _update_vggrid_zorig(self):
        self.nll_settings.vg_grid.zorig = int(self.vg_grid_widgets[5].text())
    def _update_vggrid_dx(self):
        self.nll_settings.vg_grid.dx = int(self.vg_grid_widgets[6].text())
    def _update_vggrid_gridtype(self):
        self.nll_settings.vg_grid.gridtype = int(self.vg_grid_widgets[7].text())
    def _update_vggrid_dimension(self):
        self.nll_settings.vg_grid.dimension = int(self.vg_grid_widgets[8].text())
        if self.nll_settings.vg_grid.dimension == 2:
            self.vg_grid_widgets[0].setReadOnly(True)
            self.vg_grid_widgets[3].setReadOnly(True)
            self.vg_grid_widgets[4].setReadOnly(True)
            self.vg_grid_widgets[0].setText("2")
            self.vg_grid_widgets[3].setText("0.0")
            self.vg_grid_widgets[4].setText("0.0")
        elif self.nll_settings.vg_grid.dimension == 3:
            self.vg_grid_widgets[0].setReadOnly(False)
            self.vg_grid_widgets[3].setReadOnly(False)
            self.vg_grid_widgets[4].setReadOnly(False)
        else:
            raise ValueError("Dimension cannot be anything other than 2 or 3")
    def _update_gt_ttimeFileRoot(self):
        self.nll_settings.gt_ttimeFileRoot = self.gt_ttimeFileRoot_widget.text()
    def _update_gt_out(self):
        self.nll_settings.gt_out = self.gt_out.text()
    def _update_nll_signature(self):
        self.nll_settings.nll_signature = self.nll_signature_widget.text()
    def _update_nll_locsearch_type(self):
        raise NotImplementedError("Cannot change locsearchtype yet")
    def _update_nll_locsearch_initNumCells_x(self):
        self.nll_settings.nll_locsearch["initNumCells_x"] = int(self.locsearch_widgets["initNumCells_x"].text())
    def _update_nll_locsearch_initNumCells_y(self):
        self.nll_settings.nll_locsearch["initNumCells_y"] = int(self.locsearch_widgets["initNumCells_y"].text())
    def _update_nll_locsearch_initNumCells_z(self):
        self.nll_settings.nll_locsearch["initNumCells_z"] = int(self.locsearch_widgets["initNumCells_z"].text())
    def _update_nll_locsearch_minNodeSize(self):
        self.nll_settings.nll_locsearch["minNodeSize"] = int(self.locsearch_widgets["minNodeSize"].text())
    def _update_nll_locsearch_maxNumNodes(self):
        self.nll_settings.nll_locsearch["maxNumNodes"] = int(self.locsearch_widgets["maxNumNodes"].text())
    def _update_nll_locsearch_numScatter(self):
        self.nll_settings.nll_locsearch["numScatter"] = int(self.locsearch_widgets["numScatter"].text())
    def _update_nll_locsearch_useStationsDensity(self):
        self.nll_settings.nll_locsearch["useStationsDensity"] = self.locsearch_widgets["useStationsDensity"].checkState()
    def _update_nll_locsearch_stopOnMinNodeSize(self):
        self.nll_settings.nll_locsearch["stopOnMinNodeSize"] = self.locsearch_widgets["stopOnMinNodeSize"].checkState()
    def _update_nll_locmeth_method(self):
        method = self.locmeth_widgets["method"].text()
        if method not in ["GAU_ANALYTIC", "EDT", "EDT_OT_WT", "EDT_OT_WT_ML"]:
            dialog = QErrorMessage()
            dialog.showMessage("""
                             This is an incorrect option for LOCMETH.method\n
                             Choose from: GAU_ANALYTIC, EDT, EDT_OT_WT or EDT_OT_WT_ML
                             """, "WRong")
            # dialog.setModal(True)
            dialog.exec()
            return
        self.nll_settings.nll_locmeth["method"] = method
    def _update_nll_locmeth_maxDistStaGrid(self):
        self.nll_settings.nll_locmeth["maxDistStaGrid"] = self.locmeth_lineedit_widgets["maxDistStaGrid"].text()
    def _update_nll_locmeth_minNumberPhases(self):
        self.nll_settings.nll_locmeth["minNumberPhases"] = self.locmeth_lineedit_widgets["minNumberPhases"].text()
    def _update_nll_locmeth_maxNumberPhases(self):
        self.nll_settings.nll_locmeth["maxNumberPhases"] = self.locmeth_lineedit_widgets["maxNumberPhases"].text()
    def _update_nll_locmeth_minNumberSphases(self):
        self.nll_settings.nll_locmeth["minNumberSphases"] = self.locmeth_lineedit_widgets["minNumberSphases"].text()
    def _update_nll_locmeth_maxNum3DGridMemory(self):
        self.nll_settings.nll_locmeth["maxNum3DGridMemory"] = self.locmeth_lineedit_widgets["maxNum3DGridMemory"].text()
    def _update_nll_locmeth_VpVsRatio(self):
        self.nll_settings.nll_locmeth["VpVsRatio"] = self.locmeth_lineedit_widgets["VpVsRatio"].text()
    def _update_nll_locmeth_minDistStaGrid(self):
        self.nll_settings.nll_locmeth["minDistStaGrid"] = self.locmeth_lineedit_widgets["minDistStaGrid"].text()
    def _update_nll_locmeth_iRejectDuplicateArrivals(self):
        self.nll_settings.nll_locmeth["iRejectDuplicateArrivals"] = self.locmeth_widgets["iRejectDuplicateArrivals"].text()
    def _update_nll_locgrid_xnum(self):
        self.nll_settings.nll_locgrid.xnum = int(self.nll_grid_widgets[0].text())
    def _update_nll_locgrid_ynum(self):
        self.nll_settings.nll_locgrid.ynum = int(self.nll_grid_widgets[1].text())
    def _update_nll_locgrid_znum(self):
        self.nll_settings.nll_locgrid.znum = int(self.nll_grid_widgets[2].text())
    def _update_nll_locgrid_xorig(self):
        self.nll_settings.nll_locgrid.xorig = int(self.nll_grid_widgets[3].text())
    def _update_nll_locgrid_yorig(self):
        self.nll_settings.nll_locgrid.yorig = int(self.nll_grid_widgets[4].text())
    def _update_nll_locgrid_zorig(self):
        self.nll_settings.nll_locgrid.zorig = int(self.nll_grid_widgets[5].text())
    def _update_nll_locgrid_dx(self):
        self.nll_settings.nll_locgrid.dx = int(self.nll_grid_widgets[6].text())
    def _update_nll_locgrid_gridtype(self):
        self.nll_settings.nll_locgrid.gridtype = int(self.nll_grid_widgets[7].text())
    def _update_nll_locgrid_dimension(self):
        self.nll_settings.nll_locgrid.dimension = int(self.nll_grid_widgets[8].text())
        if self.nll_settings.nll_locgrid.dimension == 2:
            self.nll_grid_widgets[0].setReadOnly(True)
            self.nll_grid_widgets[3].setReadOnly(True)
            self.nll_grid_widgets[4].setReadOnly(True)
            self.nll_grid_widgets[0].setText("2")
            self.nll_grid_widgets[3].setText("0.0")
            self.nll_grid_widgets[4].setText("0.0")
        elif self.nll_settings.nll_locgrid.dimension == 3:
            self.nll_grid_widgets[0].setReadOnly(False)
            self.nll_grid_widgets[3].setReadOnly(False)
            self.nll_grid_widgets[4].setReadOnly(False)
        else:
            raise ValueError("Dimension cannot be anything other than 2 or 3")
    def _update_nll_locgau_sigmatime(self):
        self.nll_settings.nll_locgau["sigmatime"] = float(self.locgau_sigmatime_widget.text())
    def _update_nll_locgau_corrlen(self):
        self.nll_settings.nll_locgau["corrlen"] = float(self.locgau_corrlen_widget.text())
    def _update_nll_locgau2_sigmatfraction(self):
        self.nll_settings.nll_locgau2["sigmatfraction"] = float(self.locgau2_sigmatfraction_widget.text())
    def _update_nll_locgau2_sigmatmin(self):
        self.nll_settings.nll_locgau2["sigmatmin"] = float(self.locgau2_sigmatmin_widget.text())
    def _update_nll_locgau2_sigmatmax(self):
        self.nll_settings.nll_locgau2["sigmatmax"] = float(self.locgau2_sigmatmax_widget.text())
    def _update_nll_locqual2err0(self):
        self.nll_settings.nll_locqual2err[0] = float(self.locqual2err0_widget.text())
    def _update_nll_locqual2err1(self):
        self.nll_settings.nll_locqual2err[1] = float(self.locqual2err1_widget.text())
    def _update_nll_locqual2err2(self):
        self.nll_settings.nll_locqual2err[2] = float(self.locqual2err2_widget.text())
    def _update_nll_locqual2err3(self):
        self.nll_settings.nll_locqual2err[3] = float(self.locqual2err3_widget.text())
    def _update_nll_locqual2err4(self):
        self.nll_settings.nll_locqual2err[4] = float(self.locqual2err4_widget.text())


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    # app = QApplication([])
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    # prj = Project("/space/tg286/iceland/reykjanes/qmigrate/may21/nov23_dyke_tight")
    # loc = LocateRun(prj)
    window = MainWindow("/space/tg286/iceland/reykjanes/qmigrate/may21/nov23_dyke_tight", "")
    window.show()
    app.exec()