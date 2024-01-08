from PySide6.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QListWidget, QWidget
from PySide6.QtWidgets import QCheckBox, QPushButton, QDateTimeEdit, QListWidgetItem
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QLineEdit, QMessageBox
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QApplication
from PySide6 import QtCore
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
from obspy import UTCDateTime
plt.close("all")

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd

from projects import *
from pick_window import PickWindow

class LocateWindow(QWidget):
    def __init__(self, project):
        super().__init__()
        
        self.project = project  
        self.locate_run = LocateRun(self.project)
        self.define_parameters()

        self.init_ui()

        self.plot_events()
        self.plot_cross_section("longitude")
        self.plot_event_timeseries()

        self.setup_connections()

    def define_parameters(self):
        self.current_events = self.locate_run.events
        print(self.current_events.DT)
        self.current_events.loc[:,"text"] = self.get_list_of_events(self.current_events)
        self.current_events.set_index("text", inplace=True)
        self.selected_event = None

        # project time span
        self.mintime = self.current_events.DT.min()
        self.maxtime = self.current_events.DT.max()

        # project spatial span
        self.minlat = self.project.LUT.grid_extent[0,1]
        self.minlon = self.project.LUT.grid_extent[0,0]
        self.maxlat = self.project.LUT.grid_extent[1,1]
        self.maxlon = self.project.LUT.grid_extent[1,0]
        self.mindep = self.project.LUT.grid_extent[0,2]
        self.maxdep = self.project.LUT.grid_extent[1,2]

        # project magnitude span
        self.minmag = self.locate_run.events.ML.min()
        self.maxmag = self.locate_run.events.ML.max()
        self.minmag_for_scaling = np.abs(np.floor(self.minmag))

        # max errors and covariance
        self.maxerrE = self.locate_run.events.GAU_ErrX.max()
        self.maxerrN = self.locate_run.events.GAU_ErrY.max()
        self.maxerrZ = self.locate_run.events.GAU_ErrZ.max()
        self.maxcovE = self.locate_run.events.COV_ErrX.max()
        self.maxcovN = self.locate_run.events.COV_ErrY.max()
        self.maxcovZ = self.locate_run.events.COV_ErrZ.max()

        # mapview aspect ratio
        self.aspect = 1. / np.cos(np.deg2rad((self.maxlat + self.minlat)/2))

    def plot_events(self):
        self.mapview_ax.scatter(self.locate_run.events.X, self.locate_run.events.Y,
                     c="gray", s=((self.current_events.ML+self.minmag_for_scaling)**2)/5)
        self.mapview_events = self.mapview_ax.scatter(self.current_events.X, self.current_events.Y,
                                                    c=mdates.date2num(self.current_events.DT), s=((self.current_events.ML+self.minmag_for_scaling)**2)/5)
        self.mapview_ax.set_xlim(self.project.LUT.grid_extent[0,0], self.project.LUT.grid_extent[1,0])
        self.mapview_ax.set_ylim(self.project.LUT.grid_extent[0,1], self.project.LUT.grid_extent[1,1])
        self.mapview_bounds_minlat = self.mapview_ax.axhline(self.minlat, c="gray", ls="--")
        self.mapview_bounds_maxlat = self.mapview_ax.axhline(self.maxlat, c="gray", ls="--")
        self.mapview_bounds_minlon = self.mapview_ax.axvline(self.minlon, c="gray", ls="--")
        self.mapview_bounds_maxlon = self.mapview_ax.axvline(self.maxlon, c="gray", ls="--")
        self.mapview_ax.set_aspect(self.aspect)
        self.canvas.draw()

    def plot_cross_section(self, type):
        if type == "longitude":
            ## a basic along longitude plot - used for intial plotting before 
            ## more complex custom cross section
            self.crosssection_ax.scatter(self.locate_run.events.X, self.locate_run.events.Z,
                        c="gray", s=((self.current_events.ML+self.minmag_for_scaling)**2)/5)
            self.crosssection_events = self.crosssection_ax.scatter(self.current_events.X, self.current_events.Z,
                                                                c=mdates.date2num(self.current_events.DT), s=((self.current_events.ML+self.minmag_for_scaling)**2)/5)
            self.crosssection_ax.set_xlim(self.project.LUT.grid_extent[0,0], self.project.LUT.grid_extent[1,0])
            self.crosssection_ax.set_ylim(self.project.LUT.grid_extent[0,2], self.project.LUT.grid_extent[1,2])
        
        self.crosssection_bounds_mindep = self.crosssection_ax.axhline(self.mindep, c="gray", ls="--")
        self.crosssection_bounds_maxdep = self.crosssection_ax.axhline(self.maxdep, c="gray", ls="--")

        self.crosssection_ax.invert_yaxis()
        self.canvas.draw()  

        self.crosssection_type = type

    def plot_event_timeseries(self):
        self.timeseries_ax.scatter(self.locate_run.events.DT, self.locate_run.events.ML,
                     c="gray", s=((self.current_events.ML+self.minmag_for_scaling)**2)/5)
        self.timeseries_events = self.timeseries_ax.scatter(self.current_events.DT, self.current_events.ML,
                                                    c=mdates.date2num(self.current_events.DT), s=((self.current_events.ML+self.minmag_for_scaling)**2)/5)
        self.timeseries_ax.set_xlim(self.mintime, self.maxtime)
        self.timeseries_bounds_minmag = self.timeseries_ax.axhline(self.minmag, c="gray", ls="--")
        self.timeseries_bounds_maxmag = self.timeseries_ax.axhline(self.maxmag, c="gray", ls="--")
        self.timeseries_ax.set_ylabel("Magnitude")
        self.canvas.draw()

    def plot_waveforms(self, ev):
        self.waveforms_ax.clear()
        stream = self.locate_run.get_waveforms(ev.EventID, processed=True)
        picks = self.locate_run.get_picks(ev.EventID)

        unique_stations = np.array(list(set([tr.stats.station for tr in stream])))
        hyp_distance = np.array([np.sqrt((ev.Xproj-self.project.stations.loc[station,"Xproj"])**2 + 
                                         (ev.Yproj-self.project.stations.loc[station,"Yproj"])**2) 
                                         for station in unique_stations])
        sort_order = np.argsort(hyp_distance)

        count = 0
        for i in sort_order:
            for comp, col in zip("ENZ", ["blue", "green", "red"]):
                st = stream.select(station=unique_stations[i], component=comp)
                if len(st) == 0:
                    # self.waveforms_ax.axhline(count, c=col, lw=0.5)
                    continue
                norm = np.max([np.abs(tr.data).max() for tr in st]) * 2
                for tr in st:
                    self.waveforms_ax.plot(tr.times(reftime=UTCDateTime(ev.DT)), 
                                           count + (tr.data/norm), "-", c=col, lw=0.5)
                    
            for phase, col in zip("PS",["red", "blue"]):
                try:
                    model_time = picks.loc[(unique_stations[i], phase), "ModelledTime"].tz_convert(None)
                except KeyError:
                    continue
                offset = (model_time-ev.DT).total_seconds()
                self.waveforms_ax.plot([offset, offset], [count-0.3, count+0.3], "-", c=col)
            count += 1
        
        self.waveforms_ax.axvline(0, ls="--", lw=2, c="k")
        self.waveforms_ax.set_yticks(range(count), labels=unique_stations[sort_order])
        self.waveforms_ax.set_ylim(-0.5, count-0.5)
        self.waveforms_ax.set_xlim(-0.5, (picks.loc[:, "ModelledTime"].dt.tz_convert(None)-ev.DT).dt.total_seconds().max()+5)
        self.canvas.draw()


        

    def redraw_plots(self):
        print("REDRAW PLOTS")
        self.mapview_bounds_minlat.set_ydata([self.minlat, self.minlat])
        self.mapview_bounds_maxlat.set_ydata([self.maxlat, self.maxlat])
        self.mapview_bounds_minlon.set_xdata([self.minlon, self.minlon])
        self.mapview_bounds_maxlon.set_xdata([self.maxlon, self.maxlon])
        self.crosssection_bounds_mindep.set_ydata([self.mindep, self.mindep])
        self.crosssection_bounds_maxdep.set_ydata([self.maxdep, self.maxdep])
        self.timeseries_bounds_minmag.set_ydata([self.minmag, self.minmag])
        self.timeseries_bounds_maxmag.set_ydata([self.maxmag, self.maxmag])

        self.mapview_events.set_offsets(self.current_events.loc[:,["X","Y"]])
        self.mapview_events.set_array(mdates.date2num(self.current_events.loc[:,"DT"]))
        self.mapview_events.set_sizes(((self.current_events.ML+self.minmag_for_scaling)**2)/5)

        if self.crosssection_type == "longitude":
            self.crosssection_events.set_offsets(self.current_events.loc[:,["X","Z"]])
            self.crosssection_events.set_array(mdates.date2num(self.current_events.loc[:,"DT"]))
            self.crosssection_events.set_sizes(((self.current_events.ML+self.minmag_for_scaling)**2)/5)
        else:
            print("NOT IMPLEMENTED FOR OTHER C|ROSS SECION TYPES YET")

        self.timeseries_events.set_offsets(self.current_events.loc[:,["DT","ML"]])
        self.timeseries_events.set_array(mdates.date2num(self.current_events.loc[:,"DT"]))
        self.timeseries_events.set_sizes(((self.current_events.ML+self.minmag_for_scaling)**2)/5)
        
        self.canvas.draw()

        

    def init_ui(self):

        page_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # define some widgets

        self.starttime_widget = QDateTimeEdit(self.mintime, self)
        self.starttime_widget.setMinimumDate(self.project.starttime)
        self.starttime_widget.setMaximumDate(self.project.endtime)

        self.endtime_widget = QDateTimeEdit(self.maxtime, self)
        self.endtime_widget.setMinimumDate(self.project.starttime)
        self.endtime_widget.setMaximumDate(self.project.endtime)

        # box for location
        location_box_widget = QGroupBox(self)
        self.max_latitude_widget = QLineEdit(str(np.ceil(self.maxlat)), self, 
                                                alignment=QtCore.Qt.AlignHCenter,
                                                validator=QDoubleValidator())
        self.min_latitude_widget = QLineEdit(str(np.floor(self.minlat)), self, 
                                                alignment=QtCore.Qt.AlignHCenter,
                                                validator=QDoubleValidator())
        self.min_longitude_widget = QLineEdit(str(np.ceil(self.minlon)), self, 
                                                alignment=QtCore.Qt.AlignHCenter,
                                                validator=QDoubleValidator())
        self.max_longitude_widget = QLineEdit(str(np.floor(self.maxlon)), self, 
                                                alignment=QtCore.Qt.AlignHCenter,
                                                validator=QDoubleValidator())
        self.min_depth_widget = QLineEdit(str(np.ceil(self.mindep)), self, 
                                                alignment=QtCore.Qt.AlignHCenter,
                                                validator=QDoubleValidator())
        self.max_depth_widget = QLineEdit(str(np.floor(self.maxdep)), self, 
                                                alignment=QtCore.Qt.AlignHCenter,
                                                validator=QDoubleValidator())
        location_box_layout = QVBoxLayout()
        location_box_layout.addWidget(self.max_latitude_widget)
        location_box_layout.addWidget(QLabel("N", alignment=QtCore.Qt.AlignHCenter))
        middle_line_layout = QHBoxLayout()
        middle_line_layout.addWidget(self.min_longitude_widget)
        middle_line_layout.addWidget(QLabel("W"))
        middle_line_layout.addWidget(QLabel("E"))
        middle_line_layout.addWidget(self.max_longitude_widget)
        location_box_layout.addLayout(middle_line_layout)
        location_box_layout.addWidget(QLabel("S", alignment=QtCore.Qt.AlignHCenter))
        location_box_layout.addWidget(self.min_latitude_widget)
        min_depth_layout = QHBoxLayout()
        min_depth_layout.addWidget(QLabel("Minimum Depth"))
        min_depth_layout.addWidget(self.min_depth_widget)
        location_box_layout.addLayout(min_depth_layout)
        max_depth_layout = QHBoxLayout()
        max_depth_layout.addWidget(QLabel("Maximum Depth"))
        max_depth_layout.addWidget(self.max_depth_widget)
        location_box_layout.addLayout(max_depth_layout)
        location_box_widget.setLayout(location_box_layout)

        # min and max magnitude
        self.min_magnitude_widget = QLineEdit(str(np.floor(self.minmag)), self,
                                                validator=QDoubleValidator(bottom=0.))
        self.max_magnitude_widget = QLineEdit(str(np.ceil(self.maxmag)), self,
                                                validator=QDoubleValidator(bottom=0.))

        # box for error limits
        maxerror_box_widget = QGroupBox(self)
        self.maxerr_N_widget = QLineEdit(str(np.ceil(self.maxerrN)), self, 
                                            alignment=QtCore.Qt.AlignHCenter,
                                            validator=QDoubleValidator(bottom=0.))
        self.maxerr_E_widget = QLineEdit(str(np.ceil(self.maxerrE)), self, 
                                            alignment=QtCore.Qt.AlignHCenter,
                                            validator=QDoubleValidator(bottom=0.))
        self.maxerr_Z_widget = QLineEdit(str(np.ceil(self.maxerrZ)), self, 
                                            alignment=QtCore.Qt.AlignHCenter,
                                            validator=QDoubleValidator(bottom=0.))
        maxerror_box_layout = QVBoxLayout()
        maxerror_box_layout.addWidget(QLabel("Max Error", alignment=QtCore.Qt.AlignCenter))
        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("N", alignment=QtCore.Qt.AlignCenter))
        line_layout.addWidget(self.maxerr_N_widget)
        line_layout.addWidget(QLabel("E", alignment=QtCore.Qt.AlignCenter))
        line_layout.addWidget(self.maxerr_E_widget)
        line_layout.addWidget(QLabel("Z", alignment=QtCore.Qt.AlignCenter))
        line_layout.addWidget(self.maxerr_Z_widget)
        maxerror_box_layout.addLayout(line_layout)
        maxerror_box_widget.setLayout(maxerror_box_layout)

        # box for Covariance error limits
        maxcov_box_widget = QGroupBox(self)
        self.maxcov_N_widget = QLineEdit(str(np.ceil(self.maxcovN)), self, 
                                        alignment=QtCore.Qt.AlignHCenter,
                                        validator=QDoubleValidator(bottom=0.))
        self.maxcov_E_widget = QLineEdit(str(np.ceil(self.maxcovE)), self, 
                                        alignment=QtCore.Qt.AlignHCenter,
                                        validator=QDoubleValidator(bottom=0.))
        self.maxcov_Z_widget = QLineEdit(str(np.ceil(self.maxcovZ)), self, 
                                        alignment=QtCore.Qt.AlignHCenter,
                                        validator=QDoubleValidator(bottom=0.))
        maxcov_box_layout = QVBoxLayout()
        maxcov_box_layout.addWidget(QLabel("Max Covariance Error", alignment=QtCore.Qt.AlignCenter))
        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("N", alignment=QtCore.Qt.AlignCenter))
        line_layout.addWidget(self.maxcov_N_widget)
        line_layout.addWidget(QLabel("E", alignment=QtCore.Qt.AlignCenter))
        line_layout.addWidget(self.maxcov_E_widget)
        line_layout.addWidget(QLabel("Z", alignment=QtCore.Qt.AlignCenter))
        line_layout.addWidget(self.maxcov_Z_widget)
        maxcov_box_layout.addLayout(line_layout)
        maxcov_box_widget.setLayout(maxcov_box_layout)

        self.show_error_button = QPushButton("Show Error")
        self.show_cov_button = QPushButton("Show Cov")
        self.show_bval_button = QPushButton("Show b-value")
        self.pick_button = QPushButton("Pick Event")

        # scrollable list of events
        self.event_list_widget = QListWidget(self)
        self.load_list_of_events()



        ## add to left layout
        startdate_layout = QHBoxLayout()
        startdate_layout.addWidget(QLabel("Start time", self))
        startdate_layout.addWidget(self.starttime_widget)
        left_layout.addLayout(startdate_layout)

        enddate_layout = QHBoxLayout()
        enddate_layout.addWidget(QLabel("End time", self))
        enddate_layout.addWidget(self.endtime_widget)
        left_layout.addLayout(enddate_layout)

        left_layout.addWidget(location_box_widget)

        minmag_layout = QHBoxLayout()
        minmag_layout.addWidget(QLabel("Min magnitude", self))
        minmag_layout.addWidget(self.min_magnitude_widget)
        left_layout.addLayout(minmag_layout)

        maxmag_layout = QHBoxLayout()
        maxmag_layout.addWidget(QLabel("Max magnitude", self))
        maxmag_layout.addWidget(self.max_magnitude_widget)
        left_layout.addLayout(maxmag_layout)

        left_layout.addWidget(maxerror_box_widget)
        left_layout.addWidget(maxcov_box_widget)

        left_layout.addWidget(self.show_error_button)
        left_layout.addWidget(self.show_cov_button)
        left_layout.addWidget(self.show_bval_button)

        left_layout.addWidget(self.event_list_widget)
        left_layout.addWidget(self.pick_button)

        left_layout.addStretch(1)
        page_layout.addLayout(left_layout, stretch=1)

        ## add the matplotlib figure canvas
        self.fig, self.axes = plt.subplots(2, 2, height_ratios=(3,1))
        self.mapview_ax = self.axes[0,0]
        self.crosssection_ax = self.axes[1,0]
        self.timeseries_ax = self.axes[1,1]
        self.waveforms_ax = self.axes[0,1]
        self.fig.tight_layout(pad=0.2, w_pad=1)
        self.canvas = FigureCanvas(self.fig)
        page_layout.addWidget(self.canvas, stretch=10)

        self.setLayout(page_layout)
        
    #     page_layout.addWidget(self.canvas, stretch=10)
        
    #     self.canvas.draw()
    #     self.setLayout(page_layout)
    #     # geometry = app.desktop().availableGeometry()
        # self.showMaximized()
        # self.setFixedSize(600,400)

    #     ## connect up the widgets to some actions
    #     self.focuslength_widget.inputRejected.connect(self.bad_input)
    #     self.focuslength_widget.editingFinished.connect(self.help)
    #     # self.focuslength_widget.textChanged.connect(self.help)
    
        # setup connections
    def setup_connections(self):
        self.event_list_widget.itemClicked.connect(self.select_event)

        # button pushes
        self.show_error_button.clicked.connect(self.show_error_window)
        self.show_cov_button.clicked.connect(self.show_cov_window)
        self.show_bval_button.clicked.connect(self.show_bval_window)
        self.pick_button.clicked.connect(self.pick_window)

        ## line_edit changes
        self.max_latitude_widget.editingFinished.connect(self.set_max_latitude)
        self.min_latitude_widget.editingFinished.connect(self.set_min_latitude)
        self.max_longitude_widget.editingFinished.connect(self.set_max_longitude)
        self.min_longitude_widget.editingFinished.connect(self.set_min_longitude)
        self.max_depth_widget.editingFinished.connect(self.set_max_depth)
        self.min_depth_widget.editingFinished.connect(self.set_min_depth)
        self.maxcov_E_widget.editingFinished.connect(self.set_maxcovE)
        self.maxcov_N_widget.editingFinished.connect(self.set_maxcovN)
        self.maxcov_Z_widget.editingFinished.connect(self.set_maxcovZ)
        self.maxerr_E_widget.editingFinished.connect(self.set_maxerrE)
        self.maxerr_N_widget.editingFinished.connect(self.set_maxerrN)
        self.maxerr_Z_widget.editingFinished.connect(self.set_maxerrZ)
        self.min_magnitude_widget.editingFinished.connect(self.set_min_magnitude)
        self.max_magnitude_widget.editingFinished.connect(self.set_max_magnitude)
        # update current event list
        self.max_latitude_widget.editingFinished.connect(self.update_current_event_list)
        self.min_latitude_widget.editingFinished.connect(self.update_current_event_list)
        self.max_longitude_widget.editingFinished.connect(self.update_current_event_list)
        self.min_longitude_widget.editingFinished.connect(self.update_current_event_list)
        self.max_depth_widget.editingFinished.connect(self.update_current_event_list)
        self.min_depth_widget.editingFinished.connect(self.update_current_event_list)
        self.maxcov_E_widget.editingFinished.connect(self.update_current_event_list)
        self.maxcov_N_widget.editingFinished.connect(self.update_current_event_list)
        self.maxcov_Z_widget.editingFinished.connect(self.update_current_event_list)
        self.maxerr_E_widget.editingFinished.connect(self.update_current_event_list)
        self.maxerr_N_widget.editingFinished.connect(self.update_current_event_list)
        self.maxerr_Z_widget.editingFinished.connect(self.update_current_event_list)
        self.min_magnitude_widget.editingFinished.connect(self.update_current_event_list)
        self.max_magnitude_widget.editingFinished.connect(self.update_current_event_list)
        # redraw plots
        self.max_latitude_widget.editingFinished.connect(self.redraw_plots)
        self.min_latitude_widget.editingFinished.connect(self.redraw_plots)
        self.max_longitude_widget.editingFinished.connect(self.redraw_plots)
        self.min_longitude_widget.editingFinished.connect(self.redraw_plots)
        self.max_depth_widget.editingFinished.connect(self.redraw_plots)
        self.min_depth_widget.editingFinished.connect(self.redraw_plots)
        self.maxcov_E_widget.editingFinished.connect(self.redraw_plots)
        self.maxcov_N_widget.editingFinished.connect(self.redraw_plots)
        self.maxcov_Z_widget.editingFinished.connect(self.redraw_plots)
        self.maxerr_E_widget.editingFinished.connect(self.redraw_plots)
        self.maxerr_N_widget.editingFinished.connect(self.redraw_plots)
        self.maxerr_Z_widget.editingFinished.connect(self.redraw_plots)
        self.min_magnitude_widget.editingFinished.connect(self.redraw_plots)
        self.max_magnitude_widget.editingFinished.connect(self.redraw_plots)       


    def select_event(self, event):
        self.plot_waveforms(self.current_events.loc[event.text(),:])
        self.current_event = self.current_events.loc[event.text(),:]

    def show_error_window(self):
        print("SHOW HISTOGRAMS OF THE ERROR")
        self.error_plot_window = ErrorPlotWindow()
        self.error_plot_window.plot_err_histograms(self.locate_run.events, self.current_events,
                                                 [self.maxerrE, self.maxerrN, self.maxerrZ],
                                                 option="GAU_Err")
        self.error_plot_window.show()
    
    def show_cov_window(self):
        print("SHOW HISTOGRAMS OF THE COV ERROR")
        self.error_plot_window = ErrorPlotWindow()
        self.error_plot_window.plot_err_histograms(self.locate_run.events, self.current_events,
                                                 [self.maxcovE, self.maxcovN, self.maxcovZ],
                                                 option="COV_Err")
        self.error_plot_window.show()
    
    def show_bval_window(self):
        print("SHOW BVALUE - NOT IMPLEMENTED YET")

    def pick_window(self):
        self.pick_obj = PickWindow(self.locate_run, self.current_event)
        self.pick_obj.show()

    def update_current_event_list(self):
        events = self.locate_run.events
        lon_mask = (events.X >= self.minlon) & (events.X < self.maxlon)
        lat_mask = (events.Y >= self.minlat) & (events.Y < self.maxlat)
        dep_mask = (events.Z >= self.mindep) & (events.Z < self.maxdep)

        cov_mask = (events.COV_ErrX < self.maxcovE) & (events.COV_ErrY < self.maxcovN) & (events.COV_ErrZ < self.maxcovZ)
        err_mask = (events.GAU_ErrX < self.maxerrE) & (events.GAU_ErrY < self.maxerrN) & (events.GAU_ErrZ < self.maxerrZ)

        mag_mask = (events.ML >= self.minmag) & (events.ML < self.maxmag)

        mask = lon_mask & lat_mask & dep_mask & cov_mask & err_mask & mag_mask

        self.current_events = events.loc[mask,:].copy()
        self.current_events.loc[:,"text"] = self.get_list_of_events(self.current_events)
        self.current_events.set_index("text", inplace=True)
        self.load_list_of_events()

    def load_list_of_events(self):
        self.event_list_widget.clear()
        _list_items = [QListWidgetItem(s, self.event_list_widget) for s in self.get_list_of_events(self.current_events)]

    def get_list_of_events(self, events):
        ## do some sorting here
        return events.loc[:,"DT"].dt.strftime("%Y-%m-%d %H:%M:%S") + \
                    "   " + \
                    events.loc[:,"ML"].apply(lambda val: f"{val:.1f}")

    ## some setting functions
    def set_max_latitude(self):
        self.maxlat = float(self.max_latitude_widget.text())
    def set_min_latitude(self):
        self.minlat = float(self.min_latitude_widget.text())
    def set_max_longitude(self):
        self.maxlon = float(self.max_longitude_widget.text())
    def set_min_longitude(self):
        self.minlon = float(self.min_longitude_widget.text())
    def set_max_depth(self):
        self.maxdep = float(self.max_depth_widget.text())
    def set_min_depth(self):
        self.mindep = float(self.min_depth_widget.text())
    def set_maxcovE(self):
        self.maxcovE = float(self.maxcov_E_widget.text())
    def set_maxcovN(self):
        self.maxcovN = float(self.maxcov_N_widget.text())
    def set_maxcovZ(self):
        self.maxcovZ = float(self.maxcov_Z_widget.text())
    def set_maxerrE(self):
        self.maxerrE = float(self.maxerr_E_widget.text())
    def set_maxerrN(self):
        self.maxerrN = float(self.maxerr_N_widget.text())
    def set_maxerrZ(self):
        self.maxerrZ = float(self.maxerr_Z_widget.text())
    def set_min_magnitude(self):
        self.minmag = float(self.min_magnitude_widget.text())
    def set_max_magnitude(self):
        self.maxmag = float(self.max_magnitude_widget.text())

    # def bad_input(self):
    #     print("HELP")
    #     msgBox = QMessageBox()
    #     msgBox.setText("Invalid input. Must be a integer between 300 and 172800")
    #     msgBox.exec()

class ErrorPlotWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()   
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.fig, self.axes = plt.subplots(1,3,sharey=True, sharex=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_err_histograms(self, evs, filt_evs, limits, option="GAU_Err", components="XYZ"):

        maxbin = evs.loc[:,[option+comp for comp in components]].max().max()
        if maxbin < 15:
            bin_width = 0.5
        elif maxbin < 30:
            bin_width = 1
        elif maxbin < 60:
            bin_width = 2
        elif maxbin < 100:
            bin_width = 5
        else:
            bin_width = 10

        bins = np.arange(0, np.ceil(maxbin), bin_width)
        

        for ax, comp, limit in zip(self.axes, components, limits):
            ax.hist(evs.loc[:,option+comp], bins=bins, ec="none", fc="gray")
            ax.hist(filt_evs.loc[:,option+comp], bins=bins, ec="k", fc="blue")
            ax.axvline(limit, c="r")
            ax.set_xlabel("Error [km]")
            ax.set_title(comp)

        self.axes[0].set_ylabel("Count")
        self.fig.tight_layout(pad=0.2, w_pad=1)

if __name__ == '__main__':
    # app = QApplication([])
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    prj = Project("/space/tg286/iceland/reykjanes/qmigrate/may21/nov23_dyke_tight")
    window = LocateWindow(prj)
    window.show()
    app.exec()