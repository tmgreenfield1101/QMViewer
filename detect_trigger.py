from PySide6.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QTextEdit, QWidget
from PySide6.QtWidgets import QCheckBox, QPushButton, QDateTimeEdit, QRadioButton
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QLineEdit, QMessageBox
from PySide6.QtGui import QIntValidator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd
import os

from PySide6.QtWidgets import QApplication
from projects import *

class DetectTriggerWindow(QWidget):
    def __init__(self, project):
        super().__init__()
        
        self.project = project  
        self.detect_run = DetectRun(self.project)
        self.trigger_run = TriggerRun(self.project)

        self.init_ui()

        print(self.canvas)
    def init_ui(self):

        page_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # define some widgets
        self.defsetting_widget = QPushButton("Define Settings", self)

        self.starttime_widget = QDateTimeEdit(self.project.starttime, self)
        self.starttime_widget.setMinimumDate(self.project.starttime)
        self.starttime_widget.setMaximumDate(self.project.endtime)

        self.endtime_widget = QDateTimeEdit(self.project.endtime, self)
        self.endtime_widget.setMinimumDate(self.project.starttime)
        self.endtime_widget.setMaximumDate(self.project.endtime)

        # focustime = self.project.starttime + (self.project.endtime - self.project.starttime)/2
        focustime = pd.to_datetime("2023-11-10", utc=True)
        self.focustime_widget = QDateTimeEdit(focustime, self)
        self.focustime_widget.setMinimumDate(self.project.starttime)
        self.focustime_widget.setMaximumDate(self.project.endtime)

        self.focuslength_widget = QLineEdit("86400", self)
        validator = QIntValidator(300, 172800, self)
        self.focuslength_widget.setValidator(validator)

        self.normcoa_widget = QCheckBox("Normalized COA", self)
        self.normcoa_widget.setChecked(True)

        self.minevint_widget = QCheckBox("Show Min Event Interval", self)
        self.margwind_widget = QCheckBox("Show Marginal Window", self)
        self.retrig_widget = QPushButton("Re-Trigger")
        self.write_widget = QPushButton("Write Selected Events For Locate")

        # radio button widget
        self.plotradgrp_widget = QGroupBox(self)
        # self.plotradgrp_widget.setCheckable(True)
        self.plotradgrp_widget.setChecked(False)
        vbox = QVBoxLayout()
        self.colbycoa_button = QRadioButton("Colour by COA", self)
        self.colbydate_button = QRadioButton("Colour by Date", self)
        self.colbycoa_button.setChecked(True)
        vbox.addWidget(self.colbycoa_button)
        vbox.addWidget(self.colbydate_button)
        vbox.addStretch(1)
        self.plotradgrp_widget.setLayout(vbox)


        # define the left hand column
        left_layout.addWidget(self.defsetting_widget)

        startdate_layout = QHBoxLayout()
        startdate_layout.addWidget(QLabel("Start time", self))
        startdate_layout.addWidget(self.starttime_widget)
        left_layout.addLayout(startdate_layout)

        enddate_layout = QHBoxLayout()
        enddate_layout.addWidget(QLabel("End time", self))
        enddate_layout.addWidget(self.endtime_widget)
        left_layout.addLayout(enddate_layout)

        focusdate_layout = QHBoxLayout()
        focusdate_layout.addWidget(QLabel("Focus time", self))
        focusdate_layout.addWidget(self.focustime_widget)
        left_layout.addLayout(focusdate_layout)

        focuslength_layout = QHBoxLayout()
        focuslength_layout.addWidget(QLabel("Focus length", self))
        focuslength_layout.addWidget(self.focuslength_widget)
        left_layout.addLayout(focuslength_layout)

        left_layout.addWidget(self.plotradgrp_widget)

        left_layout.addWidget(self.normcoa_widget)
        left_layout.addWidget(self.minevint_widget)
        left_layout.addWidget(self.margwind_widget)

        left_layout.addWidget(self.retrig_widget)
        left_layout.addWidget(self.write_widget)
        left_layout.addStretch(1)

        page_layout.addLayout(left_layout, stretch=1)

        # read the trigger content
        self.plot_mpl_content(self.trigger_run.triggered_events)
        
        page_layout.addWidget(self.canvas, stretch=10)
        
        self.canvas.draw()
        self.setLayout(page_layout)
        # geometry = app.desktop().availableGeometry()
        # self.showMaximized()
        # self.setFixedSize(600,400)

        ## connect up the widgets to some actions
        self.focuslength_widget.inputRejected.connect(self.bad_input)
        self.focuslength_widget.editingFinished.connect(self.help)
        # self.focuslength_widget.textChanged.connect(self.help)
    
    def bad_input(self):
        print("HELP")
        msgBox = QMessageBox()
        msgBox.setText("Invalid input. Must be a integer between 300 and 172800")
        msgBox.exec()

    def help(self):
        print("HELP ME")
        

    def plot_mpl_content(self, events):

        if self.normcoa_widget.isChecked():
            norm = "COA_NORM"
        else:
            norm = "COA"
        
        if self.colbycoa_button.isChecked():
            color_col = norm
        elif self.colbydate_button.isChecked():
            color_col = "CoaTime"
        
        vmin = events.loc[:,color_col].min()
        vmax = events.loc[:,color_col].max()

        # load scanmseed
        self.scanmseed = self.detect_run.get_scanmseed(self.focustime_widget.dateTime().toPython(), 
                                           int(self.focuslength_widget.text()))
        self.scanmseed = self.scanmseed.select(station=norm[0:5])

        # load event times for scanmseed plot
        focustime = pd.to_datetime(self.focustime_widget.dateTime().toPython(), utc=True)
        half_length = pd.to_timedelta(int(self.focuslength_widget.text()), unit="S") / 2
        self.focus_events = self.trigger_run.get_events(datemin=focustime-half_length,
                                                   datemax=focustime+half_length)

        self.fig = plt.Figure()
        gridspec_0 = GridSpec(1,2, figure=self.fig)
        gridspec_00 = GridSpecFromSubplotSpec(2,2, width_ratios=(3,1), height_ratios=(3,1),
                                    subplot_spec=gridspec_0[0])
        mapview_ax = self.fig.add_subplot(gridspec_00[0,0])
        long_ax = self.fig.add_subplot(gridspec_00[1,0])
        lat_ax = self.fig.add_subplot(gridspec_00[0,1])
        coahist_ax = self.fig.add_subplot(gridspec_00[1,1])

        gridspec_01 = GridSpecFromSubplotSpec(2,1, subplot_spec=gridspec_0[1])
        self.coatime_ax = self.fig.add_subplot(gridspec_01[0])
        self.coatimezoom_ax = self.fig.add_subplot(gridspec_01[1])

        mapview_ax = plot_mapview(mapview_ax, 
                                  self.project.stations, events, 
                                  color_col=color_col, vmin=vmin, vmax=vmax)
        
        long_ax = plot_longitude_crosssection(long_ax, 
                                  self.project.stations, events,
                                  color_col=color_col, vmin=vmin, vmax=vmax)
        
        lat_ax = plot_latitude_crosssection(lat_ax, 
                                  self.project.stations, events,
                                  color_col=color_col, vmin=vmin, vmax=vmax)
        
        coahist_ax = plot_coa_hist(coahist_ax, events)

        self.coatime_ax = plot_coa_time(self.coatime_ax, self.scanmseed, self.trigger_run,
                                   mintime=focustime-half_length,
                                   maxtime=focustime+half_length,
                                   plot_minevent=self.minevint_widget.isChecked(),
                                   plot_margwindow = self.margwind_widget.isChecked())
        self.min_zoom_line = None
        self.max_zoom_line = None
        self.zoom_box = None
        self.background = None

        self.coatimezoom_ax = plot_coa_time(self.coatimezoom_ax, self.scanmseed, self.trigger_run,
                                   mintime=focustime-half_length,
                                   maxtime=focustime+half_length,
                                   plot_minevent=self.minevint_widget.isChecked(),
                                   plot_margwindow = self.margwind_widget.isChecked())
        self.min_zoom_line_zoom = None
        self.max_zoom_line_zoom = None
        self.zoom_box_zoom = None
        self.backgroundzoom = None

        
        # join the relevant axes
        lat_ax.sharey(mapview_ax)
        long_ax.sharex(mapview_ax)
        

        # get and plot the 

        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)

        self.canvas.mpl_connect("axes_enter_event", self.on_enter_axes)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        print("YES")
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.mouse_press = None

    def on_enter_axes(self, event):
        self.canvas.setFocus()
        # print('enter_axes', event.inaxes)
        # if event.inaxes == self.coatime_ax:
        #     event.inaxes.patch.set_facecolor('yellow')
        #     event.canvas.draw()  
    def on_mouse_click(self, event):
        self.mouse_press = (event.xdata, event.ydata)
        if event.button == 1 and event.inaxes == self.coatime_ax:
            pass
        elif event.button == 1 and event.inaxes == self.coatimezoom_ax:
            pass
        else:
            return
        
        # this is creating a zoom window for the axes below
        if self.min_zoom_line:
            self.min_zoom_line.remove()
            self.max_zoom_line.remove()
            self.zoom_box.remove()

        self.min_zoom_line = self.coatime_ax.axvline(event.xdata, c="r")
        self.max_zoom_line = self.coatime_ax.axvline(event.xdata, c="r")
        self.zoom_box = self.coatime_ax.axvspan(self.mouse_press[0], event.xdata, 
                                            ec="none", fc="r", alpha=0.5)
        if event.inaxes == self.coatimezoom_ax:
            self.min_zoom_line_zoom = self.coatimezoom_ax.axvline(event.xdata, c="r")
            self.max_zoom_line_zoom = self.coatimezoom_ax.axvline(event.xdata, c="r")
            self.zoom_box_zoom = self.coatimezoom_ax.axvspan(self.mouse_press[0], event.xdata, 
                                                ec="none", fc="r", alpha=0.5)
        self.canvas.draw()

        self.zoom_box.set_animated(True)
        self.max_zoom_line.set_animated(True)
        self.background = self.canvas.copy_from_bbox(self.zoom_box.axes.bbox)
        self.coatime_ax.draw_artist(self.zoom_box)
        self.coatime_ax.draw_artist(self.max_zoom_line)
        self.canvas.blit(self.coatime_ax.bbox)

        if event.inaxes == self.coatimezoom_ax:
            self.zoom_box_zoom.set_animated(True)
            self.max_zoom_line_zoom.set_animated(True)
            self.backgroundzoom = self.canvas.copy_from_bbox(self.zoom_box_zoom.axes.bbox)
            self.coatimezoom_ax.draw_artist(self.zoom_box_zoom)
            self.coatimezoom_ax.draw_artist(self.max_zoom_line_zoom)
            self.canvas.blit(self.coatimezoom_ax.bbox) 
    def on_mouse_move(self, event):
        if event.button == 1 and event.inaxes == self.coatime_ax:
            pass
        elif event.button == 1 and event.inaxes == self.coatimezoom_ax:
            pass
        else:
            return
        
        newxy = self.zoom_box.get_xy()
        newxy[2,0] = event.xdata
        newxy[3,0] = event.xdata
        self.zoom_box.set_xy(newxy)
        self.max_zoom_line.set_xdata([event.xdata, event.xdata])
        self.canvas.restore_region(self.background)
        self.coatime_ax.draw_artist(self.zoom_box)
        self.coatime_ax.draw_artist(self.max_zoom_line)
        self.canvas.blit(self.coatime_ax.bbox)

        if event.inaxes == self.coatimezoom_ax:
            newxy = self.zoom_box_zoom.get_xy()
            newxy[2,0] = event.xdata
            newxy[3,0] = event.xdata
            self.zoom_box_zoom.set_xy(newxy)
            self.max_zoom_line_zoom.set_xdata([event.xdata, event.xdata])
            self.canvas.restore_region(self.backgroundzoom)
            self.coatimezoom_ax.draw_artist(self.zoom_box_zoom)
            self.coatimezoom_ax.draw_artist(self.max_zoom_line_zoom)
            self.canvas.blit(self.coatimezoom_ax.bbox)
    def on_mouse_release(self, event):
        if event.button == 1 and event.inaxes == self.coatime_ax:
            xdata = event.xdata
        elif event.button == 1 and event.inaxes == self.coatimezoom_ax:
            xdata = event.xdata
        elif not self.background:
            return
        else:
            xdata = self.coatime_ax.get_xlim()[1]
            
        
        self.max_zoom_line.set_animated(False)
        self.zoom_box.set_animated(False)
        self.background = None

        self.coatimezoom_ax.set_xlim(*sorted([self.mouse_press[0], xdata]))
        if self.min_zoom_line_zoom:
            self.min_zoom_line_zoom.remove()
            self.max_zoom_line_zoom.remove()
            self.zoom_box_zoom.remove()
            self.backgroundzoom = None
        self.canvas.draw()

        print(self.mouse_press[0], xdata)
        self.mouse_press = None
    def _go_home(self):
        focustime = pd.to_datetime(self.focustime_widget.dateTime().toPython(), utc=True)
        half_length = pd.to_timedelta(int(self.focuslength_widget.text()), unit="S") / 2
        self.coatimezoom_ax.set_xlim(focustime-half_length, focustime+half_length)
        self.coatime_ax.set_xlim(focustime-half_length, focustime+half_length)
    def on_key_press(self, event):
        print("Key Press", event.key)
        if event.key == "h":
            self._go_home()
            self.canvas.draw()
        

def plot_mapview(ax, stations, events, 
                 color_col="COA_NORM", vmin=None, vmax=None,
                 markersize=2):
    
    # sort the the color column
    events.sort_values(color_col, inplace=True, ascending=True)

    ax.scatter(events.Xproj, events.Yproj, c=events.loc[:,color_col],
                vmin=vmin, vmax=vmax, s=markersize)
    ax.plot(stations.Xproj, stations.Yproj, "k^")
    # ax.set_aspect(1)

    return ax

def plot_longitude_crosssection(ax, stations, events,
                 color_col="COA_NORM", vmin=None, vmax=None,
                 markersize=2):
    
    # sort the the color column
    events.sort_values(color_col, inplace=True, ascending=True)

    ax.scatter(events.Xproj, events.Zproj, c=events.loc[:,color_col],
                vmin=vmin, vmax=vmax, s=markersize)
    ax.plot(stations.Xproj, stations.Zproj, "k^")
    ax.invert_yaxis()
    # ax.set_aspect(1)

    return ax

def plot_latitude_crosssection(ax, stations, events, 
                 color_col="COA_NORM", vmin=None, vmax=None,
                 markersize=2):
    # sort the the color column
    events.sort_values(color_col, inplace=True, ascending=True)

    ax.scatter(events.Zproj, events.Yproj, c=events.loc[:,color_col],
                vmin=vmin, vmax=vmax, s=markersize)
    ax.plot(stations.Zproj, stations.Yproj, "k^")
    # ax.set_aspect(1)

    return ax

def plot_coa_hist(ax, events, normcol="COA_NORM"):
    ax.hist(events.loc[:,normcol], bins=20, fc="gray", ec="k")
    ax.set_ylabel("Count")
    ax.set_xlabel(normcol)

    return ax

def plot_coa_time(ax, stream, trigger, 
                  mintime, maxtime, 
                  plot_minevent=False,
                  plot_margwindow=False, coa="COA_NORM"):

    for tr in stream:
        ax.plot_date(tr.times("matplotlib"), tr.data/1e5, "k-", lw=1)
        threshold = trigger.get_threshold(tr.data/1e5, tr.stats.sampling_rate)
        ax.plot(tr.times("matplotlib"), threshold, "g-", lw=1)

    events = trigger.get_events(datemin=mintime, datemax=maxtime)
    

    for _, row in events.iterrows():
        ax.axvline(row.CoaTime, lw=0.5, c="darkblue")

    ax.set_ylabel(coa)
    ax.set_xlim(mintime, maxtime)
    return ax

if __name__ == '__main__':
    # app = QApplication([])
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    prj = Project("/space/tg286/iceland/reykjanes/qmigrate/may21/nov23_dyke")
    window = DetectTriggerWindow(prj)
    window.show()
    app.exec()