import matplotlib
from PySide6.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QListWidget, QWidget
from PySide6.QtWidgets import QCheckBox, QPushButton, QDateTimeEdit, QListWidgetItem
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QLineEdit, QMessageBox, QTabWidget
from PySide6.QtWidgets import QRadioButton
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QApplication
from PySide6 import QtCore
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
from obspy import UTCDateTime
import mplstereonet
plt.close("all")
import os, shutil

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd

from projects import *
from util import Filter, Taper, preprocess_stream, time_offset

from fpfit import fp_grid_search, prad, \
            plot_all_fault_planes, plot_fault_plane_histogram, \
            plot_PT_errorbar, plot_PT_axes

manual_pick_columns = ["Network", "Station", "Location", "Channel", 
                       "Quality", "Polarity", "Onset", "Phase",
                       "PickTime", "Amplitude", "Period"]
manual_pick_index = pd.MultiIndex.from_arrays([[],[]], names=["Station", "Phase"])

zoom_params = {"in":0.1, "out":-0.1, "up":0.05, "down":-0.05}

class PickWindow(QWidget):
    def __init__(self, locate_run, event):
        super().__init__()
        
        self.project = locate_run.project  
        self.locate_run = locate_run
        self.current_event = event
        self.define_parameters()

        self.init_ui()

        self.setup_connections()

    def define_parameters(self):
        ## manual_picks
        pass

    def init_ui(self):   
        layout = QVBoxLayout() 

        # Add a tab widget
        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget)

        # Add tabs to the tab widget
        self.tab1 = Picking(self.locate_run, self.current_event)
        self.tab2 = NonLinLoc(self.locate_run, self.current_event)
        self.tab3 = Wadati(self.locate_run, self.current_event.EventID)
        self.tab4 = FPS(self.locate_run)
        self.tab5 = TravelTime(self.locate_run, self.current_event)

        self.tab_widget.addTab(self.tab1, 'Picking')
        self.tab_widget.addTab(self.tab2, 'NonLinLoc')
        self.tab_widget.addTab(self.tab3, 'Wadati')
        self.tab_widget.addTab(self.tab4, 'Fault Plane Solution')
        self.tab_widget.addTab(self.tab5, 'Travel Time')

        # pd.merge(self.tab1.manual_picks, self.tab2.current_nll_event.phases, 
        #                          left_index=True, right_index=True, how="outer")

        self.setLayout(layout)

    def tab_bar_clicked(self, index):
        if index == 3:
            # Fault Plane Solution
            if len(self.tab2.current_nll_event.phases) > 0:
                self.tab4.get_picks(pd.merge(self.tab1.manual_picks, self.tab2.current_nll_event.phases, 
                                    left_index=True, right_index=True, how="outer"))
        if index == 2: 
            # wadati
            self.tab3.get_picks(self.tab1.manual_picks)
            try:
                self.tab3.read_nonlinloc_event(os.path.join(self.locate_run.project.nll_settings.root, 
                                                        "manual_locations", f"{self.current_event.EventID}.hyp"))
            except FileNotFoundError:
                pass
            self.tab3.init_plot()

        if index == 1:
            # nonlinloc
            self.tab2.get_picks(self.tab1.manual_picks)
            self.tab2.replot("STATIONS")

        if index == 4:
            # travel time
            self.tab5.get_manual_picks()
            if len(self.tab5.manual_picks) > 0:
                self.tab5.plot_manual_picks()
                self.tab5.plot_manual_traveltimes()

    def setup_connections(self):
        self.tab_widget.tabBarClicked.connect(self.tab_bar_clicked)

class Picking(QWidget):    
    
    def __init__(self, locate_run, event):
        super().__init__()
        self.locate_run = locate_run
        self.current_event = event
        self.define_parameters()
        self.init_ui()
        self.setup_connections()

        os.makedirs(os.path.join(self.locate_run.project.nll_settings.root, "manual_picks"), exist_ok=True)
        os.makedirs(os.path.join(self.locate_run.project.nll_settings.root, "manual_locations"), exist_ok=True)

    def plot_waveforms(self):
        if not self.raw_stream:
            self.load_waveforms()
        
        self.pick_cursors = []
        if not self.current_station:
            self.current_station = self.qm_picks.sort_values("hyp_distance").index.get_level_values(0)[0]
            # self.current_station = "LYNG"
            self.set_current_station_in_list()

        ymax = np.max([np.max(np.abs(tr.data)) for tr in self.proc_stream.select(station=self.current_station)]) * 1.1
        self.drawn_waveforms = {}
        for ax, comp in zip(self.axes, "ZNE"):
            self.drawn_waveforms[ax] = []
            ax.axvline(0, c="g", ls="--", lw=0.5)
            ax.set_yticklabels([])
            self.pick_cursors.append(ax.axvline(0, c="r"))

            st = self.proc_stream.select(station=self.current_station, component=comp)
            if len(st) == 0:
                self.drawn_waveforms[ax].append(ax.axhline(0, c="gray"))
                continue

            for tr in st:
                self.drawn_waveforms[ax].append(ax.plot(tr.times(reftime=UTCDateTime(self.current_event.DT)),
                                                         tr.data, "k-"))
            ax.set_ylim(-ymax, ymax)
        
        # #set a background for blitting
        # self.background = [self.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
        
        # plot the picks
        self.plot_qm_picks(self.current_station)
        if not (self.current_station, "P") in self.manual_picks.index:
            self.manual_ppick_lines = [ax.axvline(-1e6, ymin=0.2, ymax=0.8, c="r", lw=1, picker=True) for ax in self.axes]
            # annotation
            self.p_annotate = self.Z_ax.annotate("?P??", xy=(-1e6, 0.9),
                                                xycoords=self.Z_ax.get_xaxis_transform(), 
                                                bbox={"facecolor":"white", "alpha":1, "boxstyle":"round", "ec":"red"}, 
                                                horizontalalignment="right",
                                                verticalalignment="bottom")
        else:
            pick = self.manual_picks.loc[(self.current_station, "P"),:]
            self.manual_ppick_lines = [
                ax.axvline(
                    time_offset(
                        pick.PickTime, 
                        self.current_event.DT), 
                    ymin=0.2, ymax=0.8, c="r", lw=1, 
                    picker=True) 
                for ax in self.axes]
            annotate_string = f"{pick.Onset if pd.notna(pick.Onset) else '?'}{pick.Phase}{pick.Quality}{pick.Polarity if pd.notna(pick.Polarity) else '?'}"
            self.p_annotate = self.Z_ax.annotate(annotate_string, xy=(time_offset(pick.PickTime, self.current_event.DT), 0.9),
                                                xycoords=self.Z_ax.get_xaxis_transform(), 
                                                bbox={"facecolor":"white", "alpha":1, "boxstyle":"round", "ec":"red"}, 
                                                horizontalalignment="right",
                                                verticalalignment="bottom")
        if not (self.current_station, "S") in self.manual_picks.index:
            self.manual_spick_lines = [ax.axvline(-1e6, ymin=0.2, ymax=0.8, c="b", lw=1, picker=True) for ax in self.axes]
            self.s_annotate = [ax.annotate("?S?", xy=(-1e6, 0.9),
                                    xycoords=ax.get_xaxis_transform(), 
                                    bbox={"facecolor":"white", "alpha":1, "boxstyle":"round", "ec":"blue"}, 
                                    horizontalalignment="right",
                                    verticalalignment="bottom") for ax in self.axes[1:]]
        else:
            pick = self.manual_picks.loc[(self.current_station, "S"),:]
            self.manual_spick_lines = [
                ax.axvline(
                    time_offset(
                        self.manual_picks.loc[(self.current_station, "S"), "PickTime"], 
                        self.current_event.DT), 
                    ymin=0.2, ymax=0.8, c="b", lw=1, 
                    picker=True) 
                for ax in self.axes]
            annotate_string = f"{pick.Onset if pd.notna(pick.Onset) else '?'}{pick.Phase}{pick.Quality}"
            self.s_annotate = [ax.annotate(annotate_string, xy=(time_offset(pick.PickTime, self.current_event.DT), 0.9),
                        xycoords=ax.get_xaxis_transform(), 
                        bbox={"facecolor":"white", "alpha":1, "boxstyle":"round", "ec":"blue"}, 
                        horizontalalignment="right",
                        verticalalignment="bottom") for ax in self.axes[1:]]
        
        if not self.xlim:
            self.xlim = (-2, 10)
        self.ylim = (-ymax, ymax)
        
        self.Z_ax.set_xlim(self.xlim)
        self.Z_ax.set_title(self.current_station)

        # for cursor in self.pick_cursors:
        #     cursor.set_animated(True)
        self.canvas.draw()
        self.canvas.setFocus()


        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("axes_enter_event", self.on_enter_axes)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)     
        # self.canvas.mpl_connect("pick_event", self.on_select_pick)        
    
    def plot_qm_picks(self, station):
        
        if (station, "P") in self.qm_picks.index:
            Ppicks = self.qm_picks.loc[(station, "P"),:]
            modeltime = Ppicks.ModelledTime.tz_convert(None)
            self.xlim = (time_offset(modeltime,self.current_event.DT)-2,
                         time_offset(modeltime,self.current_event.DT)+4)
            picktime = Ppicks.PickTime
            self.Z_ax.axvline(time_offset(modeltime,self.current_event.DT), c="darkred", lw=0.5)
            if not picktime == "-1":
                picktime = pd.to_datetime(picktime).tz_convert(None)
                self.Z_ax.axvline(time_offset(picktime,self.current_event.DT), c="red", lw=0.5)
                self.xlim = (time_offset(picktime,self.current_event.DT)-2,
                                time_offset(picktime,self.current_event.DT)+4)


        if (station, "S") in self.qm_picks.index:
            Spicks = self.qm_picks.loc[(station, "S"),:]   
            modeltime = Spicks.ModelledTime.tz_convert(None)
            picktime = Spicks.PickTime
            self.E_ax.axvline(time_offset(modeltime,self.current_event.DT), c="darkblue", lw=0.5)
            self.N_ax.axvline(time_offset(modeltime,self.current_event.DT), c="darkblue", lw=0.5)
            if not picktime == "-1":
                picktime = pd.to_datetime(picktime).tz_convert(None)
                self.E_ax.axvline(time_offset(picktime,self.current_event.DT), c="blue", lw=0.5)
                self.N_ax.axvline(time_offset(picktime,self.current_event.DT), c="blue", lw=0.5)

    def draw_current_pick(self):
        if self.current_pick.phase == "P":
            lines = [self.manual_ppick_lines, self.manual_spick_lines]
            annotations = [[self.p_annotate], self.s_annotate]
            annotate_string = f"{self.current_pick.onset if pd.notna(self.current_pick.onset) else '?'}{self.current_pick.phase}{self.current_pick.quality}{self.current_pick.polarity if pd.notna(self.current_pick.polarity) else '?'}"
        else:
            lines = [self.manual_spick_lines, self.manual_ppick_lines]
            annotations = [self.s_annotate, [self.p_annotate]]
            annotate_string = f"{self.current_pick.onset if pd.notna(self.current_pick.onset) else '?'}{self.current_pick.phase}{self.current_pick.quality}"

        for line in lines[0]:
            line.set_xdata([time_offset(self.current_pick.picktime,self.current_event.DT)])
            line.set_linewidth(2)
        for line in lines[1]:
            line.set_linewidth(1)

        for annot in annotations[0]:
            annot.set_position((time_offset(self.current_pick.picktime,self.current_event.DT), 0.9))
            annot.set_fontweight("bold")
            annot.set_text(annotate_string)
        for annot in annotations[1]:
            annot.set_fontweight("normal")
        self.canvas.draw()

    def make_pick(self, xpos, ax):
        # add a pick to the manual picks dataframe
        manual_pick_time = self.current_event.DT + pd.to_timedelta(xpos, unit="S")
        manual_pick_phase = self.current_phase
        manual_pick_station = self.current_station
        if ax == self.Z_ax:
            manual_pick_component = "Z"
        elif ax == self.N_ax:
            manual_pick_component = "N"
        elif ax == self.E_ax:
            manual_pick_component = "E"
        try:
            manual_pick_location = self.waveform_lookup.loc[(self.current_station, manual_pick_component), "location"]
            manual_pick_network = self.waveform_lookup.loc[(self.current_station, manual_pick_component), "network"]
            manual_pick_channel = self.waveform_lookup.loc[(self.current_station, manual_pick_component), "channel"]
        except KeyError:
            print("NO data here")
            return

        if manual_pick_phase == "P":
            manual_pick_quality = 0
        else:
            manual_pick_quality = 1
        manual_pick_onset = "I"
        manual_pick_polarity = None
        manual_pick_amplitude = None
        manual_pick_period = None

        self.current_pick = Pick(manual_pick_network, manual_pick_station, manual_pick_location,
                                 manual_pick_channel, manual_pick_phase, manual_pick_time, 
                                 manual_pick_quality, manual_pick_onset, manual_pick_polarity,
                                 manual_pick_amplitude, manual_pick_period)
        
        # self.manual_picks.loc[(manual_pick_station, manual_pick_phase),:] = self.current_pick.get_pandas_dataframe().loc[(manual_pick_station, manual_pick_phase),:]
        

        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Network"] = manual_pick_network
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Station"] = manual_pick_station
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Location"] = manual_pick_location
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Channel"] = manual_pick_channel
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Quality"] = manual_pick_quality
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Polarity"] = manual_pick_polarity
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Onset"] = manual_pick_onset
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Phase"] = manual_pick_phase
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"PickTime"] = manual_pick_time
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Amplitude"] = manual_pick_amplitude
        self.manual_picks.loc[(manual_pick_station, manual_pick_phase),"Period"] = manual_pick_period
        self.manual_picks.loc[:,"Quality"] = self.manual_picks.loc[:,"Quality"].astype(int)
        self.manual_picks.Quality = self.manual_picks.loc[:,"Quality"].astype(int)
        # print(self.manual_picks.loc[(manual_pick_station, manual_pick_phase),:])
        # print(self.manual_picks)
    
        self.draw_current_pick()

    def deselect_pick(self):
        self.current_pick = None
        lines = self.manual_ppick_lines + self.manual_spick_lines
        annotations = [self.p_annotate] + self.s_annotate
        for line in lines[0]:
            line.set_linewidth(1)
        for annot in annotations:
            annot.set_fontweight("normal")
        self.canvas.draw()        
    
    def redraw_waveforms(self):

        # remove existing lines
        for ax in self.axes:
            for lines in self.drawn_waveforms[ax]:
                if not isinstance(lines, list):
                    lines.remove()
                    continue
                for line in lines:
                    line.remove()
        
        # now plot new lines
        ymax = np.max([np.max(np.abs(tr.data)) for tr in self.proc_stream.select(station=self.current_station)]) * 1.1
        self.drawn_waveforms = {}
        for ax, comp in zip(self.axes, "ZNE"):
            self.drawn_waveforms[ax] = []

            st = self.proc_stream.select(station=self.current_station, component=comp)
            if len(st) == 0:
                self.drawn_waveforms[ax].append(ax.axhline(0, c="gray"))
                continue
            for tr in st:
                self.drawn_waveforms[ax].append(ax.plot(tr.times(reftime=UTCDateTime(self.current_event.DT)),
                                                         tr.data, "k-"))
            ax.set_ylim(-ymax, ymax)
        self.ylim = (-ymax, ymax)
        self.canvas.draw()

    def set_polarity(self, direction):
        if not self.current_pick or self.current_pick.phase == "S":
            return
        
        if direction == "up":
            polarity = "U"
        elif direction == "down":
            polarity = "D"
        elif direction == "none":
            polarity = None
        else:
            raise ValueError("HELP IN POLARITY")

        self.current_pick.polarity = polarity
        self.manual_picks.loc[(self.current_station, self.current_phase),"Polarity"] = polarity  

        annotate_string = f"{self.current_pick.onset if pd.notna(self.current_pick.onset) else '?'}{self.current_pick.phase}{self.current_pick.quality}{self.current_pick.polarity if pd.notna(self.current_pick.polarity) else '?'}"
        self.p_annotate.set_text(annotate_string)
        self.canvas.draw()

    def set_onset(self, onset):

        print(self.manual_picks.loc[(self.current_station, self.current_phase),"Onset"])
        self.current_pick.onset = onset
        self.manual_picks.loc[(self.current_station, self.current_phase),"Onset"] = onset  
        print("REALLY SETTING ONSET")
        print(self.manual_picks.loc[(self.current_station, self.current_phase),"Onset"])

        if self.current_phase == "P":
            annotate_string = f"{self.current_pick.onset if pd.notna(self.current_pick.onset) else '?'}{self.current_pick.phase}{self.current_pick.quality}{self.current_pick.polarity if pd.notna(self.current_pick.polarity) else '?'}"
            self.p_annotate.set_text(annotate_string)
        else:
            annotate_string = f"{self.current_pick.onset if pd.notna(self.current_pick.onset) else '?'}{self.current_pick.phase}{self.current_pick.quality}"
            for annot in self.s_annotate:
                annot.set_text(annotate_string)
        self.canvas.draw()

    def set_quality(self, quality):

        self.current_pick.quality = quality
        self.manual_picks.loc[(self.current_station, self.current_phase),"Quality"] = quality  

        if self.current_phase == "P":
            annotate_string = f"{self.current_pick.onset if pd.notna(self.current_pick.onset) else '?'}{self.current_pick.phase}{self.current_pick.quality}{self.current_pick.polarity if pd.notna(self.current_pick.polarity) else '?'}"
            self.p_annotate.set_text(annotate_string)
        else:
            annotate_string = f"{self.current_pick.onset if pd.notna(self.current_pick.onset) else '?'}{self.current_pick.phase}{self.current_pick.quality}"
            for annot in self.s_annotate:
                annot.set_text(annotate_string)
        self.canvas.draw()

    def move_cursor(self, xpos):
        # self.canvas.restore_region(self.background)
        for cursor in self.pick_cursors:
            cursor.set_xdata([xpos,xpos])
            # cursor.axes.draw_artist(cursor)
        # self.canvas.blit(cursor.axes.bbox)
        
    def set_cursor_color(self, ax):
        if not self.selected_phase and ax == self.Z_ax:
            self.current_phase = "P"
            for cursor in self.pick_cursors:
                cursor.set_color("r")
        elif not self.selected_phase and (ax == self.N_ax or ax == self.E_ax):
            self.current_phase = "S"
            for cursor in self.pick_cursors:
                cursor.set_color("b")

    def zoom(self, type, xpos, step):
        if type == "in" or type == "out":
            R = zoom_params[type]
            x0, x1 = self.xlim
            a, b = xpos-x0, x1-xpos
            x0_, x1_ = x0+((a*R)*step), x1-((b*R)*step)
            self.xlim = (x0_, x1_)
        elif type == "up" or type == "down":
            R = zoom_params[type]
            y0, y1 = self.ylim
            yrange = y1 - y0
            self.ylim = (y0+((yrange*R)*step), y1-((yrange*R)*step))

        for ax in self.axes:
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            
    def drag_traces(self, xpos):
        shift = xpos - self.mouse_click_start_pos[1]
        self.xlim = (self.mouse_click_start_pos[0][0]-shift, self.mouse_click_start_pos[0][1]-shift)

        for ax in self.axes:
            ax.set_xlim(self.xlim)

    def shift_traces(self, direction):
        shift = (self.xlim[1]-self.xlim[0]) * 0.1
        if direction == "left":
            shift *= -1
        self.xlim = (self.xlim[0]+shift, self.xlim[1]+shift)
        for ax in self.axes:
            ax.set_xlim(self.xlim)

    def on_select_pick(self, event):
    #     print("mouse pick", type(event), event)
    #     if not event.mouseevent.button == 1:
    #         return
    #     artist = event.artist

    #     if artist in self.manual_ppick_lines:
    #         ## it is a p pick
    #         pass
    #     elif artist in self.manual_spick_lines:
    #         ## it is an s pick
    #         pass
    #     else:
    #         return
        pass

    def on_mouse_move(self, event):
        if not event.inaxes:
            return
        if event.button == 1:
            # self.drag_traces(event.xdata)
            return
        elif event.button == 3:
            return
        else:
            self.move_cursor(event.xdata)
            self.set_cursor_color(event.inaxes)
        self.canvas.draw()
    
    def on_scroll(self, event):
        if not event.inaxes:
            return
        if not self.y_zoom_mode and event.button == "up":
            self.zoom("in", event.xdata, np.abs(event.step))
        elif not self.y_zoom_mode and event.button == "down":
            self.zoom("out", event.xdata, np.abs(event.step))
        elif self.y_zoom_mode and event.button == "up":
            self.zoom("up", event.xdata, np.abs(event.step))
        elif self.y_zoom_mode and event.button == "down":
            self.zoom("down", event.xdata, np.abs(event.step))

        self.canvas.draw()

    def on_mouse_click(self, event):
        print("mouse click", type(event), event)
        if not event.inaxes:
            return
        if event.button == 1:
            self.mouse_click_start_pos = self.xlim, event.xdata
        return
    def on_mouse_release(self, event):
        print("mouse release", type(event), event)
        if event.button == 1:
            self.mouse_click_start_pos = None
        if event.inaxes:
            self.make_pick(event.xdata, event.inaxes)
    def on_key_press(self, event):
        print("PRESS", event.key, repr(event.key))
        if event.key == "n":
            self._next_station()
        elif event.key == "p":
            self._previous_station()
        if not event.inaxes:
            return
        if event.key == "shift":
            self.y_zoom_mode = True
        elif event.key == "left":
            self.shift_traces("left")
        elif event.key == "right":
            self.shift_traces("right")
        elif event.key == "up":
            self.zoom("up", event.xdata, 1)
        elif event.key == "down":
            self.zoom("down", event.xdata, 1)
        elif event.key == "u" or event.key == "+":
            self.set_polarity("up")
        elif event.key == "d" or event.key == "-":
            self.set_polarity("down")
        elif event.key == "c":
            self.set_polarity("none")
        elif event.key == "i":
            self.set_onset("I")
            print("SETTING ONSET")
        elif event.key == "e":
            self.set_onset("E")
        elif event.key == "0":
            self.set_quality(0)
        elif event.key == "1":
            self.set_quality(1)
        elif event.key == "2":
            self.set_quality(2)
        elif event.key == "3":
            self.set_quality(3)
        elif event.key == "4":
            self.set_quality(4)
        elif self.current_pick and event.key == "escape":
            self.deselect_pick()
        elif event.key == " ":
            print("SPACEBAR")
            self._set_filter_active()
        # elif event.key == "p":
        #     # self.change_phase("P")
        #     print("NOT IMPLEMENTED")
        # elif event.key == "s":
        #     # self.change_phase("S")
        #     print("NOT IMPLEMENTED")
        else:
            return
        self.canvas.draw()
    def on_key_release(self, event):
        print("key release")
        if not event.inaxes:
            return
        if event.key == "shift":
            self.y_zoom_mode = False
        else:
            return  
    def on_enter_axes(self, event):
        print("enter axes")
        return

    def load_waveforms(self):
        if self._use_qm_waveforms:
            self.raw_stream = self.locate_run.get_waveforms(self.uid, processed=False)

        self.proc_stream = preprocess_stream(self.raw_stream.copy(), self.filter, self.taper, self.detrend)

        # take the streams and create a lookup dataframe
        self.waveform_lookup = pd.DataFrame({"network" : [tr.stats.network for tr in self.proc_stream],
                                             "station" : [tr.stats.station for tr in self.proc_stream],
                                             "location" : [tr.stats.location for tr in self.proc_stream],
                                             "channel" : [tr.stats.channel for tr in self.proc_stream],
                                             "component" : [tr.stats.component for tr in self.proc_stream],
                                             "starttime" : [tr.stats.starttime for tr in self.proc_stream],
                                             "endtime" : [tr.stats.endtime for tr in self.proc_stream],
                                             "sampling_rate" : [tr.stats.sampling_rate for tr in self.proc_stream]})
        self.waveform_lookup.set_index(["station", "component"], inplace=True)
        self.waveform_lookup.sort_index(inplace=True)
        self.waveform_lookup = self.waveform_lookup.loc[~self.waveform_lookup.index.duplicated(),:]
        
    def update_proc_waveforms(self):
        print(self.taper.type)
        self.proc_stream = preprocess_stream(self.raw_stream.copy(), self.filter, self.taper, self.detrend)
        
    def add_hyp_distance_to_picks(self, picks):
        stations = picks.index.get_level_values(0)
        station_info = self.locate_run.project.stations
        picks.loc[:,"hyp_distance"] = np.array(np.sqrt((self.current_event.Xproj-station_info.loc[stations,"Xproj"])**2 + 
                                    (self.current_event.Yproj-station_info.loc[stations,"Yproj"])**2))
        return picks
        
    def define_parameters(self):
        self.uid = self.current_event.EventID
        ## manual_picks
        self.manual_picks = self.locate_run.get_manual_picks(self.uid, self.locate_run.project.nll_settings.root)
        print(self.manual_picks.Quality)
        self.qm_picks = self.locate_run.get_picks(self.uid)
        self.qm_picks = self.add_hyp_distance_to_picks(self.qm_picks)
        self.current_stations = np.array(list(set(self.qm_picks.index.get_level_values(0).to_list())))
        
        # processing paramters
        self.filter = self._set_filter(self.locate_run.onset["bandpass_filters"].get("P", [None,None,None])[0],
                                       self.locate_run.onset["bandpass_filters"].get("P", [None,None,None])[1],
                                       self.locate_run.onset["bandpass_filters"].get("P", [None,None,None])[2],
                                       True)
        self.detrend = True
        self.taper = self._set_taper(0.05)

        self.raw_stream = None
        self.proc_stream = None
        self.vel_stream = None
        self.wa_stream = None

        self.current_station = None

        ## plotting
        self.xlim = None
        self.ylim = None
        self.y_zoom_mode = False
        self.mouse_click_start_pos = None

        ## picking
        self.selected_phase = None # key button press for selecting P or S phase. If None base choice on which axes
        self.current_phase = None
        self.current_pick = None


        self._use_qm_waveforms = True

    def init_ui(self):   
        page_layout = QHBoxLayout()
        left_layout = QVBoxLayout() 

        ## LABELS
        # UID
        # OTIME
        # MAGNITUDE
        # LAT
        # LON
        # DEP

        ## CLIENT TO USE - RADIO BUTTONS
        # radio button widget
        self.qmwaveforms_widget = QGroupBox(self)
        # self.qmwaveforms_widget.setCheckable(True)
        self.qmwaveforms_widget.setChecked(False)
        vbox = QVBoxLayout()
        self.qmwaveforms_button = QRadioButton("Use QM waveforms", self)
        self.archivewaveforms_button = QRadioButton("Use Archive waveforms", self)
        self.qmwaveforms_button.setChecked(True)
        vbox.addWidget(self.qmwaveforms_button)
        vbox.addWidget(self.archivewaveforms_button)
        vbox.addStretch(1)
        self.qmwaveforms_widget.setLayout(vbox)
        
        self.maxamplitude_widget = QPushButton("Get Max Amplitude")
        self.loadwaveforms_widget = QPushButton("Load Waveforms")
        self.savepicks_widget = QPushButton("Save Picks")

        ## CLIENT TO USE - RADIO BUTTONS
        # radio button widget
        self.removeresponse_widget = QGroupBox("Response Removal", self)
        self.removeresponse_widget.setCheckable(True)
        self.removeresponse_widget.setChecked(False)
        vbox = QVBoxLayout()
        self.velocity_button = QRadioButton("Remove Response", self)
        self.woodanderson_button = QRadioButton("Apply Wood-Anderson Response", self)
        self.velocity_button.setChecked(True)
        vbox.addWidget(self.velocity_button)
        vbox.addWidget(self.woodanderson_button)
        vbox.addStretch(1)
        self.removeresponse_widget.setLayout(vbox)                
        
        # list of stations with distance
        self.station_list_widget = QListWidget(self)
        self.load_list_of_stations(self.current_stations)
        
        ## add the matplotlib figure canvas
        self.fig, self.axes = plt.subplots(3, 1, sharex=True)
        self.Z_ax = self.axes[0]
        self.N_ax = self.axes[1]
        self.E_ax = self.axes[2]
        plt.tight_layout(pad=1e-5, h_pad=1e-5, w_pad=1e-5)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )

        # low and highpass filters
        self.lowpass_widget = QLineEdit(str(self.filter.lowpass), self,
                                                validator=QDoubleValidator(bottom=0.))
        self.highpass_widget = QLineEdit(str(self.filter.highpass), self,
                                                validator=QDoubleValidator(bottom=0.))
        self.corners_widget = QLineEdit(str(self.filter.corners), self,
                                                validator=QIntValidator(2, 10))
        self.zerophase_widget = QCheckBox("Zerophase", self)
        self.zerophase_widget.setChecked(True)
        self.detrend_widget = QCheckBox("Detrend", self)
        self.detrend_widget.setChecked(True)
        self.taper_widget = QCheckBox("Taper", self)
        self.taper_widget.setChecked(True)
        self.maxpercentage_widget = QLineEdit(str(self.taper.maxpercentage), self,
                                                validator=QDoubleValidator(bottom=0.01, top=0.5))

        left_layout.addWidget(self.qmwaveforms_widget)
        left_layout.addWidget(self.loadwaveforms_widget)
        left_layout.addWidget(self.maxamplitude_widget)
        left_layout.addWidget(self.removeresponse_widget)

        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("Highpass Freq [Hz]"))
        line_layout.addWidget(self.highpass_widget)
        left_layout.addLayout(line_layout)
        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("Lowpass Freq [Hz]"))
        line_layout.addWidget(self.lowpass_widget)
        left_layout.addLayout(line_layout)
        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("N Corners"))
        line_layout.addWidget(self.corners_widget)
        left_layout.addLayout(line_layout)

        left_layout.addWidget(self.zerophase_widget)
        left_layout.addWidget(self.detrend_widget)
        left_layout.addWidget(self.taper_widget)

        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("Max Percentage"))
        line_layout.addWidget(self.maxpercentage_widget)
        left_layout.addLayout(line_layout)

        left_layout.addWidget(self.station_list_widget)
        left_layout.addWidget(self.savepicks_widget)
        left_layout.addStretch(1)

        page_layout.addLayout(left_layout)
        page_layout.addWidget(self.canvas, stretch=10)
        self.setLayout(page_layout)
        
    def setup_connections(self):
        self.loadwaveforms_widget.clicked.connect(self.plot_waveforms)
        self.savepicks_widget.clicked.connect(self.save_picks)
        self.highpass_widget.editingFinished.connect(self._set_highpass_filter)
        self.lowpass_widget.editingFinished.connect(self._set_lowpass_filter)
        self.corners_widget.editingFinished.connect(self._set_corners)
        self.zerophase_widget.stateChanged.connect(self._set_zerophase)
        self.detrend_widget.stateChanged.connect(self._set_detrend)
        self.taper_widget.stateChanged.connect(self._set_use_taper)
        self.maxpercentage_widget.editingFinished.connect(self._set_maxpercentage)
        self.station_list_widget.itemClicked.connect(self._set_current_station)

    def save_picks(self):
        print("SAVING PICKS", self.manual_picks)
        self.locate_run.save_manual_picks(self.uid, self.manual_picks, self.locate_run.project.nll_settings.root)
    def load_list_of_stations(self, stations):
        self.station_list_widget.clear()
        self.list_of_stations = [QListWidgetItem(s, self.station_list_widget) for s in stations]
    def _next_station(self):
        index_current = list(self.current_stations).index(self.current_station)
        next_index = index_current + 1
        if next_index == self.current_stations.shape[0]:
            next_index = 0
        self.station_list_widget.setCurrentRow(next_index)
        self.station_list_widget.scrollToItem(self.station_list_widget.currentItem())
        self._set_current_station(self.station_list_widget.currentItem())
    def _previous_station(self):
        index_current = list(self.current_stations).index(self.current_station)
        previous_index = index_current - 1
        if previous_index < 0:
            previous_index = self.current_stations.shape[0] - 1
        self.station_list_widget.setCurrentRow(previous_index)
        self.station_list_widget.scrollToItem(self.station_list_widget.currentItem())
        self._set_current_station(self.station_list_widget.currentItem())
    def _set_current_station(self, item):
        if not self.raw_stream:
            return
        print("NEW_STATION", item.text())
        self.current_station = item.text()

        # clear the axes and redraw
        for ax in self.axes:
            ax.clear()
        self.plot_waveforms()
    def _set_highpass_filter(self):
        self.filter.highpass = float(self.highpass_widget.text())
        if not self.raw_stream:
            return
        self.update_proc_waveforms()
        self.redraw_waveforms()
    def _set_filter_active(self):
        if self.filter.type == "none":
            self.filter.highpass = float(self.highpass_widget.text())
            self.filter.lowpass = float(self.lowpass_widget.text())
            self.lowpass_widget.setReadOnly(False)
            self.highpass_widget.setReadOnly(False)
            self.corners_widget.setReadOnly(False)
        else:
            self.filter.type = "none"
            self.lowpass_widget.setReadOnly(True)
            self.highpass_widget.setReadOnly(True)
            self.corners_widget.setReadOnly(True)
        if not self.raw_stream:
            return
        self.update_proc_waveforms()
        self.redraw_waveforms()
    def _set_lowpass_filter(self):
        self.filter.lowpass = float(self.lowpass_widget.text())
        if not self.raw_stream:
            return
        self.update_proc_waveforms()
        self.redraw_waveforms()
    def _set_corners(self):
        self.filter.corners = int(self.corners_widget.text())
        if not self.raw_stream:
            return
        self.update_proc_waveforms()
        self.redraw_waveforms()
    def _set_zerophase(self):
        self.filter.zerophase = self.zerophase_widget.isChecked()
        if not self.raw_stream:
            return
        self.update_proc_waveforms()
        self.redraw_waveforms()
    def _set_maxpercentage(self):
        self.taper.maxpercentage = float(self.maxpercentage_widget.text())
        if not self.raw_stream:
            return
        self.update_proc_waveforms()
        self.redraw_waveforms()
    def _set_detrend(self):
        self.detrend = self.detrend_widget.isChecked()
        if not self.raw_stream:
            return
        self.update_proc_waveforms()
        self.redraw_waveforms()
    def _set_use_taper(self):
        if not self.taper_widget.isChecked():
            self.taper.type = "none"
        else:
            self.taper = self._set_taper(float(self.maxpercentage_widget.text()))
        print("SETTER", self.taper.type, self.taper_widget.isChecked())
        if not self.raw_stream:
            return
        self.update_proc_waveforms()
        self.redraw_waveforms()
    def _set_filter(self, highpass, lowpass, corners, zerophase):
        filter = Filter()
        if lowpass and highpass:
            type = "bandpass"
        elif lowpass:
            type = "lowpass"
        elif highpass:
            type = "highpass"
        else:
            type = "none"
        filter.type = type
        filter.lowpass = lowpass
        filter.highpass = highpass
        filter.corners = corners
        filter.zerophase = zerophase
        return filter
    def _set_taper(self, maxpercentage, type="cosine"):
        taper = Taper()
        taper.maxpercentage = maxpercentage
        taper.type = type
        return taper    
    def set_current_station_in_list(self):
        tmp = np.mgrid[:self.station_list_widget.count()]
        mask = self.current_stations == self.current_station
        self.station_list_widget.setCurrentItem(self.list_of_stations[tmp[mask][0]])
        self.station_list_widget.scrollToItem(self.list_of_stations[tmp[mask][0]])

class NonLinLoc(QWidget):    
    
    def __init__(self, locate_run, event):
        super().__init__()
        self.locate_run = locate_run
        self.current_event = event
        self.NLL = self.locate_run.project.nll_settings
        
        self.define_parameters()
        self.init_ui()
        self.init_plot()
        self.setup_connections()

        if os.path.exists(
                os.path.join(
                    self.NLL.root, "manual_locations", f"{self.current_event.EventID}.hyp"
                    )):
            self.current_nll_event = self.NLL.read_summed_hypfile(
                os.path.join(
                    self.NLL.root, "manual_locations", f"{self.current_event.EventID}.hyp"
                        )
                    )
            # self.replot("STATIONS")
            self.plot_nllloc()
            self.residual_ax.clear()
            self.weight_ax.clear()
            self.plot_residuals()
            self.canvas.draw()

    def replot(self, option):
        print("REPLOTTING")
        if option == "QMLOC" and not self.show_qmloc:
            print("REMOVEING")
            self.mapview_qmloc.remove()
            self.crosssectionx_qmloc.remove()
            self.crosssectiony_qmloc.remove()
        elif option == "QMLOC":
            print("ADDING")
            self.plot_qmloc()

        if option == "STATIONS" and not self.show_stations:
            print("REMOVEING")
            self.mapview_stations.remove()
            self.crosssectionx_stations.remove()
            self.crosssectiony_stations.remove()
            for statplot in self.mapview_stationstext:
                statplot.remove()
        elif option == "STATIONS":
            print("ADDING")
            self.plot_stations()

        if option == "SCATTER" and not self.show_scatter:
            print("REMOVEING")
            self.mapview_scatter.remove()
            self.crosssectionx_scatter.remove()
            self.crosssectiony_scatter.remove()
        elif option == "SCATTER":
            print("ADDING")
            self.plot_scatter()

        self.canvas.draw()

    def plot_residuals(self):
        npicks = len(self.current_nll_event.phases.Residual)
        # colours = ["red" if res < 0 else "blue" for res in self.current_nll_event.phases.Residual]
        colours = []
        for index in self.current_nll_event.phases.index:
            if self.current_nll_event.phases.loc[index, "Weight"] == 0:
                colours.append("gray")
            elif self.current_nll_event.phases.loc[index, "Residual"] < 0:
                colours.append("red")
            elif self.current_nll_event.phases.loc[index, "Residual"] > 0:
                colours.append("blue")
            else: 
                colours.append("white")
        names = [ind[0]+" "+ind[1] for ind in  self.current_nll_event.phases.index]
        self.residual_ax.axvline(0, c="grey", lw=0.5)
        self.residual_ax.barh(y=range(npicks),
                             height=0.4, 
                              ec="k", color=colours, align="edge", 
                              width=self.current_nll_event.phases.Residual)
        self.residual_ax.set_yticks(range(npicks), names)
        # self.residual_ax.set_yticks(range(npicks))

        for i, residual in enumerate(self.current_nll_event.phases.Residual):
            if residual > 0.1:
                self.residual_ax.text(0.09, i+0.02, f"{residual:.2f} s", ha="right", va="bottom")
            if residual < -0.1:
                self.residual_ax.text(-0.09, i+0.02, f"{residual:.2f} s", ha="left", va="bottom")

        self.weight_ax.barh(y=range(npicks),
                             height=-0.4,
                              ec="k", color="green", align="edge", 
                              width=self.current_nll_event.phases.Weight, )
        # self.residual_ax.set_xscale("log")
        self.weight_ax.set_xlim(-1.5, 1.5)
        self.weight_ax.set_xlabel("Weight")
        self.weight_ax.xaxis.set_label_position("top")
        self.residual_ax.set_xlim(-0.1,0.1)
        self.residual_ax.set_xlabel("Residual [s]")
        self.residual_ax.set_ylim(-0.5, npicks-0.5)
                                    
    def plot_qmloc(self):
        self.mapview_qmloc, = self.mapview_ax.plot(self.current_event.Xproj, 
                                                  self.current_event.Yproj,
                                                  "go")
        self.crosssectionx_qmloc, = self.crosssectionx_ax.plot(self.current_event.Xproj, 
                                                  self.current_event.Zproj,
                                                  "go")
        self.crosssectiony_qmloc, = self.crosssectiony_ax.plot(self.current_event.Zproj, 
                                                  self.current_event.Yproj,
                                                  "go")

    def plot_nllloc(self):
        if self.plotted:
            for line in [self.mapview_nllloc,self.crosssectionx_nllloc,self.crosssectiony_nllloc]:
                line.remove()
        else:
            self.plotted = True
        self.mapview_nllloc, = self.mapview_ax.plot(self.current_nll_event.Xproj, 
                                                  self.current_nll_event.Yproj,
                                                  "bo")
        self.crosssectionx_nllloc, = self.crosssectionx_ax.plot(self.current_nll_event.Xproj, 
                                                  self.current_nll_event.Zproj,
                                                  "bo")
        self.crosssectiony_nllloc, = self.crosssectiony_ax.plot(self.current_nll_event.Zproj, 
                                                  self.current_nll_event.Yproj,
                                                  "bo")
        
        axlimits = self._define_axes_limits(nonlinloc=True)
        self._set_axes_limits(axlimits)
        if self.show_scatter:
            self.plot_scatter()
    
    def plot_scatter(self):
        try:
            for line in [self.mapview_scatter,self.crosssectionx_scatter,self.crosssectiony_scatter]:
                line.remove()
        except:
            pass
        self.mapview_scatter = self.mapview_ax.scatter(self.current_nll_event.scatter.Xproj.iloc[::5], 
                                                  self.current_nll_event.scatter.Yproj.iloc[::5],
                                                  c="r", s=0.5, alpha=0.2, zorder=-100)
        self.crosssectionx_scatter = self.crosssectionx_ax.scatter(self.current_nll_event.scatter.Xproj.iloc[::5], 
                                                  self.current_nll_event.scatter.Zproj.iloc[::5],
                                                  c="r", s=0.5, alpha=0.2, zorder=-100)
        self.crosssectiony_scatter = self.crosssectiony_ax.scatter(self.current_nll_event.scatter.Zproj.iloc[::5], 
                                                  self.current_nll_event.scatter.Yproj.iloc[::5],
                                                  c="r", s=0.5, alpha=0.2, zorder=-100)
    def plot_stations(self):
        station_colours = self.get_station_colours()
        station_symbols = self.get_station_symbols()
        self.mapview_stations = self.mapview_ax.scatter(self.locate_run.project.stations.Xproj,
                                                     self.locate_run.project.stations.Yproj,
                                                     marker="^", s=50, c=station_colours,
                                                     edgecolors="k")
        self.mapview_stationstext = [self.mapview_ax.text(row.Xproj, row.Yproj, station) 
                                     for station, row in self.locate_run.project.stations.iterrows()]
        
        self.crosssectionx_stations = self.crosssectionx_ax.scatter(self.locate_run.project.stations.Xproj,
                                                     -self.locate_run.project.stations.Elevation,
                                                     marker="^", s=50, c=station_colours,
                                                     edgecolors="k")
        
        self.crosssectiony_stations = self.crosssectiony_ax.scatter(-self.locate_run.project.stations.Elevation,
                                                     self.locate_run.project.stations.Yproj,
                                                     marker="<", s=50, c=station_colours,
                                                     edgecolors="k")
    def get_station_colours(self):
        colour = []
        for name in self.locate_run.project.stations.index:
            if not name in self.manual_picks.index:
                colour.append("white")
                continue
            picks = self.manual_picks.loc[name]
            if self.colour_by_phase:
                mask = picks.Quality<4
                nphases = sum(mask)
                if nphases == 2:
                    colour.append("green")
                elif nphases == 1 and picks.index[mask][0] == "P":
                    colour.append("red")
                elif nphases == 1 and picks.index[mask][0] == "S":
                    colour.append("blue")
                else:
                    colour.append("white")
        return colour

    def get_station_symbols(self):
        symbol = []
        for name in self.locate_run.project.stations.index:
            if not name in self.manual_picks.index:
                symbol.append("^")
                continue
            picks = self.manual_picks.loc[name]
            if self.colour_by_phase:
                mask = picks.Quality<4
                nphases = sum(mask)
                if nphases == 2:
                    symbol.append("^")
                elif nphases == 1 and picks.index[mask][0] == "P":
                    symbol.append("^")
                elif nphases == 1 and picks.index[mask][0] == "S":
                    symbol.append("^")
                else:
                    symbol.append("o")       
        return symbol
    def define_parameters(self):
        self.show_stations = True
        self.show_qmloc = True
        self.show_scatter = True
        
        self.phases_used = "PS"
        self.plotted = False

        self.colour_by_phase = True
        self.colour_by_residual = True
        self.manual_picks = pd.DataFrame([])

        # self.current_nll_event = self.NLL.read_summed_hypfile(
        #     os.path.join(
        #         self.NLL.root, "manual_locations", f"{self.current_event.EventID}.hyp"
        #             )
        #         )
        
        # # self.replot("STATIONS")
        # self.plot_nllloc()
        # self.residual_ax.clear()
        # self.weight_ax.clear()
        # self.plot_residuals()
        # self.canvas.draw()

        self.current_nll_event = NonLinLocEvent()

    def get_picks(self, picks):
        self.manual_picks = picks

    def _define_axes_limits(self, nonlinloc=False):
        if nonlinloc:
            scatter_percentile = self.current_nll_event.get_scatter_percentiles([30, 60])
            minx = np.floor(min([self.locate_run.project.stations.Xproj.min(),
                                self.current_event.Xproj, 
                                self.current_nll_event.Xproj,
                                scatter_percentile[0,0]]))
            maxx = np.ceil(max([self.locate_run.project.stations.Xproj.max(),
                                self.current_event.Xproj, 
                                self.current_nll_event.Xproj,
                                scatter_percentile[1,0]]))
            miny = np.floor(min([self.locate_run.project.stations.Yproj.min(),
                                self.current_event.Yproj, 
                                self.current_nll_event.Yproj,
                                scatter_percentile[0,1]]))
            maxy = np.ceil(max([self.locate_run.project.stations.Yproj.max(),
                                self.current_event.Yproj, 
                                self.current_nll_event.Yproj,
                                scatter_percentile[1,1]]))
            minz = np.floor(min([-1, self.locate_run.project.stations.Zproj.min(),
                                self.current_event.Zproj, 
                                self.current_nll_event.Zproj,
                                scatter_percentile[0,2]]))
            maxz = np.ceil(max([5, self.locate_run.project.stations.Zproj.max(),
                                self.current_event.Zproj, 
                                self.current_nll_event.Zproj,
                                scatter_percentile[1,2]]))
        else:
            minx = np.floor(min([self.locate_run.project.stations.Xproj.min(), self.current_event.Xproj.min()]))
            maxx = np.ceil(max([self.locate_run.project.stations.Xproj.max(), self.current_event.Xproj.max()]))
            miny = np.floor(min([self.locate_run.project.stations.Yproj.min(), self.current_event.Yproj.min()]))
            maxy = np.ceil(max([self.locate_run.project.stations.Yproj.max(), self.current_event.Yproj.max()]))
            minz = np.floor(min([-1, self.locate_run.project.stations.Zproj.min(), self.current_event.Zproj.min()]))
            maxz = np.ceil(max([5, self.locate_run.project.stations.Zproj.max(), self.current_event.Zproj.max()]))
        xbuffer = (maxx-minx) * 0.05
        ybuffer = (maxy-miny) * 0.05
        zbuffer = (maxz-minz) * 0.05
        minx -= xbuffer
        maxx += xbuffer
        miny -= ybuffer
        maxy += ybuffer
        minz -= zbuffer
        maxz += zbuffer
        return [minx, maxx, miny, maxy, minz, maxz]
    def _set_axes_limits(self, limits):
        minx, maxx, miny, maxy, minz, maxz = limits
        self.mapview_ax.set_xlim(minx, maxx)
        self.mapview_ax.set_ylim(miny, maxy)
        self.crosssectionx_ax.set_xlim(minx, maxx)
        self.crosssectionx_ax.set_ylim(minz, maxz)
        self.crosssectiony_ax.set_xlim(minz, maxz)
        self.crosssectiony_ax.set_ylim(miny, maxy)
        self.crosssectionx_ax.invert_yaxis()
    def init_plot(self):
        self.plot_stations()
        self.plot_qmloc()
        
        # define axes limits
        self._set_axes_limits(self._define_axes_limits())

        self.mapview_ax.set_aspect(1)
        self.crosssectionx_ax.set_aspect(0.5)
        self.crosssectiony_ax.set_aspect(2)
        self.canvas.draw()

    def init_ui(self):

        self.relocate_event_button = QPushButton("Relocate Event")
        self.show_stations_widget = QCheckBox("Show Stations")
        self.show_stations_widget.setChecked(True)
        self.show_qmloc_widget = QCheckBox("Show QM Location")
        self.show_qmloc_widget.setChecked(True)
        self.show_scatter_widget = QCheckBox("Show Scatter")
        self.show_scatter_widget.setChecked(True)  

        self.use_PSphases_widget = QRadioButton("Use P and S Phases")
        self.use_PSphases_widget.setChecked(True)
        # self.use_PSphases_widget.setID(1)
        self.use_Pphases_widget = QRadioButton("Use P Only")
        # self.use_PSphases_widget.setID(2)
        self.use_Sphases_widget = QRadioButton("Use S Only")
        # self.use_PSphases_widget.setID(3)
        self.phaseselection_widget = QButtonGroup()
        self.phaseselection_widget.addButton(self.use_PSphases_widget)
        self.phaseselection_widget.addButton(self.use_Pphases_widget)
        self.phaseselection_widget.addButton(self.use_Sphases_widget)
        self.phaseselection_widget.setId(self.use_PSphases_widget, 0)
        self.phaseselection_widget.setId(self.use_Pphases_widget, 1)    
        self.phaseselection_widget.setId(self.use_Sphases_widget, 2)
        phase_selection_box = QGroupBox("Phase Selection")
        phase_selection_box.setCheckable(False)
        phase_selection_layout = QVBoxLayout()
        # phase_selection_layout.addWidget(self.phaseselection_widget)
        phase_selection_layout.addWidget(self.use_PSphases_widget)
        phase_selection_layout.addWidget(self.use_Pphases_widget)
        phase_selection_layout.addWidget(self.use_Sphases_widget)
        phase_selection_box.setLayout(phase_selection_layout)
              
        
        gs = GridSpec(3, 4)
        self.fig = plt.figure()
        self.mapview_ax = self.fig.add_subplot(gs[:2,:2])
        self.crosssectionx_ax = self.fig.add_subplot(gs[2,:2])
        self.crosssectiony_ax = self.fig.add_subplot(gs[:2,2])
        self.residual_ax = self.fig.add_subplot(gs[:2,3])
        self.weight_ax = self.residual_ax.twiny()
        self.hypview_ax = None # self.fig.add_subplot(gs[2,2:])
        self.axes = [self.mapview_ax, self.crosssectionx_ax,
                     self.crosssectiony_ax, self.residual_ax, self.hypview_ax]
        # self.fig, self.axes = plt.subplots(2, 2, height_ratios=(3,1))
        # self.mapview_ax = self.axes[0,0]
        # self.crosssection_ax = self.axes[1,0]
        # self.timeseries_ax = self.axes[1,1]
        # self.waveforms_ax = self.axes[0,1]
        self.fig.tight_layout(pad=0.2, w_pad=1)
        self.canvas = FigureCanvas(self.fig)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.addWidget(phase_selection_box)
        left_layout.addWidget(self.relocate_event_button)
        left_layout.addWidget(self.show_qmloc_widget)
        left_layout.addWidget(self.show_stations_widget)
        left_layout.addWidget(self.show_scatter_widget)
        left_layout.addStretch(1)

        right_layout.addWidget(self.canvas)

        overall_layout = QHBoxLayout()
        overall_layout.addLayout(left_layout, stretch=1)
        overall_layout.addLayout(right_layout, stretch=10)
        self.setLayout(overall_layout)
    def setup_connections(self):
        self.relocate_event_button.clicked.connect(self.relocate_event)
        self.show_stations_widget.stateChanged.connect(self._set_show_stations)
        self.show_qmloc_widget.stateChanged.connect(self._set_show_qmloc)
        self.show_scatter_widget.stateChanged.connect(self._set_show_scatter)
        # self.use_PSphases_widget.toggled.connect(self._set_use_phase)
        # self.use_Pphases_widget.toggled.connect(self._set_use_phase)
        # self.use_Sphases_widget.toggled.connect(self._set_use_phase)
        self.phaseselection_widget.idClicked.connect(self._set_use_phase)
    def relocate_event(self):
        print("RELOCATE EVENT")
        print(self.manual_picks)
        print(self.current_event)
        tempdir = self.NLL.relocate_event(self.current_event.EventID,self.manual_picks)
        print("...DONE")

        # some checking here for bad NLLOC runs
        print("COPYING")
        fs = os.listdir(tempdir)
        for f in fs:
            if self.current_event.EventID in f:
                shutil.copy(os.path.join(tempdir, f), 
                            os.path.join(self.NLL.root, "manual_locations"))
        
        self.current_nll_event = self.NLL.read_summed_hypfile(
            os.path.join(
                self.NLL.root, "manual_locations", f"{self.current_event.EventID}.hyp"
                    )
                )
        
        # self.replot("STATIONS")
        self.plot_nllloc()
        self.residual_ax.clear()
        self.weight_ax.clear()
        self.plot_residuals()
        self.canvas.draw()


    def _set_show_stations(self):
        print("SHOW STATIONS")
        self.show_stations = self.show_stations_widget.isChecked()
        # if self.plotted:
        self.replot("STATIONS")
    def _set_show_qmloc(self):
        print("SHOW QMLOC", self.show_qmloc_widget.isChecked())
        self.show_qmloc = self.show_qmloc_widget.isChecked()
        # if self.plotted:
        self.replot("QMLOC")
    def _set_show_scatter(self):
        print("SHOW SCATTER")
        self.show_scatter = self.show_scatter_widget.isChecked()
        if self.plotted:
            self.replot("SCATTER")
    def _set_use_phase(self, _id):
        print("PHASE SELECT")
        if _id == 0:
            self.phases_used = "PS"
        elif _id == 1:
            self.phases_used = "P"
        elif _id == 2:
            self.phases_used = "S"
        else:
            raise ValueError("HOW HAS THIS HAPPENED")

class FPS(QWidget):
    def __init__(self, locate_run):
        super().__init__()
        self.locate_run = locate_run
        
        self.define_parameters()
        self.init_ui()
        self.init_plot()
        self.setup_connections()

    def plot_stations(self):
        print("plotting stations")
        mask = (self.manual_picks.Polarity > 0) & ~self.manual_picks.upper_hemisphere & (self.manual_picks.weight > 0)
        self.ax.line(self.manual_picks.plunge[mask], self.manual_picks.bearing[mask], "r^")
        mask = (self.manual_picks.Polarity > 0) & self.manual_picks.upper_hemisphere & (self.manual_picks.weight > 0)
        self.ax.line(self.manual_picks.plunge[mask], self.manual_picks.bearing[mask], "^", mec="r", mfc="white")

        mask = (self.manual_picks.Polarity < 0) & ~self.manual_picks.upper_hemisphere & (self.manual_picks.weight > 0)
        self.ax.line(self.manual_picks.plunge[mask], self.manual_picks.bearing[mask], "bv")
        mask = (self.manual_picks.Polarity < 0) & self.manual_picks.upper_hemisphere & (self.manual_picks.weight > 0)
        self.ax.line(self.manual_picks.plunge[mask], self.manual_picks.bearing[mask], "v", mec="b", mfc="white")

        mask = (self.manual_picks.Polarity == 0) | (self.manual_picks.weight == 0)
        self.ax.line(self.manual_picks.plunge[mask], self.manual_picks.bearing[mask], "o", mfc="white", mec="black")

        for index, row in self.manual_picks.iterrows():
            lon, lat = mplstereonet.stereonet_math.line(row.plunge, row.bearing)
            self.ax.text(lon, lat, index[0])

    def plot_solution(self):
        if self.plot_solution_range and len(self.solution_range) > 1:
            if len(self.solution_range) < 10:
                self.ax = plot_all_fault_planes(self.ax, self.solution_range, color="black", alpha=0.4)
            elif len(self.solution_range) < 500:
                self.ax = plot_all_fault_planes(self.ax, self.solution_range, color="black")
            else:
                self.ax = plot_fault_plane_histogram(self.ax, self.solution_range)
        if self.plot_PT_error:
            self.ax = plot_PT_errorbar(self.solution_range, self.ax)
        if self.plot_PT:
            self.ax = plot_PT_axes(self.solution, self.ax)

        self.ax.plane(self.solution.plane1.strike, self.solution.plane1.dip, "r-", lw=2, label="fine")
        self.ax.plane(self.solution.plane2.strike, self.solution.plane2.dip, "r-", lw=2)

    def add_plunge_bearing(self):
        # calculate plunge/bearing for plotting 
        self.manual_picks["plunge"] = 90 - self.manual_picks.receiver_takeoffangle
        self.manual_picks["bearing"] = self.manual_picks.station_azimuth
        self.manual_picks["upper_hemisphere"] = False
        # where the plunge is negative, correct
        mask = self.manual_picks.plunge<0
        self.manual_picks.loc[mask,"plunge"] = self.manual_picks.loc[mask,"plunge"].apply(np.abs)
        self.manual_picks.loc[mask,"bearing"] = (self.manual_picks.loc[mask,"bearing"] + 180)%360
        self.manual_picks.loc[mask,"upper_hemisphere"] = True

    def add_weight(self):
        # this section sets the weight based on the picks
        qual2weight = {0:0, 1:0.2, 2:0.4, 3:0.5, 4:1}
        self.manual_picks["weight"] = self.manual_picks.loc[:,"Quality"].apply(lambda key: qual2weight.get(key, None))
        mask1 = self.manual_picks["weight"] < 0.001
        mask2 = self.manual_picks["weight"] >= 0.5
        mask3 = ~mask1 & ~mask2
        self.manual_picks.loc[mask1, "weight"] = 29.6386
        self.manual_picks.loc[mask2, "weight"] = 0.
        self.manual_picks.loc[mask3, "weight"] = 1 / (self.manual_picks.loc[mask3,"weight"] - self.manual_picks.loc[mask3,"weight"]**2) - 2
        print("AVERAGE WEIGHT", (1./len(self.manual_picks)) * np.sum(self.manual_picks.weight), "30 = all best picks, 0 = rubbish") 

    def convert_polarity_to_number(self):
        dic = {"U":1, "D":-1, np.nan:0, None:0, "u":1, "d":-1, "+":1, "-":-1}
        self.manual_picks.loc[:,"Polarity"] = self.manual_picks.loc[:,"Polarity"].apply(lambda key: dic.get(key, 0))

    def get_picks(self, picks):
        # only get Ppicks
        mask = picks.index.get_level_values(1) == "P"
        picks = picks.loc[mask,:]
        self.manual_picks = picks
        self.add_plunge_bearing()
        self.add_weight()
        self.convert_polarity_to_number()

        if len(self.manual_picks) > 0:
            self.plot_stations()
            self.canvas.draw()

    def get_radiation_pattern(self, solution):
        # get the predicted radiation amplitude
        return prad(solution.plane1.strike, solution.plane1.dip, solution.plane1.rake, 
                        self.manual_picks.receiver_takeoffangle, self.manual_picks.station_azimuth)
    
    def check_polarities(self, solution):
        # set the predicted polarities
        pred = np.sign(self.get_radiation_pattern(solution))
        isTrue = self.manual_picks.Polarity == pred[:,0]

        if np.sum(~isTrue) == 0:
            print("ALL ARE TRUE")
        else:
            print("FALSE STATIONS")
            print(self.manual_picks.Polarity[~isTrue])
            # quality parameters
    
    def return_quality_from_misfit(self, value):
        if value <= 0.025:
            return "A"
        elif value <= 0.1:
            return "B"
        else:
            return "C"
        
    def run_grid_search(self):
        sol, sol_range, sol_misfit = fp_grid_search(self.manual_picks, 
                                                    dstrike=self.dstrike, 
                                                    ddip=self.ddip, 
                                                    drake=self.drake,
                                                    output_range=True)
        return sol, sol_range, sol_misfit
    
    def define_parameters(self):
        self.manual_picks = pd.DataFrame([])

        self.dstrike = 2
        self.ddip = 2
        self.drake = 10

        self.plot_solution_range = False
        self.plot_PT_error = False
        self.plot_PT = True

    def init_ui(self):

        self.run_gridsearch_widget = QPushButton("Run Grid Search")
        self.plot_solution_range_widget = QCheckBox("Plot Solution Range")
        self.plot_solution_range_widget.setChecked(False)
        self.plot_PT_error_widget = QCheckBox("Plot PT Error")
        self.plot_PT_error_widget.setChecked(False)
        self.plot_PT_widget = QCheckBox("Plot PT axes")
        self.plot_PT_widget.setChecked(True)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='stereonet')
        self.fig.tight_layout(pad=0.2, w_pad=1)
        self.canvas = FigureCanvas(self.fig)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.run_gridsearch_widget)
        left_layout.addWidget(self.plot_PT_error_widget)
        left_layout.addWidget(self.plot_solution_range_widget)
        left_layout.addWidget(self.plot_PT_widget)
        left_layout.addStretch(1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)

        overall_layout = QHBoxLayout()
        overall_layout.addLayout(left_layout)
        overall_layout.addLayout(right_layout, stretch=10)
        
        # layout.addWidget(QLabel("HELP"))
        self.setLayout(overall_layout)

    def init_plot(self):
        if len(self.manual_picks) > 0:
            self.plot_stations()
        self.canvas.draw()
    def setup_connections(self):
        self.run_gridsearch_widget.clicked.connect(self._run_fps)
        self.plot_PT_error_widget.clicked.connect(self._set_plot_PT_error)
        self.plot_solution_range_widget.clicked.connect(self._set_plot_solution_range)
        self.plot_PT_widget.clicked.connect(self._set_plot_PT)
    def _run_fps(self):
        self.solution, self.solution_range, self.misfit = self.run_grid_search()
        self.ax.clear()
        self.plot_stations()
        self.plot_solution()
        self.canvas.draw()
    def _set_plot_PT(self):
        self.plot_PT = self.plot_PT_widget.isChecked()
        self.ax.clear()
        self.plot_stations()
        self.plot_solution()
        self.canvas.draw()
    def _set_plot_PT_error(self):
        self.plot_PT_error = self.plot_PT_error_widget.isChecked()
        self.ax.clear()
        self.plot_stations()
        self.plot_solution()
        self.canvas.draw()
    def _set_plot_solution_range(self):
        self.plot_solution_range = self.plot_solution_range_widget.isChecked()
        self.ax.clear()
        self.plot_stations()
        self.plot_solution()
        self.canvas.draw()

class Wadati(QWidget):
    def __init__(self, locate_run, uid):
        super().__init__()
        self.locate_run = locate_run
        self.uid = uid
        
        self.define_parameters()
        self.init_ui()
        self.init_plot()
        self.setup_connections()

    def get_picks(self, picks):
        self.manual_picks = picks

    def define_parameters(self):

        nll = self.locate_run.project.nll_settings
        
        try:
            self.manual_picks = self.locate_run.get_manual_picks(self.uid, nll.root)
        except FileNotFoundError:
            self.manual_picks = pd.DataFrame([])

        # read nonlinloc hypfile
        nll_hypfile = os.path.join(nll.root, "manual_locations", f"{self.uid}.hyp")
        if os.path.exists(nll_hypfile):
            self.read_nonlinloc_event(nll_hypfile)
        else:
            self.current_nll_event = NonLinLocEvent()

    def init_ui(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=0.2, w_pad=1)
        self.canvas = FigureCanvas(self.fig)

        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        # layout.addWidget(QLabel("HELP"))
        self.setLayout(layout)

    def init_plot(self):
        
        if len(self.current_nll_event.phases) > 0:
            self.ax.clear()
            self.plot_wadati()

        self.annot = self.ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"), ha="right",
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        self.ax.set_xlabel("P-wave travel time [s]")
        self.ax.set_ylabel("S-P time [s]")
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)
        self.canvas.draw()

    def read_nonlinloc_event(self, fname):
        nll = self.locate_run.project.nll_settings
        self.current_nll_event = nll.read_summed_hypfile(fname)
        
    def plot_wadati(self):
        # residual = observed - predicted
        phases = pd.merge(self.manual_picks, self.current_nll_event.phases, left_index=True, right_index=True, how="outer")
        x, y, xerr, yerr = self.get_traveltimes(phases)

        self.markers, self.caplines, self.barlinecols = self.ax.errorbar(x, y, xerr=xerr, yerr=yerr, ls="none", marker="o", mec="k", mfc="red")
        self.highlighter, = self.ax.plot(x[0], y[0], "yo", ms=10)
        self.highlighter.set_visible(False)

    def get_traveltimes(self, phases):
        traveltime = phases.TTpred + phases.Residual
        error = phases.Quality.apply(lambda key: self.locate_run.project.nll_settings.nll_locqual2err.get(key, 9999))

        self.stations = list(set(traveltime.index.get_level_values(0)))

        ptimes, sptimes, perrs, sperrs = [], [], [], []
        for station in self.stations:
            picks = phases.loc[station,:]
            if not len(picks) == 2:
                continue
            perr, sperr = self.get_errors(error.loc[station])
            
            ptimes.append(traveltime.loc[(station, "P")])
            sptimes.append(traveltime.loc[(station, "S")] - traveltime.loc[(station, "P")])

            perrs.append(perr)
            sperrs.append(sperr)
        
        return ptimes, sptimes, perr, sperr

    def get_errors(self, err):
        p_error = err.loc["P"]
        s_error = err.loc["S"]
        if p_error > 10:
            p_error = None
        if s_error > 10:
            s_error = None
            
        if p_error and s_error:
            sp_error = p_error+s_error
        else:
            sp_error = None
        return p_error, sp_error
    def setup_connections(self):

        self.canvas.mpl_connect("motion_notify_event", self._hover)
    
    def _hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.markers.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.highlighter.set_visible(True)
                self.canvas.draw()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.highlighter.set_visible(False)
                    self.canvas.draw()

    def update_annot(self, ind):
        x,y = self.markers.get_data()
        self.highlighter.set_xdata([x[ind["ind"][0]]])
        self.highlighter.set_ydata([y[ind["ind"][0]]])

        self.annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        text = "{}".format(" ".join([self.stations[n] for n in ind["ind"]]))
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.4)    

class TravelTime(QWidget):

    def __init__(self, locate_run, event):
        super().__init__()
        self.locate_run = locate_run
        self.current_event = event
        self.uid = self.current_event.EventID
        
        nll = self.locate_run.project.nll_settings
        stations = self.locate_run.project.stations
        self.lut = nll.read_travel_time_tables(stations.index[0], "P")

        self.define_parameters()
        self.init_ui()
        self.init_plot()
        self.setup_connections()

    def plot_qm_traveltimes(self):
        dist_max = np.ceil(self.qm_picks.epi_distance.max())
        dist_max += dist_max*0.1

        distance, traveltimes = self.get_travel_times(dist_max, self.current_event.Zproj)

        self.qmtt_lines = []
        self.qmtt_lines.append(self.ax.plot(distance, traveltimes, "-", c="darkred", label="QM P-wave", alpha=0.5)[0])
        self.qmtt_lines.append(self.ax.plot(distance, traveltimes*self.vpvs, "-", c="darkblue", label="QM S-wave", alpha=0.5)[0])

        self._set_vis_qmtt()

    def get_name_array(self, picks, label):
        values = picks.index.get_level_values(0).to_list()
        return [f"{val} {label}" for val in values]

    def plot_qm_picks(self):
        print("PLOTTING QM PICKS")

        picks = self.qm_picks[self.qm_picks.SNR>0]
        traveltimes = pd.to_datetime(picks.PickTime).dt.tz_convert(None) - self.current_event.DT
        # pritn(traveltimes)

        
        mask = picks.index.get_level_values(1) == "P"
        marker, caps, lines = self.ax.errorbar(picks.loc[mask, "epi_distance"], traveltimes[mask].dt.total_seconds(), 
                                                yerr=picks.loc[mask, "PickError"], 
                                                ls="none", color="darkred", alpha=0.5, marker="o", capsize=2)
        self.qmpk_lines = [marker, caps, lines]
        self.markers[0] = marker
        self.names[0] = self.get_name_array(picks.loc[mask,:], "QM P")

        mask = picks.index.get_level_values(1) == "S"
        marker, caps, lines = self.ax.errorbar(picks.loc[mask, "epi_distance"], traveltimes[mask].dt.total_seconds(), 
                                            yerr=picks.loc[mask, "PickError"], 
                                            ls="none", color="darkblue", alpha=0.5, marker="o", capsize=2)
        self.qmpk_lines += [marker, caps, lines]
        self.markers[1] = marker
        self.names[1] = self.get_name_array(picks.loc[mask,:], "QM S")

        print(self.markers)

        self._set_vis_qmpk()

    def plot_manual_traveltimes(self):
        dist_max = np.ceil(self.current_nll_event.phases.Distance.max())
        dist_max += dist_max*0.1

        distance, traveltimes = self.get_travel_times(dist_max, self.current_nll_event.Zproj)

        if len(self.mantt_lines) > 0:
            for line in self.mantt_lines:
                line.remove()
            self.canvas.draw()

        self.mantt_lines = []
        self.mantt_lines.append(self.ax.plot(distance, traveltimes, "r-", label="Manual P-wave")[0])
        self.mantt_lines.append(self.ax.plot(distance, traveltimes*self.vpvs, "b-", label="Manual S-wave")[0])

        self._set_vis_mantt()

    def plot_manual_picks(self):

        picks = pd.merge(self.manual_picks, self.current_nll_event.phases, left_index=True, right_index=True, how="inner")
        picks.loc[:, "Error"] = picks.loc[:, "Quality"].apply(lambda key: self.locate_run.project.nll_settings.nll_locqual2err.get(key, None))

        mask = picks.Error > 10
        picks.loc[mask, "Error"] = None

        # picks = picks.loc[picks.Quality<4, :]
        traveltimes = pd.to_datetime(picks.PickTime) - self.current_nll_event.otime
        # pritn(traveltimes)

        if len(self.manpk_lines) > 0:
            for line in self.manpk_lines:
                if isinstance(line, matplotlib.lines.Line2D):
                    line.remove()
                elif len(line) == 0:
                    continue
                else:
                    [ln.remove() for ln in line]
            self.canvas.draw()

        mask = picks.index.get_level_values(1) == "P"
        marker, caps, lines = self.ax.errorbar(picks.loc[mask, "Distance"], traveltimes[mask].dt.total_seconds(), 
                                                yerr=picks.loc[mask, "Error"],
                                                ls="none", color="r", marker="s", capsize=2)
        self.manpk_lines = [marker, caps, lines]
        self.markers[2] = marker
        self.names[2] = self.get_name_array(picks.loc[mask,:], "P")

        mask = picks.index.get_level_values(1) == "S"
        marker, caps, lines = self.ax.errorbar(picks.loc[mask, "Distance"], traveltimes[mask].dt.total_seconds(), 
                                                yerr=picks.loc[mask, "Error"],
                                                ls="none", color="b", marker="s", capsize=2)
        self.manpk_lines += [marker, caps, lines]
        self.markers[3] = marker
        self.names[3] = self.get_name_array(picks.loc[mask,:], "S")

        self._set_vis_manpk()

    def get_manual_picks(self):
        nll = self.locate_run.project.nll_settings
        
        try:
            self.manual_picks = self.locate_run.get_manual_picks(self.uid, nll.root)
        except FileNotFoundError:
            self.manual_picks = pd.DataFrame([])

        # read nonlinloc hypfile
        nll_hypfile = os.path.join(nll.root, "manual_locations", f"{self.current_event.EventID}.hyp")
        if os.path.exists(nll_hypfile):
            self.read_nonlinloc_event(nll_hypfile)
        else:
            self.current_nll_event = NonLinLocEvent()

    def get_travel_times(self, dmax, z):
        traveltimes = self.lut[0]
        _nx, ny, nz = self.lut[1]
        _x0, _y0, z0 = self.lut[2]
        dx = self.lut[3]

        dvec = np.arange(ny) * dx
        # zvec = z0 + np.arange(nz) * dx

        i = int(np.ceil(dmax/dx))
        j = int(np.round((z-z0)/dx))

        return dvec[:i], traveltimes[0,:i, j]


    def add_epi_distance_to_picks(self, picks):
        stations = picks.index.get_level_values(0)
        station_info = self.locate_run.project.stations
        picks.loc[:,"epi_distance"] = np.array(np.sqrt((self.current_event.Xproj-station_info.loc[stations,"Xproj"])**2 + 
                                    (self.current_event.Yproj-station_info.loc[stations,"Yproj"])**2))
        return picks


    def define_parameters(self):

        self.qm_picks = self.locate_run.get_picks(self.uid)
        self.qm_picks = self.add_epi_distance_to_picks(self.qm_picks)

        self.vpvs = 1.76

        self.get_manual_picks()

        self.mantt_lines = []
        self.manpk_lines = []
        self.markers = [[],[],[],[]]
        self.names = [[],[],[],[]]

    def init_ui(self):        
        
        self.qm_traveltime_widget = QCheckBox("QM Travel Time", self)
        self.qm_traveltime_widget.setChecked(True)
        self.qm_picks_widget = QCheckBox("QM Picks", self)
        self.qm_picks_widget.setChecked(True)
        self.manual_traveltime_widget = QCheckBox("Manual Travel Time", self)
        self.manual_traveltime_widget.setChecked(True)
        self.manual_picks_widget = QCheckBox("Manual Picks", self)
        self.manual_picks_widget.setChecked(True)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=0.2, w_pad=1)
        self.canvas = FigureCanvas(self.fig)

        left_layout = QHBoxLayout()
        left_layout.addWidget(self.qm_traveltime_widget)
        left_layout.addWidget(self.qm_picks_widget)
        left_layout.addWidget(self.manual_traveltime_widget)
        left_layout.addWidget(self.manual_picks_widget)
        # left_layout.addStretch(1)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.canvas)

        overall_layout = QVBoxLayout()
        overall_layout.addLayout(right_layout)
        overall_layout.addLayout(left_layout)
        self.setLayout(overall_layout)

    def init_plot(self):
        # if len(self.current_nll_event.phases) > 0:
        #     self.ax.clear()
        #     self.plot_wadati()

        self.ax.clear()

        self.plot_qm_traveltimes()
        self.plot_qm_picks()

        if len(self.manual_picks) > 0:
            self.plot_manual_traveltimes()
            self.plot_manual_picks()
        
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"), ha="right",
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.highlighter, = self.ax.plot(0, 0, "yo", ms=10)
        self.highlighter.set_visible(False)

        # set visibility from checkboxes
        if len(self.manual_picks) > 0:
            self._set_vis_manpk()
            self._set_vis_mantt()
        self._set_vis_qmpk()
        self._set_vis_qmtt()

        self.ax.set_ylabel("travel time [s]")
        self.ax.set_xlabel("Distance [km]")
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)
        self.canvas.draw()

    def read_nonlinloc_event(self, fname):
        nll = self.locate_run.project.nll_settings
        self.current_nll_event = nll.read_summed_hypfile(fname)
        
    def setup_connections(self):

        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self.qm_traveltime_widget.stateChanged.connect(self._set_vis_qmtt)
        self.qm_picks_widget.stateChanged.connect(self._set_vis_qmpk)
        self.manual_traveltime_widget.stateChanged.connect(self._set_vis_mantt)
        self.manual_picks_widget.stateChanged.connect(self._set_vis_manpk)
    
    def _set_vis_qmtt(self):
        for line in self.qmtt_lines:
            line.set_visible(self.qm_traveltime_widget.isChecked())
        self.canvas.draw()
    def _set_vis_qmpk(self):
        for line in self.qmpk_lines:
            if isinstance(line, matplotlib.lines.Line2D):
                line.set_visible(self.qm_picks_widget.isChecked())
            elif len(line) == 0:
                continue
            else:
                [ln.set_visible(self.qm_picks_widget.isChecked()) for ln in line]
        self.canvas.draw()
    def _set_vis_mantt(self):
        for line in self.mantt_lines:
            line.set_visible(self.manual_traveltime_widget.isChecked())
        self.canvas.draw()
    def _set_vis_manpk(self):
        for line in self.manpk_lines:
            if isinstance(line, matplotlib.lines.Line2D):
                line.set_visible(self.manual_picks_widget.isChecked())
            elif len(line) == 0:
                continue
            else:
                [ln.set_visible(self.manual_picks_widget.isChecked()) for ln in line]
        self.canvas.draw()
    def _hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            for ii, marker in enumerate(self.markers):
                if not marker.get_visible():
                    continue
                cont, ind = marker.contains(event)
                if cont:
                    break
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.highlighter.set_visible(False)
                    self.canvas.draw()
                return
            
            self.update_annot(marker, ind, ii)
            self.annot.set_visible(True)
            self.highlighter.set_visible(True)
            self.canvas.draw()


    def update_annot(self, marker, ind, i):
        x,y = marker.get_data()
        self.highlighter.set_xdata([x[ind["ind"][0]]])
        self.highlighter.set_ydata([y[ind["ind"][0]]])

        self.annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        text = "{}".format(" ".join([self.names[i][n] for n in ind["ind"]]))
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.4)  

if __name__ == '__main__':
    # app = QApplication([])
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    prj = Project("/space/tg286/iceland/reykjanes/qmigrate/may21/nov23_dyke_tight")
    loc = LocateRun(prj)
    window = PickWindow(loc, loc.events.iloc[100])
    window.show()
    app.exec()