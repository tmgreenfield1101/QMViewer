from util import *
from quakemigrate.io import read_lut
from obspy import read, Stream, UTCDateTime, read_inventory
import pandas as pd
import numpy as np
import os
import ast

from quakemigrate.io.data import WaveformData
from quakemigrate.signal.local_mag import Magnitude

manual_pick_columns = ["Network", "Station", "Location", "Channel", 
                       "Quality", "Polarity", "Type", "Phase",
                       "PickTime"]
manual_pick_index = pd.MultiIndex.from_arrays([[],[]], names=["Station", "Phase"])

amplitude_pick_columns = ["Network", "Station", "Location", "Channel", 
                            "Component", "StreamID",
                            "RawAmplitude", "RawPeriod", "RawTime",
                            "RealAmplitude", "RealPeriod", "RealTime",
                            "WAAmplitude", "WAPeriod", "WATime", "QMFlag", "Filter"]
amplitude_pick_index = pd.MultiIndex.from_arrays([[],[]], names=["Station", "Component"])

class Project():
    """QM project conatiner"""
    def __init__(self, fpath):
        self.filepath = fpath

        self.project_name = None
        self.starttime = None
        self.endtime = None
        
        self.lut_path = None
        self.LUT = None

        self.station_path = None
        self.stations = None

        self.read_project_file()
        self.read_nonlinloc_settings()

    def __str__(self):
        print("THIS WOULD BE HJELPFUL")
        pass

    def read_nonlinloc_settings(self):
        nll_settings_file = os.path.join(self.filepath, "nonlinloc.yaml")

        self.nll_settings = NonLinLoc(self.filepath)
        self.nll_settings.read_yaml(nll_settings_file)
        self.nll_settings.vmodel = VModel(self.LUT.velocity_model)
        self.nll_settings.stations = self.stations
        os.makedirs(self.nll_settings.root, exist_ok=True)


    def read_project_file(self):

        project_file = os.path.join(self.filepath, "project.yaml")
        if os.path.exists(project_file):
            project_settings = read_settings(project_file)

            self.project_name = project_settings["project_name"]
            self.starttime = pd.to_datetime(project_settings["starttime"])
            self.endtime = pd.to_datetime(project_settings["endtime"])
            self.lut_path = os.path.join(self.filepath, project_settings["lut_path"])
            self.station_path = os.path.join(self.filepath, project_settings["station_path"])
            
        else:
            print()
            self.project_name = input("Project Name? ")
            self.starttime = read_dt_string("Start time? ")
            self.endtime = read_dt_string("End time? ")
            self.lut_path = read_filepath("Path to LUT? ", prepath=self.filepath)
            self.station_path = read_filepath("Path to station file? ", prepath=self.filepath)
            write_settings({"project_name" : self.project_name,
                            "starttime" : self.starttime.isoformat(),
                            "endtime" : self.endtime.isoformat(),
                            "lut_path" : self.lut_path.split("/")[-1],
                            "station_path" : self.station_path.split("/")[-1]},
                            project_file)

        self.LUT = read_lut(self.lut_path)
        self.stations = pd.read_csv(self.station_path)
        self.stations.set_index("Name", inplace=True)
        self.stations = self.project_stations(self.stations)

    def project_stations(self, stations):
        stations["Xproj"] = self.LUT.coord2grid(stations.loc[:,["Longitude","Latitude","Elevation"]])[:,0]
        stations["Yproj"] = self.LUT.coord2grid(stations.loc[:,["Longitude","Latitude","Elevation"]])[:,1]
        stations["Zproj"] = self.LUT.coord2grid(stations.loc[:,["Longitude","Latitude","Elevation"]])[:,2]
        return stations

class DetectRun():
    """Container for parameters to do the detect run"""
    def __init__(self, project):
        self.project = project
        self.detect_filepath = os.path.join(self.project.filepath, "detect")

        self.read_settings()

    def __str__(self):
        raise NotImplementedError()

    def read_settings(self):
        detect_file = os.path.join(self.detect_filepath, "detect.yaml")
        if os.path.exists(detect_file):
            detect_settings = read_settings(detect_file)

            self.decimate = detect_settings["decimate"]
            self.onset_function = detect_settings["onset_function"]
            self.phases = detect_settings["phases"]
            self.bandpass_filters = detect_settings["bandpass_filters"]
            self.sta_lta_windows = detect_settings["sta_lta_windows"]
            self.timestep = detect_settings["timestep"]
        else:
            print()
            raise NotImplementedError("manual reading of detect parameters not implemented yet")

    def get_scanmseed(self, dt, length, unit="S"):
        """length"""

        pth = os.path.join(self.detect_filepath, "scanmseed")

        dt = pd.to_datetime(dt)
        length = pd.to_timedelta(length, unit=unit)
        half_length = length / 2

        tstart = dt - half_length
        tend = dt + half_length

        years = [(dt-half_length).year, dt.year, (dt+half_length).year]
        juldays = [(dt-half_length).dayofyear, dt.dayofyear, (dt+half_length).dayofyear]

        st = Stream()
        done_files = []
        for year, julday in zip(years, juldays):
            fname = os.path.join(pth, f"{year}_{julday}.scanmseed")
            if fname in done_files:
                continue
            if not os.path.exists(fname):
                print(fname, "does not exists...skipping")
                continue
            st += read(fname)
            done_files.append(fname)
        
        st.merge()
        st.trim(UTCDateTime(tstart), UTCDateTime(tend))
        return st





class TriggerRun():
    """Container for the Trigger run"""
    def __init__(self, project):
        self.project = project
        self.trigger_filepath = os.path.join(self.project.filepath, "trigger")

        self.read_settings()
        self.triggered_events = self.project_events(self.read_events())

    def __str__(self):
        raise NotImplementedError()

    def read_settings(self):
        fname = os.path.join(self.trigger_filepath, "trigger.yaml")
        if os.path.exists(fname):
            settings = read_settings(fname)

            self.min_event_interval = settings["min_event_interval"]
            self.marginal_window = settings["marginal_window"]
            self.normalise_coalescence = settings["normalise_coalescence"]
            self.threshold_method = settings["threshold_method"]
            if self.threshold_method == "static":
                self.static_threshold = settings["static_threshold"]
            elif self.threshold_method == "dynamic":
                self.mad_window_length = settings["mad_window_length"]
                self.mad_multiplier = settings["mad_multiplier"]
            else:
                raise ValueError(self.threshold_method + " is not a valid option")
                
        else:
            print()
            raise NotImplementedError("manual reading of detect parameters not implemented yet")

    def read_events(self):
        folder = os.path.join(self.trigger_filepath, "events")
        fs = os.listdir(folder)
        dataframes = []
        for f in fs:
            _df = pd.read_csv(os.path.join(folder, f),
                              parse_dates=[1])
            if len(_df) > 0:
                dataframes.append(_df.copy())

        return pd.concat(dataframes)
    
    def project_events(self, events):
        events["Xproj"] = self.project.LUT.coord2grid(events.loc[:,["COA_X","COA_Y","COA_Z"]])[:,0]
        events["Yproj"] = self.project.LUT.coord2grid(events.loc[:,["COA_X","COA_Y","COA_Z"]])[:,1]
        events["Zproj"] = self.project.LUT.coord2grid(events.loc[:,["COA_X","COA_Y","COA_Z"]])[:,2]
        return events
    
    def get_events(self, datemin=None, datemax=None, 
                   minlongitude=None, maxlongitude=None, 
                   minlatitude=None, maxlatitude=None):
        
        datemask = (self.triggered_events.CoaTime >= datemin) & (self.triggered_events.CoaTime <= datemax)
        return self.triggered_events.loc[datemask, :]

    def get_threshold(self, scandata, sampling_rate):
        """
        Determine the threshold to use when triggering candidate events.

        Parameters
        ----------
        scandata : `pandas.Series` object
            (Normalised) coalescence values for which to calculate the threshold.
        sampling_rate : int
            Number of samples per second of the coalescence scan data.

        Returns
        -------
        threshold : `numpy.ndarray` object
            Array of threshold values.

        """


        if self.threshold_method == "dynamic":
            # Split the data in window_length chunks
            breaks = np.arange(len(scandata))
            breaks = breaks[breaks % int(self.mad_window_length * sampling_rate) == 0][
                1:
            ]
            chunks = np.split(scandata, breaks)

            # Calculate the mad and median values
            mad_values = np.asarray([calculate_mad(chunk) for chunk in chunks])
            median_values = np.asarray([np.median(chunk) for chunk in chunks])
            mad_trace = chunks2trace(mad_values, (len(chunks), len(chunks[0])))
            median_trace = chunks2trace(median_values, (len(chunks), len(chunks[0])))
            mad_trace = mad_trace[: len(scandata)]
            median_trace = median_trace[: len(scandata)]

            # Set the dynamic threshold
            threshold = median_trace + (mad_trace * self.mad_multiplier)
        else:
            # Set static threshold
            threshold = np.zeros_like(scandata) + self.static_threshold

        return threshold       


class LocateRun():
    """Container for the Locate run"""
    def __init__(self, project):
        self.project = project
        self.locate_filepath = os.path.join(self.project.filepath, "locate")
        self.picks_filepath = os.path.join(self.locate_filepath, "picks")
        self.events_filepath = os.path.join(self.locate_filepath, "events")
        self.amplitudes_filepath = os.path.join(self.locate_filepath, "amplitudes")
        self.waveforms_filepath = os.path.join(self.locate_filepath, "raw_cut_waveforms")

        self.read_settings()
        self.read_events()

    def __str__(self):
        raise NotImplementedError()

    def read_settings(self):
        fname = os.path.join(self.locate_filepath, "locate.yaml")
        if os.path.exists(fname):
            settings = read_settings(fname)

            self.response_file = settings["response_file"]
            self.response_params = settings["response_params"]
            self.amp_params = settings["amp_params"]
            self.mag_params = settings["mag_params"]
            self.onset = settings["onset"]
            self.picker = settings["picker"]
            self.marginal_window = settings["marginal_window"]   
            if isinstance(self.mag_params["A0"], dict):
                n = self.mag_params["A0"]["n"]
                k = self.mag_params["A0"]["k"]
                ref_distance = self.mag_params["A0"]["ref_distance"]
                self.mag_params["A0"] = lambda dist: n*np.log10(dist/ref_distance) + k*(dist-ref_distance) + 2.
        else:
            print()
            raise NotImplementedError("manual reading of detect parameters not implemented yet")

    def read_events(self):
        fname = os.path.join(self.locate_filepath, "events.csv")
        if os.path.exists(fname):
            self.events = pd.read_csv(fname, parse_dates=[1], date_format="ISO8601", dtype={"EventID":str})
        else:
            self.events = pd.concat([pd.read_csv(os.path.join(self.events_filepath,f), parse_dates=[1], dtype={"EventID":str})
                                        for f in os.listdir(self.events_filepath)])
            self.events.to_csv(fname, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

        self.events = self.project_events(self.events)

        # sort by date
        self.events.sort_values("DT", inplace=True)

    def get_picks(self, uid):
        return pd.read_csv(os.path.join(self.picks_filepath, f"{uid}.picks"),
                           parse_dates=[2,3], index_col=[0,1], dtype={"Quality":int})
    
    def get_amplitudes(self, uid):
        return pd.read_csv(os.path.join(self.amplitudes_filepath, f"{uid}.amps"),
                           parse_dates=[6,11])
    
    def project_events(self, events):
        events["Xproj"] = self.project.LUT.coord2grid(events.loc[:,["X","Y","Z"]])[:,0]
        events["Yproj"] = self.project.LUT.coord2grid(events.loc[:,["X","Y","Z"]])[:,1]
        events["Zproj"] = self.project.LUT.coord2grid(events.loc[:,["X","Y","Z"]])[:,2]
        return events
    
    def get_waveforms(self, uid, processed=True):
        st = read(os.path.join(self.waveforms_filepath, f"{uid}.m"))

        if processed:
            st.detrend("linear")
            st.taper(type="cosine", max_percentage=0.05)
            st.select(component="Z").filter("bandpass", 
                                            freqmin=self.onset["bandpass_filters"]["P"][0],
                                            freqmax=self.onset["bandpass_filters"]["P"][1],
                                            corners=self.onset["bandpass_filters"]["P"][2],
                                            zerophase=True)
            st.select(component="[EN]").filter("bandpass", 
                                            freqmin=self.onset["bandpass_filters"]["P"][0],
                                            freqmax=self.onset["bandpass_filters"]["P"][1],
                                            corners=self.onset["bandpass_filters"]["P"][2],
                                            zerophase=True)
        print(os.getcwd())
        wf = WaveformData(min([tr.stats.starttime for tr in st]), 
                            max([tr.stats.starttime for tr in st]),
                            stations=pd.Series(list(set([tr.stats.station for tr in st]))),
                            response_inv=read_inventory(os.path.join(
                                self.project.filepath, 
                                self.response_file)
                                ),
                            water_level=self.response_params["water_level"],
                            pre_filt=self.response_params["pre_filt"],
                            remove_full_response=False,
                            read_all_stations=True)
        wf.raw_waveforms = st.copy()
        wf.waveforms = wf.raw_waveforms.copy()
        return wf

    def save_manual_picks(self, uid, picks, path):
        os.makedirs(os.path.join(path, "manual_picks"), exist_ok=True)
        picks.to_csv(os.path.join(path, "manual_picks", f"{uid}.csv"))
        return
    def get_manual_picks(self, uid, path):
        fname = os.path.join(path, "manual_picks", f"{uid}.csv")
        if not os.path.exists(fname):
            return pd.DataFrame([], columns=manual_pick_columns, index=manual_pick_index)
        picks = pd.read_csv(fname, index_col=[0,1], parse_dates=[10], 
                            dtype={"Network":str,
                                   "Station":str,
                                   "Channel":str,
                                   "Location":str,
                                   "Polarity":str,
                                   "Type":str})
        picks.rename(columns={"Station.1":"Station", "Phase.1":"Phase"}, inplace=True)
        return picks
    
    def save_amplitude_picks(self, uid, picks, path):
        os.makedirs(os.path.join(path, "amplitude_picks"), exist_ok=True)
        picks.to_csv(os.path.join(path, "amplitude_picks", f"{uid}.csv"))
        return
    def get_amplitude_picks(self, uid, path):
        fname = os.path.join(path, "amplitude_picks", f"{uid}.csv")
        if not os.path.exists(fname):
            return pd.DataFrame([], columns=amplitude_pick_columns, index=amplitude_pick_index)
        picks = pd.read_csv(fname, index_col=[0,1], parse_dates=[10,13,16], 
                            dtype={"Network":str,
                                   "Station":str,
                                   "Channel":str,
                                   "Location":str})
        print(picks)
        picks.rename(columns={"Station.1":"Station", "Component.1":"Component"}, inplace=True)
        picks.loc[:,"Filter"] = picks.loc[:,"Filter"].apply(ast.literal_eval)
        return picks

    def reformat_amplitudes(self, amps, dists, ids, depth):
        """reformats amplitudes to that used by QM for magnitude calculation"""
        stations = [trid.split(".")[1] for trid in ids]
        zdist = depth + self.project.stations.loc[stations,"Elevation"]
        return pd.DataFrame({"epi_dist" : dists,"z_dist" : zdist.to_numpy(),
                             "P_amp":np.nan,"P_freq":np.nan,"P_Time":pd.NaT,
                             "P_avg_amp":np.nan,"P_filter_gain":np.nan,
                             "S_amp":amps.to_numpy()*1e3, "S_freq":np.nan, "S_time":pd.NaT,
                             "S_avg_amp":np.nan,"S_filter_gain":np.nan,
                             "Noise_amp":0.0, "is_picked":True}, 
                             index=ids.to_numpy())



if __name__ == "__main__":
    prj = Project("/space/tg286/iceland/reykjanes/qmigrate/may21/nov23_dyke_tight")

    # det = DetectRun(prj)
    # det.get_scanmseed(pd.to_datetime("2023-11-10 12:00:00"), 1, unit="D")
    # trig = TriggerRun(prj)
    loc = LocateRun(prj)



