from yaml import safe_load, safe_dump
import pandas as pd
import numpy as np
from obspy import UTCDateTime
import subprocess
import tempfile
import struct
import os
import quakemigrate.util as util
from quakemigrate.signal.local_mag import Amplitude
from scipy.signal import iirfilter
nll_phase_format = "{0:6s} {1:4s} {2:4s} {3:1s} {4:1s} {5:1s} {6:1s} GAU {7:9.2f} -1 {8:9.2f} {9:9.2f} 1"

def read_settings(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError
    
    with open(fpath, "r") as fid:
        return safe_load(fid)
    
def write_settings(dict, fname):
    with open(fname, "w") as fid:
        safe_dump(dict, fid)

def read_dt_string(string):
    """Reads a dt string from the command line 
    checking whether the input is valid"""

    while True:
        try:
            a = pd.to_datetime(input(string), utc=True)
        except ValueError:
            print("Not a valid DT string...Try Again")
            continue
        else:
            break
        
    return a

def read_filepath(string, prepath="."):
    """Reads a filepath string from the command line 
    checking whether the path exists"""

    while True:
        fpath = os.path.join(prepath, input(string))
        if os.path.exists(fpath):
            break
        print("Not a vaild path...Try Again")
        
    return fpath

def calculate_mad(x, scale=1.4826):
    """
    Calculates the Median Absolute Deviation (MAD) of the input array x.

    Parameters
    ----------
    x : array-like
        Input data.
    scale : float, optional
        A scaling factor for the MAD output to make the calculated MAD factor a
        consistent estimation of the standard deviation of the distribution.

    Returns
    -------
    scaled_mad : array-like
        Array of scaled mean absolute deviation values for the input array, x, scaled
        to provide an estimation of the standard deviation of the distribution.

    """

    x = np.asarray(x)

    if not x.size:
        return np.nan

    if np.isnan(np.sum(x)):
        return np.nan

    # Calculate median and mad values:
    med = np.apply_over_axes(np.median, x, 0)
    mad = np.median(np.abs(x - med), axis=0)

    return scale * mad

def chunks2trace(a, new_shape):
    """
    Create a trace filled with chunks of the same value.

    Parameters:
    -----------
    a : array-like
        Array of chunks.
    new_shape : tuple of ints
        (number of chunks, chunk_length).

    Returns:
    --------
    b : array-like
        Single array of values contained in `a`.

    """

    b = np.broadcast_to(a[:, None], new_shape)
    b = np.reshape(b, np.product(new_shape))

    return b

class Filter():
    def __init__(self):
        super().__init__()   
        self._type = "none" 
        self._lowpass = None
        self._highpass = None
        self._corners = None
        self._zerophase = False

    def as_dict(self):
        return {"type":self._type, "lowpass":self._lowpass, 
                "highpass":self._highpass, "corners":self._corners,
                "zerophase":self._zerophase}

    def as_sos(self, sampling_rate):
        # Check specified freqmax is possible for this trace
        f_nyquist = 0.5 * sampling_rate
        if self._type == "none":
            return None
        elif self._type == "lowpass":
            f_crit = self._lowpass/ f_nyquist
        elif self._type == "highpass":
            f_crit = self._highpass / f_nyquist
        elif self._type == "bandpass":
            low_f_crit = self._highpass / f_nyquist
            high_f_crit = self._lowpass / f_nyquist
        if high_f_crit - 1.0 > -1e-6:
            raise util.NyquistException(self._lowpass, f_nyquist, sampling_rate)
        # if f_crit - 1.0 > -1e-6:
        #     raise util.NyquistException(self._lowpass, f_nyquist, sampling_rate)
        return iirfilter(
                    N=self._corners,
                    Wn=[low_f_crit, high_f_crit],
                    btype="bandpass",
                    ftype="butter",
                    output="sos",
                )
#     def __str__(self):
#         return f"""
# {type}
# {lowpass} - {highpass}
# {corners} {zerophase}
#         """    

    def _set_type(self):
        if self._lowpass and self._highpass:
            self._type = "bandpass"
        elif self._lowpass:
            self._type = "lowpass"
        elif self._highpass:
            self._type = "highpass"
        else:
            self._type = "none"

    @property
    def type(self):
        return self._type
    @property
    def lowpass(self):
        return self._lowpass
    @property
    def highpass(self):
        return self._highpass
    @property
    def corners(self):
        return self._corners
    @property
    def zerophase(self):
        return self._zerophase
    
    @highpass.setter
    def highpass(self, new_highpass):
        if new_highpass == 0:
            self._highpass = None
        else:
            self._highpass = new_highpass
        self._set_type()
    @lowpass.setter
    def lowpass(self, new_lowpass):
        if new_lowpass == 0:
            self._lowpass = None
        else:
            self._lowpass = new_lowpass
        self._set_type()

    @type.setter
    def type(self, new_type):
        if new_type == "none":
            self._highpass = None
            self._lowpass = None
        elif new_type == "lowpass" and self._highpass:
            self._highpass = None
        elif new_type == "highpass" and self._lowpass:
            self._lowpass = None
        self._type = new_type
    @corners.setter
    def corners(self, new_corners):
        self._corners = new_corners
    @zerophase.setter
    def zerophase(self, new_zerophase):
        self._zerophase = new_zerophase

class Taper():
    def __init__(self):
        super().__init__()   
        self._type = "none" 
        self._maxpercentage = None
    @property
    def type(self):
        return self._type
    @property
    def maxpercentage(self):
        return self._maxpercentage
    @type.setter
    def type(self, new_type):
        self._type = new_type
    @maxpercentage.setter
    def maxpercentage(self, new_maxpercentage):
        self._maxpercentage = new_maxpercentage

# class MyStream():
#     def __init__(self):
#         super().__init__() 

#         self.raw_stream = None


def preprocess_stream(stream, filter, taper, detrend):
    if detrend:
        stream.detrend("linear")
    
    if taper.type == "none":
        pass
    else:
        stream.taper(max_percentage=taper.maxpercentage, type=taper.type)

    if filter.type == "none":
        pass
    elif filter.type == "bandpass":
        stream.filter(filter.type, freqmin=filter.highpass,
                      freqmax=filter.lowpass, corners=filter.corners,
                      zerophase=filter.zerophase)
    elif filter.type == "lowpass":
        stream.filter(filter.type, freq=filter.lowpass, 
                      corners=filter.corners,
                      zerophase=filter.zerophase)
    elif filter.type == "lowpass":
        stream.filter(filter.type, freq=filter.highpass, 
                      corners=filter.corners,
                      zerophase=filter.zerophase)
        
    return stream

class Pick:
    def __init__(self, network, station, location, channel,
                 phase, picktime, quality, onset, polarity,
                 amplitude, period):
        super().__init__()  

        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.phase = phase
        self.picktime = picktime
        self.quality = quality
        self.onset = onset
        self.polarity = polarity
        self.amplitude = amplitude
        self.period = period

        self.component = self.channel[-1]

    def get_pandas_series(self):
        return pd.Series([self.network, self.station, self.location, self.channel, 
                          self.quality, self.polarity, self.onset, self.phase, 
                          self.picktime, self.amplitude, self.period],
                        index=["Network", "Station", "Location", "Channel", 
                                "Quality", "Polarity", "Onset", "Phase",
                                "PickTime", "Amplitude", "Period"]
                        )
    def get_pandas_dataframe(self):
        df = pd.DataFrame([[self.network, self.station, self.location, self.channel, 
                          self.quality, self.polarity, self.onset, self.phase, 
                          self.picktime, self.amplitude, self.period]],
                        columns=["Network", "Station", "Location", "Channel", 
                                "Quality", "Polarity", "Onset", "Phase",
                                "PickTime", "Amplitude", "Period"],
                        )
        df.set_index(["Station","Phase"], inplace=True)
        df.loc[:,"Quality"] = df.loc[:,"Quality"].astype(int)
        return df
def time_offset(t1, t2):
    return (t1-t2).total_seconds()

class NonLinLoc():
    def __init__(self, root):
        self.tmp_directory = "/tmp"
        self.root = root + os.path.sep + "nonlinloc"

        ## generic settings
        self.message_flag = 1
        self.random_number = 12345
        self.transform = "none" # probaly should be updated but not needed for relocation

        ## Vel2Grid settings
        self.vg_out = None
        self.vg_grid = NLLGrid()

        ## Grid2Time settings
        self.gt_ttimeFileRoot = None
        self.gt_out = None
        self.gt_iSwapBytesOnInput = 0
        self.gt_plfd = [1e-3, 0]
        self.stations = None

        # nonlinloc settings
        self.nll_signature = None
        self.nll_comment = None
        self.nll_obsFileType = "NLLOC_OBS"
        self.nll_iSwapBytes = 0
        self.nll_lochypout = "SAVE_NLLOC_ALL"
        self.nll_locsearch = dict(type="OCT", 
                                  initNumCells_x=10, 
                                  initNumCells_y=10,
                                  initNumCells_z=4,
                                  minNodeSize=0.01,
                                  maxNumNodes=50000, 
                                  numScatter=5000,
                                  useStationsDensity=False,
                                  stopOnMinNodeSize=False)
        self.nll_locmeth = dict(method="EDT_OT_WT_ML",
                                maxDistStaGrid=50,
                                minNumberPhases=4,
                                maxNumberPhases=False,
                                minNumberSphases=False,
                                VpVsRatio=False,
                                maxNum3DGridMemory=False,
                                minDistStaGrid=False,
                                iRejectDuplicateArrivals=True)
        self.nll_locgau = dict(SigmaTime=0., CorrLen=0.)
        self.nll_locgau2 = dict(SigmaTfraction=0.01,SigmaTmin=0.0,SigmaTmax=2.)
        self.nll_locqual2err = {0 : 0.1}
        self.nll_locgrid = NLLGrid()
        self.nll_locangles = dict(angleMode="ANGLES_YES", qualityMin=5)
    
    def read_yaml(self, yaml_file):
        settings = read_settings(yaml_file)
        self.message_flag = settings.get('message_flag', 1)
        self.random_number = settings.get("random_number", 12345)
        self.transform = settings.get('transform', "none")

        self.vg_out = os.path.join(self.root, settings.get("vg_out", "model/TMP"))
        self.vg_grid = NLLGrid()
        self.vg_grid.populate(settings.get("vg_grid", {}))
        
        self.gt_ttimeFileRoot = os.path.join(self.root, settings.get("gt_ttimeFileRoot", "model/TMP"))
        self.gt_out = os.path.join(self.root, settings.get("gt_out", "time/TMP"))

        self.nll_signature = settings.get("nll_signature", "")
        self.nll_comment = settings.get("nll_comment", "")
        self.nll_locsearch = settings.get("nll_locsearch", self.nll_locsearch)
        self.nll_locmeth = settings.get("nll_locmeth", self.nll_locmeth)
        self.nll_locgau = settings.get("nll_locgau", self.nll_locgau)
        self.nll_locgau2 = settings.get("nll_locgau2", self.nll_locgau2)
        self.nll_locqual2err = settings.get("nll_locqual2err", self.nll_locqual2err)
        self.nll_locgrid = NLLGrid()
        self.nll_locgrid.populate(settings.get("nll_locgrid", {}))
        self.nll_locangles = settings.get("nll_locangles", self.nll_locangles)
    
    def write_yaml(self, fpath):
        settings = {"message_flag" : self.message_flag,
                    "random_number" : self.random_number,
                    "transform" : self.transform,
                    "vg_out" : self.vg_out,
                    "vg_grid" : self.vg_grid.as_dict(),
                    "gt_ttimeFileRoot" : self.gt_ttimeFileRoot,
                    "gt_out" : self.gt_out,
                    "nll_signature" : self.nll_signature,
                    "nll_locsearch" : self.nll_locsearch,
                    "nll_locmeth" : self.nll_locmeth,
                    "nll_locgau" : self.nll_locgau,
                    "nll_locgau2" : self.nll_locgau2,
                    "nll_locqual2err" : self.nll_locqual2err,
                    "nll_locgrid" : self.nll_locgrid.as_dict(),
                    "nll_locangles" : self.nll_locangles}
        write_settings(settings, os.path.join(fpath, "nonlinloc.yaml"))
        return


    def read_travel_time_tables(self, station, phase):
        froot = os.path.sep+os.path.join(*self.gt_out.split(os.path.sep))
        hdr = ".".join([froot, phase, station, "time.hdr"])
        buf = ".".join([froot, phase, station, "time.buf"])

        with open(hdr, "r") as fid:
            lines = [line.split() for line in fid.readlines()]
        nx, ny, nz = [int(val) for val in lines[0][:3]]
        x0, y0, z0 = [float(val) for val in lines[0][3:6]]
        dx = float(lines[0][6])

        nxyz = nx*ny*nz
        with open(buf, "rb") as fid:
            data = fid.read(nxyz*4)
        data = np.array(struct.unpack("f"*nxyz, data))
        data = data.reshape((nx,ny,nz), order="C")

        return data, [nx,ny,nz], [x0,y0,z0], dx

    def make_traveltime_tables(self, dir):
        os.makedirs(os.path.sep+os.path.join(*self.vg_out.split(os.path.sep)[:-1]), exist_ok=True)
        os.makedirs(os.path.sep+os.path.join(*self.gt_out.split(os.path.sep)[:-1]), exist_ok=True)
        print(os.path.sep+os.path.join(*self.vg_out.split(os.path.sep)[:-1]))
        print(os.path.sep+os.path.join(*self.gt_out.split(os.path.sep)[:-1]))
        control_file = self.write_control_file("Vel2Grid", dir, phase="P")
        ret_code = subprocess.call(f"Vel2Grid {control_file}", shell=True)
        control_file = self.write_control_file("Vel2Grid", dir, phase="S")
        ret_code = subprocess.call(f"Vel2Grid {control_file}", shell=True)

        control_file = self.write_control_file("Grid2Time", dir, phase="P")
        ret_code = subprocess.call(f"Grid2Time {control_file}", shell=True)
        control_file = self.write_control_file("Grid2Time", dir, phase="S")
        ret_code = subprocess.call(f"Grid2Time {control_file}", shell=True)
        
    def relocate_event(self, uid, picks):
        tempdir = tempfile.mkdtemp(prefix="nll_")
        print("NLLTEMPDIR", tempdir)
        pick_file = self.write_nll_obs(tempdir, picks)
        control_file = self.write_control_file("NonLinLoc", tempdir, uid=uid, pickfile=pick_file)

        ## assemble all the files into the rundir
        subprocess.call(f"NLLoc {control_file}", shell=True)
        # LocSum SizeGridRoot decimFactor OutRoot LocRoot
        subprocess.call(f"LocSum {tempdir}/last 1 {tempdir}/{uid} {tempdir}/last", shell=True)

        return tempdir


    def write_control_file(self, program, dir, uid=None, 
                           pickfile=None, phase=None):
        control_file = [f"""
CONTROL {self.message_flag} {self.random_number}
TRANS {self.get_trans_statement()}
"""]
        if program == "Vel2Grid":
            control_file.append(f"""
VGOUT {self.vg_out}
VGTYPE {phase}
VGGRID {self.vg_grid.get_statement()}
""")
            control_file.append(self.vmodel.get_layers())
        
        elif program == "Grid2Time":
            control_file.append(f"""
GTFILES {self.gt_ttimeFileRoot} {self.gt_out} {phase} {self.gt_iSwapBytesOnInput}
GTMODE {'GRID2D' if self.vg_grid.dimension == 2 else 'GRID3D'} ANGLES_YES
GT_PLFD {"{} {}".format(*self.gt_plfd)}
""")
            control_file.append(self.get_stations_for_Grid2Time())

        elif program == "NonLinLoc":
            control_file.append(f"""
LOCSIG {self.nll_signature}
LOCCOM {uid}
LOCFILES {pickfile} {self.nll_obsFileType} {self.gt_out} {dir}/loc {self.nll_iSwapBytes}
LOCHYPOUT {self.nll_lochypout}
""")
            if self.nll_locsearch["type"] == "OCT":
                nll_locsearch = ["LOCSEARCH", self.nll_locsearch['type']]
                nll_locsearch.append("{}".format(self.nll_locsearch["initNumCells_x"]))
                nll_locsearch.append("{}".format(self.nll_locsearch["initNumCells_y"]))
                nll_locsearch.append("{}".format(self.nll_locsearch["initNumCells_z"]))
                nll_locsearch.append("{}".format(self.nll_locsearch["minNodeSize"]))
                nll_locsearch.append("{}".format(self.nll_locsearch["maxNumNodes"]))
                nll_locsearch.append("{}".format(self.nll_locsearch["numScatter"]))
                nll_locsearch.append("{}".format(1 if self.nll_locsearch["useStationsDensity"] else 0))
                nll_locsearch.append("{}".format(1 if self.nll_locsearch["stopOnMinNodeSize"] else 0))
            else:
                raise NotImplementedError
            control_file.append(" ".join(nll_locsearch))

            nll_locmeth = ["LOCMETH", self.nll_locmeth['method']]
            nll_locmeth.append("{}".format(self.nll_locmeth["maxDistStaGrid"] if self.nll_locmeth["maxDistStaGrid"] else -1))
            nll_locmeth.append("{}".format(self.nll_locmeth["minNumberPhases"] if self.nll_locmeth["minNumberPhases"] else -1))
            nll_locmeth.append("{}".format(self.nll_locmeth["maxNumberPhases"] if self.nll_locmeth["maxNumberPhases"] else -1))
            nll_locmeth.append("{}".format(self.nll_locmeth["minNumberSphases"] if self.nll_locmeth["minNumberSphases"] else -1))
            nll_locmeth.append("{}".format(self.nll_locmeth["VpVsRatio"] if self.nll_locmeth["VpVsRatio"] else -1))
            nll_locmeth.append("{}".format(self.nll_locmeth["maxNum3DGridMemory"] if self.nll_locmeth["maxNum3DGridMemory"] else -1))
            nll_locmeth.append("{}".format(self.nll_locmeth["minDistStaGrid"] if self.nll_locmeth["minDistStaGrid"] else -1))
            nll_locmeth.append("{}".format(1 if self.nll_locmeth["iRejectDuplicateArrivals"] else 0))
            control_file.append(" ".join(nll_locmeth))

            control_file.append(f"""
LOCGAU {self.nll_locgau['SigmaTime']} {self.nll_locgau['CorrLen']}
LOCGAU2 {self.nll_locgau2['SigmaTfraction']} {self.nll_locgau2['SigmaTmin']} {self.nll_locgau2['SigmaTmax']}
""")
            control_file.append("LOCQUAL2ERR " + " ".join([str(self.nll_locqual2err[key]) 
                                                           for key in sorted(self.nll_locqual2err.keys())]))
            control_file.append("LOCGRID "+self.nll_locgrid.get_statement() + " SAVE")

            control_file.append(f"LOCANGLES {self.nll_locangles['angleMode']} {self.nll_locangles['qualityMin']}")

        with open(os.path.join(dir, "run.ctrl"), "w") as fid:
            fid.write("\n".join(control_file))
            
        return os.path.join(dir, "run.ctrl")
    def write_nll_obs(self, dir, picks): 
        """ Station name (char*6)
                station name or code 
            Instrument (char*4)
                instument identification for the trace for which the time pick corresponds (i.e. SP, BRB, VBB) 
            Component (char*4)
                component identification for the trace for which the time pick corresponds (i.e. Z, N, E, H) 
            P phase onset (char*1)
                description of P phase arrival onset; i, e 
            Phase descriptor (char*6)
                Phase identification (i.e. P, S, PmP) 
            First Motion (char*1)
                first motion direction of P arrival; c, C, u, U = compression; d, D = dilatation; +, -, Z, N; . or ? = not readable. 
            Date (yyyymmdd) (int*6)
                year (with century), month, day 
            Hour/minute (hhmm) (int*4)
                Hour, min 
            Seconds (float*7.4)
                seconds of phase arrival 
            Err (char*3)
                Error/uncertainty type; GAU 
            ErrMag (expFloat*9.2)
                Error/uncertainty magnitude in seconds 
            Coda duration (expFloat*9.2)
                coda duration reading 
            Amplitude (expFloat*9.2)
                Maxumim peak-to-peak amplitude 
            Period (expFloat*9.2)
                Period of amplitude reading 
            PriorWt (expFloat*9.2) 
        """
        lines = []
        for index, pick in picks.iterrows():
            # print(pick.Station, pick.Channel[:-1], pick.Channel[-1], 
            # pick.Onset, pick.Polarity, pick.PickTime.strftime("%Y%m%d %H%M %S.%f"),
            # self.nll_locqual2err[pick.Quality], 0.0, pick.Amplitude, pick.Period, 1)

            # print(index)
            # print(pick)

            station = index[0]
            instrument = pick.Channel[:-1] if pd.notna(pick.Channel) else "?"
            component = pick.Channel[-1] if pd.notna(pick.Channel) else "?"
            onset = pick.Onset if pd.notna(pick.Onset) else "?"
            phase = index[1]
            first_motion = pick.Polarity if pd.notna(pick.Polarity) else "?"
            datestr = pick.PickTime.strftime("%Y%m%d %H%M %S.%f")
            error = self.nll_locqual2err[pick.Quality]
            amplitude = pick.Amplitude if pd.notna(pick.Amplitude) else -1.
            period = pick.Period if pd.notna(pick.Period) else -1.

            
            # print(station, type(station)) # <class 'str'>
            # print(instrument, type(instrument)) # <class 'str'>
            # print(component, type(component)) # <class 'str'>
            # print(onset, type(onset)) # <class 'str'>
            # print(phase, type(phase)) # <class 'str'>
            # print(first_motion, type(first_motion)) # <class 'float'>
            # print(datestr, type(datestr)) # <class 'str'>
            # print(error, type(error)) # <class 'float'>
            # print(amplitude, type(amplitude)) # <class 'float'>
            # print(period, type(period)) # <class 'float'>

            lines.append(nll_phase_format.format(station, instrument, component, onset, phase, 
                                                 first_motion, datestr, error, amplitude, period))
        with open(os.path.join(dir, "pickfile.obs"), "w") as fid:
            fid.write("\n".join(lines))
        return os.path.join(dir, "pickfile.obs")
    def get_stations_for_Grid2Time(self):
        lines = []
        if self.transform == "none":
            for index, row in self.stations.iterrows():
                lines.append(f"GTSRCE {index} XYZ {row.Xproj} {row.Yproj} {-row.Elevation} 0.0")
        else:
            print("NOT IMPLEMENTED YET")
            raise NotImplementedError
        return "\n".join(lines)
    def read_summed_hypfile(self, fname):
        # pass
        ev = NonLinLocEvent()
        ev.populate_from_summed_hypfile(fname)
        return ev
    def latlon2xy(self, lat, lon):
        if self.transform == "none":
            print("Note: This is a Grid with no Transform")
            return lat, lon
        else:
            print("Not implemented")
            return
    def xy2latlon(self, x, y):
        if self.transform == "none":
            print("Note: This is a Grid with no Transform")
            return x, y
        else:
            print("Not implemented")
            return
    def get_trans_statement(self):
        if self.transform == "none":
            return "NONE"
    # def write_stations(self, stations):
    #     print(stations)

class NLLGrid():
    def __init__(self):
        self.xnum = 0
        self.ynum = 0
        self.znum = 0
        self.xorig = None
        self.yorig = None
        self.zorig = None
        self.dx = None
        self.grid_type = None
        self.dimension = None

    def populate(self, dic):
        if len(dic) == 0:
            return
        self.dimension = dic["dimension"]
        if self.dimension == 2:
            self.xnum = 2
            self.ynum = dic["num"][0]
            self.znum = dic["num"][1]
            self.xorig, self.yorig = 0., 0.
            self.zorig = dic["orig"]
            self.dx = dic["dx"]
        elif self.dimension == 3:
            self.xnum,self.ynum,self.znum = dic["num"]
            self.xorig, self.yorig, self.zorig = dic["orig"]
            self.dx = dic["dx"]
        self.grid_type = dic["grid_type"]
    def get_statement(self):
        """xNum yNum zNum xOrig yOrig zOrig dx dy dz gridType"""
        num = f"{self.xnum} {self.ynum} {self.znum} "
        orig = f"{self.xorig} {self.yorig} {self.zorig} "
        dx = f"{self.dx} {self.dx} {self.dx} "
        return num+orig+dx+" "+self.grid_type
    def as_dict(self):
        return {"num" : [self.ynum, self.znum] if self.dimension == 2 else [self.xnum, self.ynum, self.znum],
                                 "orig" : self.zorig if self.dimension == 2 else [self.xorig, self.yorig, self.zorig],
                                 "dx" : self.dx,
                                 "grid_type" : self.grid_type,
                                 "dimension" : self.dimension}


class VModel():
    def __init__(self, vmodel):
        self._depth = vmodel.Depth
        self._pmodel = vmodel.Vp
        if "Vs" in vmodel.columns:
            self._smodel = vmodel.Vs
        else:
            print("NOT IMPLEMENTED YET")
            raise ValueError
        self.p_gradient = self.get_gradient(self._pmodel.to_numpy(), self._depth.to_numpy())
        self.s_gradient = self.get_gradient(self._smodel.to_numpy(), self._depth.to_numpy())

        self.nlayers = len(vmodel)
        self.vmodel = vmodel
    def get_layers(self):
        """return layers for NLL control file"""

        lines = []
        for i in range(self.nlayers):
            lines.append(self.get_layer(i))
            print(self.get_layer(i))

        return "\n".join(lines)
    def get_gradient(self, vel, dep):
        grad = (vel[1:] - vel[:-1]) / (dep[1:] - dep[:-1])
        grad = np.hstack((grad, 0.))
        return grad
    def get_layer(self, ii):
        pmodel = f"LAYER {self.depth[ii]:.2f} {self.pmodel[ii]:3f} {self.p_gradient[ii]:4f} "
        smodel = f"{self.smodel[ii]:.3f} {self.s_gradient[ii]:.4f} "
        rhomodel = f"0.0 0.0"
        return pmodel + smodel + rhomodel

    @property
    def pmodel(self):
        return self._pmodel
    @property
    def smodel(self):
        return self._smodel
    @property
    def depth(self):
        return self._depth

class NonLinLocEvent():
    def __init__(self):
        self.scatter = None
        self.phases = pd.DataFrame([])
    def get_scatter_percentiles(self, percentiles):
        out = np.zeros((len(percentiles), 3))
        for i,percentile in enumerate(percentiles):
            out[i,0] = np.percentile(self.scatter.Xproj, percentile)
            out[i,1] = np.percentile(self.scatter.Yproj, percentile)
            out[i,2] = np.percentile(self.scatter.Zproj, percentile)
        return out
    def populate_from_summed_hypfile(self, fname):

        with open(fname, "r") as fid:
            rawlines = fid.readlines()
        lines = [line.split() for line in rawlines]
        
        i = 0
        for line in lines:
            if len(line) == 0:
                i += 1
                continue
            if line[0] == "HYPOCENTER":
                self._read_hypocentre(line)
            if line[0] == "GEOGRAPHIC":
                self._read_geographic(line)
            if line[0] == "QUALITY":
                self.quality = self._read_quality(line)
            if line[0] == "VPVSRATIO":
                self.vpvs = self._read_vpvs_ratio(line)
            if line[0] == "STATISTICS":
                self.statistics = self._read_statistics(line)
            if line[0] == "STAT_GEOG":
                self._read_stat_geog(line)
            if line[0] == "PHASE":
                start_phaseline = i
            if line[0] == "END_PHASE":
                self.phases = self._read_phases(rawlines[start_phaseline+1:i])
            if line[0] == "SCATTER":
                n_scatsamples = int(line[2])
                start_scatline = i
            if line[0] == "END_SCATTER":
                print("NSCATTER", n_scatsamples, i-start_scatline-1)
                # assert i-start_scatline-1 == n_scatsamples
                self.scatter = self._read_scatter(lines[start_scatline+1:i])
            i+=1
    
    def _read_hypocentre(self, line):
        dic = _read_line_to_dict(line[1:7])
        self.Xproj = float(dic["x"])
        self.Yproj = float(dic["y"])
        self.Zproj = float(dic["z"])
    def _read_geographic(self, line):
        self.X = float(line[9])
        self.Y = float(line[11])
        self.Z = float(line[13])
        self.otime = pd.to_datetime(f"{line[2]}-{line[3]}-{line[4]} {line[5]}:{line[6]}:{line[7]}")
    def _read_quality(self, line):
        dic = _read_line_to_dict(line[1:-6])
        for key in dic.keys():
            if key == "Nphs":
                dic[key] = int(dic[key])
                continue
            dic[key] = float(dic[key])
        return dic
    def _read_vpvs_ratio(self, line):
        dic = _read_line_to_dict(line[1:])
        dic["VpVsRatio"] = float(dic["VpVsRatio"])
        dic["Npair"] = int(dic["Npair"])
        dic["Diff"] = float(dic["Diff"])
        return dic
    def _read_statistics(self, line):
        self.expectXproj = float(line[2])
        self.expectYproj = float(line[4])
        self.expectZproj = float(line[6])
        dic = _read_line_to_dict(line[7:])
        for key in dic.keys():
            dic[key] = float(dic[key])
        return dic
    def _read_stat_geog(self, line):
        self.expectX = float(line[2])
        self.expectY = float(line[4])
        self.expectZ = float(line[6])
    def _read_phases(self, rawlines):
        # trim lines
        lines = [line.split(">")[1].split() for line in rawlines]
        index = [(line.split()[0], line.split()[4]) for line in rawlines]
        df = pd.DataFrame(lines, columns=["TTpred","Residual","Weight",
                                          "StaLocX","StaLocY","StaLocZ",
                                          "Distance","station_azimuth","receiver_azimuth",
                                          "receiver_takeoffangle",
                                          "RQual","Tcorr","TTerr"],
                            index=index)
        df.loc[:,"Station"] = [line.split()[0] for line in rawlines]
        df.loc[:,"Phase"] = [line.split()[4] for line in rawlines]
        df.set_index(["Station", "Phase"], inplace=True)
        df = df.loc[:, ["TTpred","Residual","Weight",
                    "Distance","station_azimuth","receiver_azimuth",
                    "receiver_takeoffangle"]].astype(float)
        return df
    def _read_scatter(self, lines):
        df = pd.DataFrame(lines, columns=["Xproj","Yproj","Zproj","Probability"])
        df = df.astype(float)
        return df

    
    
def _read_line_to_dict(line):
    line_length = len(line)
    if not line_length%2 == 0:
        raise ValueError("Length of line should be even", line, line_length)
    return dict([(line[i], line[i+1]) for i in range(0,line_length,2)])


class myAmplitude(Amplitude):
    """QM viewer implementation of the QM Amplitude class. Needed as QM viewer doesn't always 
    use QM events"""

    def _get_picks(self, station, manual_picks, qm_picks, nll_event, qm_otime):
        if isinstance(nll_event, NonLinLocEvent) and (station, "P") in manual_picks.index and (station, "S") in manual_picks.index:
            ppick = manual_picks.loc[(station, "P"), "PickTime"]
            ptraveltime = time_offset(ppick, nll_event.otime)
            spick = manual_picks.loc[(station, "S"), "PickTime"]
            straveltime = time_offset(spick, nll_event.otime)
            qmflag = False
        elif isinstance(nll_event, NonLinLocEvent) and (station, "P") in manual_picks.index:
            ppick = manual_picks.loc[(station, "P"), "PickTime"]
            ptraveltime = time_offset(ppick, nll_event.otime)
            straveltime = ptraveltime * 1.73
            spick = ppick + pd.to_timedelta(straveltime-ptraveltime, unit="S")
            qmflag = False
        elif isinstance(nll_event, NonLinLocEvent) and (station, "S") in manual_picks.index:
            spick = manual_picks.loc[(station, "S"), "PickTime"]
            straveltime = time_offset(spick, nll_event.otime)
            ptraveltime = straveltime / 1.73
            ppick = spick - pd.to_timedelta(straveltime-ptraveltime, unit="S")
            qmflag = False
        elif (station, "P") in qm_picks.index and (station, "S") in qm_picks.index:
            ppick = qm_picks.loc[(station, "P"), "ModelledTime"].tz_convert(None)
            ptraveltime = time_offset(ppick, qm_otime)
            spick = qm_picks.loc[(station, "S"), "ModelledTime"].tz_convert(None)
            straveltime = time_offset(spick, qm_otime)
            qmflag = True
        elif (station, "P") in qm_picks.index:
            ppick = qm_picks.loc[(station, "P"), "ModelledTime"].tz_convert(None)
            ptraveltime = time_offset(ppick, qm_otime)
            straveltime = ptraveltime * 1.73
            spick = ppick + pd.to_timedelta(straveltime-ptraveltime, unit="S")
            qmflag = True
        elif (station, "S") in qm_picks.index:
            spick = qm_picks.loc[(station, "S"), "ModelledTime"].tz_convert(None)
            straveltime = time_offset(spick, qm_otime)
            ptraveltime = straveltime / 1.73
            ppick = spick - pd.to_timedelta(straveltime-ptraveltime, unit="S")
            qmflag = True
        else:
            return None, None, None, None, None
        
        return ppick, ptraveltime, spick, straveltime, qmflag

    def _get_amplitude_windows(
        self, p_pick, p_ttime, s_pick, s_ttime, fraction_tt, marginal_window=1
    ):
        """
        Calculate the start and end time of the windows to measure the max P- and S-wave
        amplitudes in. This is done on the basis of the pick times, the event marginal
        window, the traveltime and uncertainty and the specified S-wave signal window.

        P_window_start : P_pick - marginal_window - traveltime_uncertainty
        P_window_end : equivalent to start, or S_pick time; whichever is
                       earlier
        S_window_start : same as P
        S_window_end : S_pick + signal_window + marginal_window +
                       traveltime_uncertainty

        traveltime_uncertainty = traveltime * fraction_tt
            (where fraction_tt is as specified for the lookup table).

        Parameters
        ----------
        p_ttimes : array-like
            Array of interpolated P traveltimes to the requested grid position.
        s_ttimes : array-like
            Array of interpolated S traveltimes to the requested grid position.
        fraction_tt : float
            An estimate of the uncertainty in the velocity model as a function of the
            traveltime.

        Returns
        -------
        windows : array-like
            [[P_window_start, P_window_end], [S_window_start, S_window_end]]

        Raises
        ------
        PickOrderException
            If the P pick for an event/station is later than the S pick.

        """

        # Check p_pick is before s_pick
        try:
            assert p_pick < s_pick
        except AssertionError:
            raise util.PickOrderException(p_pick, s_pick)

        # For P:
        p_start = UTCDateTime(p_pick - pd.to_timedelta(marginal_window - p_ttime * fraction_tt, unit="S"))
        p_end = UTCDateTime(p_pick + pd.to_timedelta(marginal_window + p_ttime * fraction_tt, unit="S"))
        # For S:
        s_start = UTCDateTime(s_pick - pd.to_timedelta(marginal_window - s_ttime * fraction_tt, unit="S"))
        s_end = UTCDateTime(
            s_pick
            + pd.to_timedelta(marginal_window
            + s_ttime * fraction_tt
            + self.signal_window, unit="S")
        )

        # Check for overlaps
        if s_start < p_end:
            mid_time = p_end + (s_start - p_end) / 2
            windows = [[p_start, mid_time], [mid_time, s_end]]
        elif s_start - p_end < self.signal_window:
            windows = [[p_start, s_start], [s_start, s_end]]
        else:
            windows = [[p_start, p_end + self.signal_window], [s_start, s_end]]

        return windows   