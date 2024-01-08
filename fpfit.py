import numpy as np
from _mtconvert import *
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiPoint, MultiLineString
from shapely import Polygon
from descartes import PolygonPatch
from scipy.spatial import ConvexHull, Delaunay
from doublecouple import DoubleCouple

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap

import mplstereonet


def fp_grid_search(dataframe: pd.DataFrame, 
                   dstrike:float=20., ddip:float=10., drake:float=20, 
                   strikelim: tuple=(0,180), diplim: tuple=(0,91), rakelim: tuple=(-180,180),
                   output_range: bool=False):

    minstrike, maxstrike = strikelim
    mindip, maxdip = diplim
    minrake, maxrake = rakelim
    
    strikes = np.arange(minstrike, maxstrike, dstrike)
    dips = np.arange(mindip, maxdip, ddip)
    rakes = np.arange(minrake, maxrake, drake)

    STRIKES, DIPS, RAKES = np.meshgrid(strikes, dips, rakes, indexing="ij")
    STRIKES = STRIKES.flatten()
    DIPS = DIPS.flatten()
    RAKES = RAKES.flatten()
    ntest = len(STRIKES)

    # correct parameters to only positive dips and to keep parameters within ranges
    STRIKES = (STRIKES+360)%360
    # correct dip
    mask = DIPS > 90
    DIPS[mask] = 180 - DIPS[mask]
    STRIKES[mask] = (STRIKES[mask]+180)%360
    mask = DIPS < 0
    DIPS[mask] = np.abs(DIPS[mask])
    STRIKES[mask] = (STRIKES[mask]+180)%360
    # correct rakes
    RAKES[RAKES<-180] = 360 + RAKES[RAKES<-180]
    RAKES[RAKES>180] = -(360 - RAKES[RAKES>180])

    # get the predicted radiation amplitude
    rad = prad(STRIKES, DIPS, RAKES, 
               dataframe.receiver_takeoffangle, 
               dataframe.station_azimuth)

    # set the weight array 
    weight = np.sqrt(np.abs(rad))

    # set the predicted polarities
    pred = np.sign(rad) * 0.5

    mfit, mdenom = calculate_misfit(dataframe.Polarity.to_numpy(), pred,
                                    dataframe.weight.to_numpy(), weight)


    # plt.figure()
    # plt.plot(mfit, "k+", ms=1)
    
    solution = get_solution(mfit, mdenom)
    # print(mfit.shape)
    # print(solution, mfit[solution])
    # plt.axvline(solution)
    # plt.axhline(mfit[solution])
    # plt.axhline(mfit[solution] * 1.1)
    solution = DoubleCouple([STRIKES[solution], DIPS[solution], RAKES[solution]])

    # observed = dataframe.Polarity.to_numpy()
    # # get the predicted radiation amplitude
    # rad = prad(solution.plane1.strike, solution.plane1.dip, solution.plane1.rake, 
    #            df.TakeOffAngle, df.Azimuth)
    # weight = np.sqrt(np.abs(rad))
    # pred = np.sign(rad) * 0.5
    # predicted = pred
    # weight_ob = dataframe.weight.to_numpy()
    # weight_rad = weight

    # misfit_nom = np.sum(np.abs((observed*0.5)[:,np.newaxis] - predicted) * weight_ob[:,np.newaxis] * weight_rad, axis=0)
    # misfit_denom = np.sum(weight_ob[:,np.newaxis] * weight_rad, axis=0)
    # misfit = misfit_nom / misfit_denom
    # print(misfit)
    # # sys.exit()

    if output_range:
        solution_range = [DoubleCouple([STRIKES[solution], DIPS[solution], RAKES[solution]]) for solution in get_solution_range(mfit)] 
        return solution, solution_range, np.min(mfit)
    return solution, [], np.min(mfit)

def plot_all_fault_planes(ax, fps, color="yellow", alpha=0.1, linewidth=1):
    for bf in fps:
        ax.plane(bf.plane1.strike, bf.plane1.dip, "-", c=color, lw=linewidth, alpha=alpha)
        ax.plane(bf.plane2.strike, bf.plane2.dip, "-", c=color, lw=linewidth, alpha=alpha)
    return ax

def plot_fault_plane_histogram(ax, fps, bins=50, segments=100):
    # fault planes
    positions_x, positions_y = mplstereonet.stereonet_math.plane([fp.plane1.strike for fp in fps], 
                                                [fp.plane1.dip for fp in fps], segments=segments)
    positions1 = np.vstack((positions_x.flatten(), positions_y.flatten())).T

    positions_x, positions_y = mplstereonet.stereonet_math.plane([fp.plane2.strike for fp in fps], 
                                                [fp.plane2.dip for fp in fps], segments=segments)
    positions2 = np.vstack((positions_x.flatten(), positions_y.flatten())).T
    positions = np.vstack((positions1, positions2))

    a, x, y = np.histogram2d(positions[:,0],positions[:,1], bins=bins)
    X, Y = np.meshgrid(x, y)

    cmap = ListedColormap(['grey'])
    cmap.set_under("white")
    ax.pcolormesh(X, Y, a.T, zorder=-1, vmin=1, cmap=cmap)

    return ax

def plot_PT_errorbar(fps, ax, alpha=1.5, T_color="r", P_color="b"):
    ax = plot_outlined_polygon(ax, get_line_positions(fps, "P", alpha), P_color)
    ax = plot_outlined_polygon(ax, get_line_positions(fps, "T", alpha), T_color)
    return ax

def plot_PT_axes(fps, ax, T_marker="s", P_marker="s",
                     T_color="r", P_color="b", markersize=10):
    ax.line(fps.axis["P"]["dip"], fps.axis["P"]["azimuth"], 
                marker=P_marker, color=P_color, ms=markersize)
    ax.line(fps.axis["T"]["dip"], fps.axis["T"]["azimuth"], 
                marker=T_marker, color=T_color, ms=markersize)
    return ax

def get_line_positions(fps, option, alpha):
    positions = np.vstack(mplstereonet.stereonet_math.line([fp.axis[option]["dip"] for fp in fps], 
                                                            [fp.axis[option]["azimuth"] for fp in fps])).T
    # qhull_Paxis = ConvexHull(positions)
    alpha_shape = alphashape(positions, alpha)
    return alpha_shape

def plot_outlined_polygon(ax, polygon, color):
    if isinstance(polygon, Polygon):
        plot_polygon(ax, polygon, alpha=0.5, fc=color, ec="none")
        plot_polygon(ax, polygon, fc="none", ec=color)
    else:
        for polygon in list(polygon.geoms):
            plot_polygon(ax, polygon, alpha=0.5, fc=color, ec="none")
            plot_polygon(ax, polygon, fc="none", ec=color)
    return ax

def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection  

def calculate_misfit(observed: np.ndarray, predicted: np.ndarray, weight_ob: np.ndarray, weight_rad: np.ndarray):
    misfit_nom = np.sum(np.abs((observed*0.5)[:,np.newaxis] - predicted) * weight_ob[:,np.newaxis] * weight_rad, axis=0)
    misfit_denom = np.sum(weight_ob[:,np.newaxis] * weight_rad, axis=0)
    misfit = misfit_nom / misfit_denom
    misfit[misfit==0] += 1e-6

    return misfit, misfit_denom

def get_solution_range(misfit: np.ndarray, limit: float=0.1):
    n = len(misfit)
    min_misfit = np.min(misfit)
    return np.mgrid[:n][misfit < min_misfit*(1+limit)]

def get_solution(misfit: np.ndarray, misfit_denom: np.ndarray):
    # count how many solutions have the same (minimum) misfit
    min_misfit = np.min(misfit)
    n_solutions = np.sum(misfit <= min_misfit)
    if n_solutions == 1:
        indmin = np.argmin(misfit)
    else:
        # maximise nominator to get "best" solution
        ind = np.argmax(misfit_denom[misfit <= min_misfit])
        indmin = np.mgrid[:len(misfit)][misfit <= min_misfit][ind]
    return indmin

def circumcenter(points: np.ndarray) -> np.ndarray:
    """
    Calculate the circumcenter of a set of points in barycentric coordinates.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumcenter of a set of points in barycentric coordinates.
    """
    points = np.asarray(points)
    num_rows, num_columns = points.shape
    A = np.bmat([[2 * np.dot(points, points.T),
                  np.ones((num_rows, 1))],
                 [np.ones((1, num_rows)), np.zeros((1, 1))]])
    b = np.hstack((np.sum(points * points, axis=1),
                   np.ones((1))))
    return np.linalg.solve(A, b)[:-1]

def circumradius(points: np.ndarray) -> float:
    """
    Calculte the circumradius of a given set of points.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumradius of a given set of points.
    """
    points = np.asarray(points)
    return np.linalg.norm(points[0, :] - np.dot(circumcenter(points), points))

def alphasimplices(points: np.ndarray) -> np.ndarray:
    """
    Returns an iterator of simplices and their circumradii of the given set of
    points.

    Args:
      points: An `N`x`M` array of points.

    Yields:
      A simplex, and its circumradius as a tuple.
    """
    coords = np.asarray(points)
    tri = Delaunay(coords)

    for simplex in tri.simplices:
        simplex_points = coords[simplex]
        try:
            yield simplex, circumradius(simplex_points)
        except np.linalg.LinAlgError:
            warnings.warn('Singular matrix. Likely caused by all points '
                          'lying in an N-1 space.')

def alphashape(points: np.ndarray,
               alpha: None|float = None):
    """
    Compute the alpha shape (concave hull) of a set of points.  If the number
    of points in the input is three or less, the convex hull is returned to the
    user.  For two points, the convex hull collapses to a `LineString`; for one
    point, a `Point`.

    Args:

      points (``np.ndarray``) : an iterable container of points
      alpha (float): alpha value

    Returns:

      ``np.ndarray`` : the resulting geometry
    """

    # If given a triangle for input, or an alpha value of zero or less,
    # return the convex hull.
    if len(points) < 4 or (alpha is not None and not callable(alpha) and alpha <= 0):
        result = ConvexHull(points)
        return result

    # Determine alpha parameter if one is not given
    if alpha is None:
        try:
            from optimizealpha import optimizealpha
        except ImportError:
            from .optimizealpha import optimizealpha
        alpha = optimizealpha(points)

    # Create a set to hold unique edges of simplices that pass the radius
    # filtering
    edges = set()

    # Create a set to hold unique edges of perimeter simplices.
    # Whenever a simplex is found that passes the radius filter, its edges
    # will be inspected to see if they already exist in the `edges` set.  If an
    # edge does not already exist there, it will be added to both the `edges`
    # set and the `permimeter_edges` set.  If it does already exist there, it
    # will be removed from the `perimeter_edges` set if found there.  This is
    # taking advantage of the property of perimeter edges that each edge can
    # only exist once.
    perimeter_edges = set()

    for point_indices, circumradius in alphasimplices(points):
        if callable(alpha):
            resolved_alpha = alpha(point_indices, circumradius)
        else:
            resolved_alpha = alpha

        # Radius filter
        if circumradius < 1.0 / resolved_alpha:
            for edge in itertools.combinations(
                    point_indices, r=points.shape[-1]):
                if all([e not in edges for e in itertools.combinations(
                        edge, r=len(edge))]):
                    edges.add(edge)
                    perimeter_edges.add(edge)
                else:
                    perimeter_edges -= set(itertools.combinations(
                        edge, r=len(edge)))

    
    # Create the resulting polygon from the edge points
    m = MultiLineString([points[np.array(edge)] for edge in perimeter_edges])
    triangles = list(polygonize(m))
    result = unary_union(triangles)

    return result

    
def prad(strike, dip, rake, toa, az):

    # convert all to arrays
    if isinstance(strike, float) or isinstance(strike, int):
        strike = np.array([strike])
        dip = np.array([dip])
        rake = np.array([rake])
    else:
        strike = np.array(strike)
        dip = np.array(dip)
        rake = np.array(rake)
    if isinstance(toa, float) or isinstance(toa, int):
        toa = np.array([toa])
        az = np.array([az])
    else:
        toa = np.array(toa)
        az = np.array(az)

    # # define size of output array
    # nev = strike.shape
    # nstation = toa
        
    # print(strike.shape)
    # print(toa.shape)

    # convert all to radians
    strike = np.deg2rad(strike)
    dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)
    toa = np.deg2rad(toa)
    az = np.deg2rad(az)

    # print(strike, type(strike))
    # print(az[:,np.newaxis]-strike[np.newaxis,:])

    part1 = np.cos(rake[np.newaxis,:]) * np.sin(dip[np.newaxis,:]) * np.sin(toa[:,np.newaxis])**2 * np.sin(2*(az[:,np.newaxis]-strike[np.newaxis,:]))
    part2 = np.cos(rake[np.newaxis,:]) * np.cos(dip[np.newaxis,:]) * np.sin(2*toa[:,np.newaxis]) * np.cos(az[:,np.newaxis]-strike[np.newaxis,:])
    part3 = np.sin(rake[np.newaxis,:]) * np.sin(2*dip[np.newaxis,:]) * (np.cos(toa[:,np.newaxis])**2 - np.sin(toa[:,np.newaxis])**2*np.sin(az[:,np.newaxis]-strike[np.newaxis,:])**2)
    part4 = np.sin(rake[np.newaxis,:]) * np.cos(2*dip[np.newaxis,:]) * np.sin(2*toa[:,np.newaxis]) * np.sin(az[:,np.newaxis]-strike[np.newaxis,:])

    return part1-part2+part3+part4


if __name__ == "__main__":
    ## this section sets the input data
    strike = np.array([0, 45])
    dip = np.array([90, 90])
    rake = np.array([0, 0])

    az = np.linspace(0, 360, 100)
    toa = np.zeros_like(az)+90

    ## outputs as nstations by nevents array
    rad = prad(strike,dip,rake,toa,az)
    print(rad.shape)

    az = np.linspace(0, 360, 40)
    toa = np.linspace(10, 150, 20)
    # az = np.random.random(40) * 360
    # toa = np.random.random(40)*180
    AZ, TOA = np.meshgrid(az, toa, indexing="ij")
    AZ = AZ.flatten()
    TOA = TOA.flatten()
    nstations = len(TOA)
    choice = np.random.choice(np.mgrid[:nstations], 15, replace=False)
    AZ = AZ[choice]
    TOA = TOA[choice]
    nstations = len(TOA)
    phases = ["P"]*nstations
    names = [f"{count:03d}" for count in range(nstations)]
    df = pd.DataFrame({"Station" : names, "Phase" : phases, "Azimuth" : AZ, "TakeOffAngle" : TOA})
    df.set_index(["Station", "Phase"], inplace=True)

    df["PolWeight"] = np.random.choice(range(3), len(df))
    mask = np.random.random(len(df)) < 0.4
    df.loc[mask, "PolWeight"] = 0 # set some polarity's to 0 weight (best)

    # calculate polarity for focal mechanism
    dc = DoubleCouple([np.random.random()*360, np.random.random()*90, (np.random.random()*360)-180])
    # dc = DoubleCouple([62, 42, 50])
    print(dc.plane1)
    print(dc.plane2)
    theoretical_pick = np.sign(prad(dc.plane1.strike, dc.plane1.dip, dc.plane1.rake, df.TakeOffAngle, df.Azimuth)).astype(int)
    df["ACTUAL_Polarity"] = theoretical_pick
    # add noise
    theoretical_pick = np.where(np.random.random(nstations)<0.1, theoretical_pick[:,0]*-1, theoretical_pick[:,0])
    df["Polarity"] = theoretical_pick

    ## 

    # calculate plunge/bearing for plotting 
    df["plunge"] = 90 - df.TakeOffAngle
    df["bearing"] = df.Azimuth
    df["upper_hemisphere"] = False
    # where the plunge is negative, correct
    mask = df.plunge<0
    df.loc[mask,"plunge"] = df.loc[mask,"plunge"].apply(np.abs)
    df.loc[mask,"bearing"] = (df.loc[mask,"bearing"] + 180)%360
    df.loc[mask,"upper_hemisphere"] = True

    # this section sets the weight based on the picks
    qual2weight = {0:0, 1:0.2, 2:0.4, 3:0.5, 4:1}
    df["weight"] = df.loc[:,"PolWeight"].apply(lambda key: qual2weight.get(key, None))
    mask1 = df["weight"] < 0.001
    mask2 = df["weight"] >= 0.5
    mask3 = ~mask1 & ~mask2
    df.loc[mask1, "weight"] = 29.6386
    df.loc[mask2, "weight"] = 0.
    df.loc[mask3, "weight"] = 1 / (df.loc[mask3,"weight"] - df.loc[mask3,"weight"]**2) - 2
    print("AVERAGE WEIGHT", (1./len(df)) * np.sum(df.weight), "30 = all best picks, 0 = rubbish") 

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="polar")
    # ax.plot(np.deg2rad(az), np.abs(prad(strike,dip,rake,toa,az)), "k+")
    # ax.set_theta_zero_location("N")
    # plt.show()

    # # plot station locations
    # import mplstereonet
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='stereonet')
    # ax.line(df.plunge, df.bearing, "ko")
    # ax.grid()



    # from mplstereonet import stereonet_math as mpl_math
    # # lines = mpl_math.line()
    # lon, lat = mpl_math.plane(45, 80)



    # # coarse search
    # crs_sol, crs_sol_range, crs_sol_misfit = fp_grid_search(df, output_range=True)

    # # fine search
    # width = 30
    # minstrike, maxstrike = crs_sol.plane1.strike-width, crs_sol.plane1.strike+width
    # mindip, maxdip = crs_sol.plane1.dip-width, crs_sol.plane1.dip+width
    # minrake, maxrake = crs_sol.plane1.rake-width, crs_sol.plane1.rake+width
    # print(minstrike, maxstrike)
    # print(mindip, maxdip)
    # print(minrake, maxrake)
    # fn_sol, fn_sol_range, fn_sol_misfit = fp_grid_search(df, dstrike=2, ddip=2, drake=5,
    #                                                     strikelim=(minstrike, maxstrike),
    #                                                     diplim=(mindip, maxdip),rakelim=(minrake, maxrake),
    #                                                     output_range=True)
    fn_sol, fn_sol_range, fn_sol_misfit = fp_grid_search(df, dstrike=5, ddip=5, drake=10,
                                                        output_range=True)

    # get the predicted radiation amplitude
    rad = prad(fn_sol.plane1.strike, fn_sol.plane1.dip, fn_sol.plane1.rake, 
            df.TakeOffAngle, df.Azimuth)
    # set the predicted polarities
    pred = np.sign(rad)
    isTrue = df.Polarity == pred[:,0]

    if np.sum(~isTrue) == 0:
        print("ALL ARE TRUE")
    else:
        print("FALSE STATIONS")
        print(df.Polarity[~isTrue])


    # # get the station distribution ratio for the best solution
    # stdr = np.sum(df.PolWeight * weight[:,indmin]) / np.sum(df.PolWeight)
    # print("STATION DISTRIUBUTION RATIO", stdr, ">0.5 = good, otherwise worth checking, many stations close to nodal planes")

    # quality parameters
    if fn_sol_misfit <= 0.025:
        print("QF = A")
    elif fn_sol_misfit <= 0.1:
        print("QF = B")
    else:
        print("QF = C")

    # plt.figure()
    # plt.plot(misfit, "k.", ms=1)
    # plt.plot(indmin, misfit[indmin], "go")
    # bestfit_fine_chosen = DoubleCouple([STRIKES[indmin], DIPS[indmin], RAKES[indmin]])



    # plot focal mechanism
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')
    mask = (df.Polarity > 0) & ~df.upper_hemisphere & (df.weight > 0)
    ax.line(df.plunge[mask], df.bearing[mask], "r^")
    mask = (df.Polarity > 0) & df.upper_hemisphere & (df.weight > 0)
    ax.line(df.plunge[mask], df.bearing[mask], "^", mec="r", mfc="white")

    mask = (df.Polarity < 0) & ~df.upper_hemisphere & (df.weight > 0)
    ax.line(df.plunge[mask], df.bearing[mask], "bv")
    mask = (df.Polarity < 0) & df.upper_hemisphere & (df.weight > 0)
    ax.line(df.plunge[mask], df.bearing[mask], "v", mec="b", mfc="white")

    mask = (df.Polarity == 0) | (df.weight == 0)
    ax.line(df.plunge[mask], df.bearing[mask], "o", mfc="white", mec="black")

    for index, row in df.iterrows():
        lon, lat = mplstereonet.stereonet_math.line(row.plunge, row.bearing)
        ax.text(lon, lat, index[0])

    ax.plane(dc.plane1.strike, dc.plane1.dip, "-", color="purple", lw=3, label="actual")
    ax.plane(dc.plane2.strike, dc.plane2.dip, "-", color="purple", lw=3)

    # ax.plane(crs_sol.plane1.strike, crs_sol.plane1.dip, "b-", lw=2, label="coarse")
    # ax.plane(crs_sol.plane2.strike, crs_sol.plane2.dip, "b-", lw=2)

    # if len(crs_sol_range) > 1:
    #     ax = plot_all_fault_planes(ax, fn_sol_range, color="red", linewidth=1)

    if len(fn_sol_range) > 1:   
        if len(fn_sol_range) < 10:
            ax = plot_all_fault_planes(ax, fn_sol_range, color="black", alpha=0.4)
        elif len(fn_sol_range) < 500:
            ax = plot_all_fault_planes(ax, fn_sol_range, color="black")
        else:
            ax = plot_fault_plane_histogram(ax, fn_sol_range)
        ax = plot_PT_errorbar(fn_sol_range, ax)
        ax = plot_PT_axes(fn_sol, ax)

    ax.plane(fn_sol.plane1.strike, fn_sol.plane1.dip, "r-", lw=2, label="fine")
    ax.plane(fn_sol.plane2.strike, fn_sol.plane2.dip, "r-", lw=2)

    # ax.grid()
    ax.legend()
    # fig.colorbar(hdl)
    plt.show()
    plt.close("all")


    # for bf in bestfit_coarse:
    #     ax.plane(bf.plane1.strike, bf.plane1.dip, "r-", lw=2, alpha=0.1)
    #     ax.plane(bf.plane2.strike, bf.plane2.dip, "r-", lw=2, alpha=0.1)

    #     ax.line(bf.axis["P"]["dip"], bf.axis["P"]["azimuth"], "rs", alpha=0.1)
    #     ax.line(bf.axis["T"]["dip"], bf.axis["T"]["azimuth"], "s", alpha=0.1, mec="r", mfc="white")



    #     # ax.line(bf.axis["P"]["dip"], bf.axis["P"]["azimuth"], "ys", alpha=0.1)
    #     # ax.line(bf.axis["T"]["dip"], bf.axis["T"]["azimuth"], "s", alpha=0.1, mec="y", mfc="white")
        
    # Plots a Polygon to pyplot `ax`



    # # fault planes
    # positions_x, positions_y = mplstereonet.stereonet_math.plane([bf.plane1.strike for bf in bestfit_fine], 
    #                                              [bf.plane1.dip for bf in bestfit_fine])
    # positions1 = np.vstack((positions_x.flatten(), positions_y.flatten())).T
    # positions_x, positions_y = mplstereonet.stereonet_math.plane([bf.plane2.strike for bf in bestfit_fine], 
    #                                              [bf.plane2.dip for bf in bestfit_fine])
    # positions2 = np.vstack((positions_x.flatten(), positions_y.flatten())).T
    # positions = np.vstack((positions1, positions2))
    # # # qhull_Paxis = ConvexHull(positions)
    # # alpha_Taxis = alphashape(positions[::10], 2)
    # # # ax.plot(positions[qhull_Paxis.vertices,0], positions[qhull_Paxis.vertices,1], 'r--', lw=2)
    # # if isinstance(alpha_Taxis, Polygon):
    # #     plot_polygon(ax, alpha_Taxis, alpha=0.5, fc="gray", ec="none")
    # # else:
    # #     for polygon in list(alpha_Taxis.geoms):
    # #         plot_polygon(ax, polygon, alpha=0.5, fc="gray", ec="none")
    # a, x, y = np.histogram2d(positions[:,0],positions[:,1], bins=50)
    # X, Y = np.meshgrid(x, y)

    # from mplstereonet.stereonet_transforms import LambertTransform, InvertedLambertTransform
    # tf = LambertTransform(0,0,60)
    # tfi = tf.inverted()
    # plt.figure()
    # a = tf.transform(positions)
    # plt.subplot(221)
    # plt.plot(a[:,0],a[:,1], "k+")
    # a, x, y = np.histogram2d(a[:,0],a[:,1], bins=50)
    # plt.subplot(222)
    # plt.pcolormesh(x, y, a.T, cmap="Greys", vmax=np.max(a)/2)
    # X, Y = np.meshgrid(x, y)
    # plt.subplot(2,2,3)
    # plt.plot(X, Y, "ko")
    # XY = tfi.transform(np.stack((X.flatten(),Y.flatten())).T)
    # X = np.reshape(XY[:,0], (51,51))
    # Y = np.reshape(XY[:,1], (51,51))
    # plt.subplot(2,2,4)
    # plt.plot(X.T, Y.T, "ko")
    # plt.show()
    # plt.figure()
    # plt.pcolormesh(x, y, a.T, cmap="Greys", vmax=np.max(a)/2)
    # plt.show()

