import numpy as np

def MT33_MT6(MT33):
    """
    Convert a 3x3 array to six vector maintaining normalisation. 6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    Args
        M33: 3x3 numpy array

    Returns
        numpy.array: MT 6-vector

    """
    MT6 = np.array([[MT33[0, 0]], [MT33[1, 1]], [MT33[2, 2]], [np.sqrt(2)*MT33[0, 1]],
                     [np.sqrt(2)*MT33[0, 2]], [np.sqrt(2)*MT33[1, 2]]])
    MT6 = np.array(MT6/np.sqrt(np.sum(np.multiply(MT6, MT6), axis=0)))
    return MT6


def MT6_MT33(MT6):
    """
    Convert a six vector to a 3x3 MT maintaining normalisation. 6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    Args
        MT6: numpy array Moment tensor 6-vector

    Returns
        numpy.array: 3x3 Moment Tensor
    """
    if np.prod(MT6.shape) != 6:
        raise ValueError("Input MT must be 6 vector not {}".format(MT6.shape))
    if len(MT6.shape) == 1:
        MT6 = np.array(MT6)
    if len(MT6.shape) == 2 and MT6.shape[1] == 6:
        MT6 = MT6.T
    return np.array([[MT6[0, 0], (1/np.sqrt(2))*MT6[3, 0], (1/np.sqrt(2))*MT6[4, 0]],
                      [(1/np.sqrt(2))*MT6[3, 0], MT6[1, 0],
                       (1/np.sqrt(2))*MT6[5, 0]],
                      [(1/np.sqrt(2))*MT6[4, 0], (1/np.sqrt(2))*MT6[5, 0], MT6[2, 0]]])

def MT6_TNPE(MT6):
    """
    Convert the 6xn Moment Tensor to the T,N,P vectors and the eigenvalues.

    Args
        MT6: 6xn numpy array

    Returns
        (numpy.array, numpy.array, numpy.array, numpy.array): tuple of T, N, P
                        vectors and Eigenvalue array
    """

    try:
        n = MT6.shape[1]
    except Exception:
        MT6 = np.array([MT6]).T
        n = MT6.shape[1]
    T = np.array(np.empty((3, n)))
    N = np.array(np.empty((3, n)))
    P = np.array(np.empty((3, n)))
    E = np.empty((3, n))
    for i in range(n):
        T[:, i], N[:, i], P[:, i], E[:, i] = MT33_TNPE(MT6_MT33(MT6[:, i]))
    return T, N, P, E

def MT33_TNPE(MT33):
    """
    Convert the 3x3 Moment Tensor to the T,N,P vectors and the eigenvalues.

    Args
        MT33: 3x3 numpy array

    Returns
        (numpy.array, numpy.array, numpy.array, numpy.array): tuple of T, N, P
                        vectors and Eigenvalue array
    """
    E, L = np.linalg.eig(MT33)
    idx = E.argsort()[::-1]
    E = E[idx]
    L = L[:, idx]
    T = L[:, 0]
    P = L[:, 2]
    N = L[:, 1]
    return (T, N, P, E)

def MT33_SDR(MT33):
    """
    Convert the 3x3 Moment Tensor to the strike, dip and rake.

    Args
        MT33: 3x3 numpy array

    Returns
        (float, float, float): tuple of strike, dip, rake angles in radians
    """
    T, N, P, E = MT33_TNPE(MT33)
    N1, N2 = TP_FP(T, P)
    return FP_SDR(N1, N2)

def TNP_SDR(T, N, P):
    """
    Convert the T,N,P vectors to the strike, dip and rake in radians

    Args
        T: numpy array of T vectors.
        N: numpy array of N vectors.
        P: numpy array of P vectors.

    Returns
        (float, float, float): tuple of strike, dip and rake angles of fault plane in radians

    """

    N1, N2 = TP_FP(T, P)
    return FP_SDR(N1, N2)


def TP_FP(T, P):
    """
    Convert the 3x3 Moment Tensor to the fault normal and slip vectors.

    Args
        T: numpy array of T vectors.
        P: numpy array of P vectors.

    Returns
        (numpy.array, numpy.array): tuple of Normal and slip vectors
    """
    if T.ndim == 1:
        T = np.array(T)
    if P.ndim == 1:
        P = np.array(P)
    if T.shape[0] != 3:
        T = T.T
    if P.shape[0] != 3:
        P = P.T
    TP1 = T+P
    TP2 = T-P
    N1 = (TP1)/np.sqrt(np.einsum('ij,ij->j', TP1, TP1))
    N2 = (TP2)/np.sqrt(np.einsum('ij,ij->j', TP2, TP2))
    return (N1, N2)


def FP_SDR(normal, slip):
    """
    Convert fault normal and slip to strike, dip and rake

    Coordinate system is North East Down.

    Args
        normal: numpy array - Normal vector
        slip: numpy array - Slip vector


    Returns
        (float, float, float): tuple of strike, dip and rake angles in radians

    """
    if not isinstance(slip, np.ndarray):
        slip = slip/np.sqrt(np.sum(slip*slip, axis=0))
    else:
        # Do we need to replace this with einsum
        slip = slip/np.sqrt(np.einsum('ij,ij->j', slip, slip))
    if not isinstance(normal, np.ndarray):
        normal = normal/np.sqrt(np.sum(normal*normal, axis=0))
    else:
        normal = normal/np.sqrt(np.einsum('ij,ij->j', normal, normal))
    slip[:, np.array(normal[2, :] > 0).flatten()] *= -1
    normal[:, np.array(normal[2, :] > 0).flatten()] *= -1
    normal = np.array(normal)
    slip = np.array(slip)
    print(normal.shape, slip.shape)
    strike, dip = normal_SD(normal)
    rake = np.arctan2(-slip[2], slip[0]*normal[1]-slip[1]*normal[0])
    print(strike.shape)
    print(dip.shape)
    print(rake.shape)
    strike[dip > np.pi/2] += np.pi
    rake[dip > np.pi/2] = 2*np.pi-rake[dip > np.pi/2]
    dip[dip > np.pi/2] = np.pi-dip[dip > np.pi/2]
    strike = np.mod(strike, 2*np.pi)
    rake[rake > np.pi] -= 2*np.pi
    rake[rake < -np.pi] += 2*np.pi
    return (np.array(strike).flatten(), np.array(dip).flatten(), np.array(rake).flatten())

def SDR_TNP(strike, dip, rake):
    """
    Convert strike, dip  rake to TNP vectors

    Coordinate system is North East Down.

    Args
        strike: float radians
        dip: float radians
        rake: float radians

    Returns
        (numpy.array, numpy.array, numpy.array): tuple of T,N,P vectors.

    """
    strike = np.array(strike).flatten()
    dip = np.array(dip).flatten()
    rake = np.array(rake).flatten()
    N1 = np.array([(np.cos(strike)*np.cos(rake))+(np.sin(strike)*np.cos(dip)*np.sin(rake)),
                    (np.sin(strike)*np.cos(rake)) -
                    np.cos(strike)*np.cos(dip)*np.sin(rake),
                    -np.sin(dip)*np.sin(rake)])
    N2 = np.array([-np.sin(strike)*np.sin(dip), np.cos(strike)*np.sin(dip), -np.cos(dip)])
    return FP_TNP(N1, N2)


def SDR_SDR(strike, dip, rake):
    """
    Convert strike, dip  rake to strike, dip  rake for other fault plane

    Coordinate system is North East Down.

    Args
        strike: float radians
        dip: float radians
        rake: float radians

    Returns
        (float, float, float): tuple of strike, dip and rake angles of alternate fault
                        plane in radians

    """
    # Handle multiple inputs
    N1, N2 = SDR_FP(strike, dip, rake)
    print(type(N1), type(N2))
    s1, d1, r1 = FP_SDR(N1, N2)
    s2, d2, r2 = FP_SDR(N2, N1)
    # This should be ok to return s2,d2,r2 but doesn't seem to work
    try:
        r2[np.abs(strike-s2) < 1] = r1[np.abs(strike-s2) < 1]
        d2[np.abs(strike-s2) < 1] = d1[np.abs(strike-s2) < 1]
        s2[np.abs(strike-s2) < 1] = s1[np.abs(strike-s2) < 1]
        return s2, d2, r2
    except Exception:
        if np.abs(strike-s1) < 1:
            return (s2, d2, r2)
        else:
            return (s1, d1, r1)


def FP_TNP(normal, slip):
    """
    Convert fault normal and slip to TNP axes

    Coordinate system is North East Down.

    Args
        normal: numpy array - normal vector
        slip: numpy array - slip vector

    Returns
        (numpy.array, numpy.array, numpy.array): tuple of T, N, P vectors
    """
    T = (normal+slip)
    T = T/np.sqrt(np.einsum('ij,ij->j', T, T))
    P = (normal-slip)
    P = P/np.sqrt(np.einsum('ij,ij->j', P, P))
    N = np.array(-np.cross(T.T, P.T)).T
    return (T, N, P)


def SDSD_FP(strike1, dip1, strike2, dip2):
    """
    Convert strike and dip pairs to fault normal and slip

    Converts the strike and dip pairs in radians to the fault normal and slip.

    Args
        strike1: float strike angle of fault plane 1 in radians
        dip1: float dip angle of fault plane 1 in radians
        strike2: float strike angle of fault plane 2 in radians
        dip2: float dip  of fault plane 2 in radians

    Returns
        (numpy.array, numpy.array): tuple of Normal and slip vectors
    """
    strike1 = np.array(strike1).flatten()
    dip1 = np.array(dip1).flatten()
    strike2 = np.array(strike2).flatten()
    dip2 = np.array(dip2).flatten()
    N1 = np.array([-np.sin(strike2)*np.sin(dip2),
                    np.cos(strike2)*np.sin(dip2),
                    -np.cos(dip2)])
    N2 = np.array([-np.sin(strike1)*np.sin(dip1),
                    np.cos(strike1)*np.sin(dip1),
                    -np.cos(dip1)])
    return (N1, N2)


def SDR_FP(strike, dip, rake):
    """
    Convert the strike, dip  and rake in radians to the fault normal and slip.

    Args
        strike: float strike angle of fault plane  in radians
        dip: float dip angle of fault plane  in radians
        rake: float rake angle of fault plane  in radians

    Returns
        (numpy.array, numpy.array): tuple of Normal and slip vectors
    """
    T, N, P = SDR_TNP(strike, dip, rake)
    return TP_FP(T, P)


def SDR_SDSD(strike, dip, rake):
    """
    Convert the strike, dip  and rake to the strike and dip pairs (all angles in radians).

    Args
        strike: float strike angle of fault plane  in radians
        dip: float dip angle of fault plane  in radians
        rake: float rake angle of fault plane  in radians

    Returns
        (float, float, float, float): tuple of strike1, dip1, strike2, dip2 angles in radians
    """
    N1, N2 = SDR_FP(strike, dip, rake)
    return FP_SDSD(N1, N2)


def FP_SDSD(N1, N2):
    """
    Convert the the fault normal and slip vectors to the strike and dip pairs
    (all angles in radians).

    Args
        Normal: numpy array - Normal vector
        Slip: numpy array - Slip vector

    Returns
        (float, float, float, float): tuple of strike1, dip1, strike2, dip2 angles
                        in radians
    """
    s1, d1 = normal_SD(N1)
    s2, d2 = normal_SD(N2)
    return (s1, d1, s2, d2)

def normal_SD(normal):
    """
    Convert a plane normal to strike and dip

    Coordinate system is North East Down.

    Args
        normal: numpy array - Normal vector


    Returns
        (float, float): tuple of strike and dip angles in radians
    """
    if not isinstance(normal, np.ndarray):
        normal = np.array(normal)/np.sqrt(np.sum(normal*normal, axis=0))
    else:
        normal = normal/np.sqrt(np.diag(normal.T*normal))
    normal[:, np.array(normal[2, :] > 0).flatten()] *= -1
    normal = np.array(normal)
    strike = np.arctan2(-normal[0], normal[1])
    dip = np.arctan2((normal[1]**2+normal[0]**2),
                     np.sqrt((normal[0]*normal[2])**2+(normal[1]*normal[2])**2))
    strike = np.mod(strike, 2*np.pi)
    return strike, dip


def toa_vec(azimuth, plunge, radians=False):
    """
    Convert the azimuth and plunge of a vector to a cartesian description of the vector

    Args
        azimuth: float, vector azimuth
        plunge: float, vector plunge

    Keyword Arguments
        radians: boolean, flag to use radians [default = False]

    Returns
        np.array: vector
    """
    if not radians:
        azimuth = np.pi*np.array(azimuth)/180.
        plunge = np.pi*np.array(plunge)/180.
    if not isinstance(plunge, np.ndarray):
        plunge = np.array([plunge])
    try:
        return np.array([np.cos(azimuth)*np.sin(plunge),
                          np.sin(azimuth)*np.sin(plunge),
                          np.cos(plunge)])
    except Exception:
        return np.array([np.cos(azimuth)*np.sin(plunge),
                         np.sin(azimuth)*np.sin(plunge),
                         np.cos(plunge)])

