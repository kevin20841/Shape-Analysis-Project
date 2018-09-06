import numpy as np
from numpy import exp, cos, sin, pi, sqrt, arctan, array


def curve_example(name, t, param1=None, param2=None, param3=None):
    """Generates the specified example curve.

    Generates a 2xT array of numbers that give the coordinates of
    the nodes of the curve (and optionally its derivative w.r.t. its
    parameterization t) specified by 'name' and param1,param2,param3.
    The input arguments 'name' and 't' need to be specified, but the
    curve-specific parameters are optional. 't' is either an integer
    specifying the number of nodes on the curve or it is an array
    of increasing real numbers starting at 0 ending at 1 representing
    a partitioning of the interval [0,1]. Possible choices for 'name'
    and the corresponding default parameter values are the following:

      'circle'  =>  radius=1,
      'ellipse'  =>  a=1, b=2,
      'superellipse'  =>  a=1, b=2, p=6,
      'limacon'  =>  r=1.25,
      'hippopede'  =>  a1=1.25, a2=1,
      'epitrochoid'  =>  r1=1, r2=3, d=0.65,
      'flower'  =>  amplitude=0.3, frequency=6,
      'bumps'  =>  locations=[0.6 0.3], magnitude=[0.5 0.3].

    For example, the call to obtain a superellipse with 128 nodes and
    its derivative with parameters a=1.5, b=3, p=4 would be as follows:

       curve,curve_deriv = curve_example('superellipse', 128, 1.5, 3, 4)

    Parameters
    ----------
    name: string
       The name of the curve, one of the values listed above.
    t: int or NumPy array
       Specifies the parameterization domain. It is either the integer
       denoting t equispaced points: 0,h,2h,3h,...,1-h,1. Or it is a
       one-dimensional array containing the points: 0,t1,t2,...,1.
    param1: double, optional
       The first of the curve definition parameters.
    param2: double, optional
       The second of the curve definition parameters.
    param3: double, optional
       The third of the curve definition parameters.

    Returns
    -------
    curve: NumPy array
        An array of shape (2,len(t)) storing the (x,y) coords of the curve
        nodes.
    curve_deriv:
        An array of shape (2,len(t)) storing the (x,y) components of the
        curve derivative.

    Raises
    ------
    ValueError
        If name is not one of the defined curves.
    """

    if np.isscalar(t): # then t is the number of nodes
        t = np.linspace( 0.0, 1.0, t ) # now t is the array of nodes [0 h 2h .... 1]

    n = len(t)

    if name == 'circle':
        r = param1 if param1 is not None else 1.0
        C = cos(2*pi*t);   S = sin(2*pi*t)
        b = array([ r*C, r*S ])
        bdot = array([ -2*pi*r*S, 2*pi*r*C ])

    elif name == 'ellipse':
        r1 = param1 if param1 is not None else 1.0
        r2 = param2 if param2 is not None else 2.0
        C = cos(2*pi*t);   S = sin(2*pi*t)
        b = array([ r1*C, r2*S ])
        bdot = array([ -2*pi*r1*S, 2*pi*r2*C ])

    elif name == 'superellipse':
        r1 = param1 if param1 is not None else 1.0
        r2 = param2 if param2 is not None else 2.0
        p  = param3 if param3 is not None else 6.0
        if np.mod(p,2) != 0:
            raise ValueError('The power parameter p should be an even number!')
        C = cos(2*pi*t);   S = sin(2*pi*t);   CSp = (C**p + S**p)
        d = CSp**(1.0/p)
        b = array([ r1*C/d, r2*S/d ])
        ddot = 2*pi * CSp**(1.0/p-1) * (-S*C**(p-1) + C*S**(p-1))
        bdot = array([-2*pi*r1 * S/d - r1 * C/d**2*ddot,
                       2*pi*r2 * C/d - r2 * S/d**2*ddot ])

    elif name == 'limacon':
        r0 = param1 if param1 is not None else 1.25
        C = cos(2*pi*t);   S = sin(2*pi*t)
        r = r0 + C
        b = array([ r*C, r*S ])
        rdot = -2*pi*S
        bdot = array([ rdot*C - 2*pi*r*S,  rdot*S + 2*pi*r*C ])

    elif name == 'hippopede':
        a1 = param1 if param1 is not None else 1.25
        a2 = param2 if param2 is not None else 1.0
        if (np.sign(a1) != np.sign(a2)) or (abs(a1) < abs(a2)):
            raise ValueError('Need sign(a1) == sign(a2) and abs(a1) > abs(a2) for hippopede!')
        C = cos(2*pi*t);   S = sin(2*pi*t)
        r = sqrt(4*a2*(a1 - a2*S**2))
        b = array([ r*S, r*C ])
        rdot = -8*pi*a2**2 * S * C / r
        bdot = array([ rdot*S + 2*pi*r*C, rdot*C - 2*pi*r*S ])

    elif name == 'epitrochoid':
        r1 = param1 if param1 is not None else 1.0
        r2 = param2 if param2 is not None else 3.0
        d  = param3 if param3 is not None else 0.65
        C = cos(2*pi*t);   S = sin(2*pi*t)
        C2 = cos(2*pi*t*(r1+r2)/r1)
        S2 = sin(2*pi*t*(r1+r2)/r1)
        b = array([ (r1+r2)*C - d*C2, (r1+r2)*S - d*S2 ])
        bdot = array([-2*pi*(r1+r2)*S + 2*pi*d*(r1+r2)/r1*S2,
                       2*pi*(r1+r2)*C - 2*pi*d*(r1+r2)/r1*C2 ])

    elif name == 'flower':
        amp  = param1 if param1 is not None else 0.3
        freq = param2 if param2 is not None else 6.0
        r = 1 + amp*sin(2*pi*freq*t)
        C = cos(2*pi*t);   S = sin(2*pi*t)
        b = array([ r*C, r*S ])
        rdot = 2*pi*amp*freq*cos(2*pi*freq*t)
        bdot = array([ rdot * C - 2*pi*r * S,
                       rdot * S + 2*pi*r * C ])

    elif name == 'bumps':
        locations  = param1 if param1 is not None else [0.6, 0.3]
        magnitudes = param2 if param2 is not None else [0.5, 0.3]
        if len(locations) != len(magnitudes):
            raise ValueError("For curve 'bumps', number of locations (param1) and number of magnitudes (param2 should be the same!")

        r = np.ones(n)
        rdot = np.zeros(n)
        for t0,C in zip(locations,magnitudes):
            r = r + C * exp(-1000.0*(t-t0)**2)
            rdot = rdot - 2000 * C * (t-t0) * exp(-1000*(t-t0)**2)
        C = cos(2*pi*t);   S = sin(2*pi*t)
        b = array([ r*C, r*S ])
        bdot = array([ rdot*C - 2*pi*r*S, rdot*S + 2*pi*r*C ])

    else:
        raise ValueError('Unknown curve name!')

    center = np.mean(b,1);
    curve_length = np.sum( np.sqrt( np.sum((b[:,1:] - b[:,:-1])**2, 0) ) );

    b[0,:] = (b[0,:] - center[0]) / curve_length;
    b[1,:] = (b[1,:] - center[1]) / curve_length;
    bdot = bdot / curve_length;

    return (b,bdot)


def gamma_example(name, param1=None, param2=None):
    """Generates an example reparameterization function gamma(t).

    This function returns an example gamma function GAMMA:[0,1]->[0,1]
    (and optionally its derivative GAMMA_DERIV) such that
        GAMMA(0) = 0,  GAMMA(1) = 1,
        GAMMA'(0) = GAMMA'(1).

    The argument NAME specifies which gamma is returned.
    Optional parameters PARAM1, PARAM2 can be provided to control
    the behavior of the function gamma. The following are the names,
    default parameters and function expressions for possible choices:

        'identity'  =>  gamma(s) = s
        'polynomial', a=0.1  =>  gamma(s) = s + 16*a*s.^2.*(s-1).^2
        'sine', a=0.025, w=2  =>  gamma(s) = s + a*sin(2*w*pi*s)
        'flat', p=5  =>  gamma(s) = 0.5 + 0.5*(2*s-1).^p
        'steep', a=0.25, w=40  =>  gamma(s) = a*atan(w*pi*(s-0.5))
        'bumpy'  =>  gamma(s) = (1/2.4) * (0.7 + 0.5*(1 + (2*s-1).^5) +
                                           0.25*atan(40*pi*(s-0.75)) +
                                           0.20*atan(40*pi*(s-0.4)))
        'custom', f = @(s) 4*exp(s), df = @(s) 4*exp(s),
                    =>  gamma(s) = f(s) + 0.5*(df0-df1)*s.^2 +
                                   (1+f0-f1-0.5*(df0-df1))*s - f(0)

    For example, the function call to obtain a sine function with
    the parameters a=0.37, w=4 would be as follows:

       gamma,gamma_deriv = gamma_example('sine',0.37,4);

    Parameters
    ----------
    name: string
       The name of the gamma function, one of the values listed above.
    param1: double or lambda function, optional
       The first of the gamma definition parameters.
    param2: double or lambda function, optional
       The second of the gamma definition parameters.

    Returns
    -------
    gamma: lambda function
    gamma_deriv: lambda function

    Raises
    ------
    ValueError
        If name is not one of the defined gamma functions.
    """

    if name == 'identity':
        gamma  = lambda s: s
        dgamma = lambda s: np.ones(len(s))

    elif name == "flat+steep":
        gamma = flat_steep
        dgamma = lambda s: 1
    elif name == 'polynomial':
        a = param1 if param1 is not None else 0.1
        gamma  = lambda s: s + 16*a * s**2 * (s-1)**2
        dgamma = lambda s: 1 + 32*a * s * (s-1) * (2*s-1)

    elif name == 'sine':
        a = param1 if param1 is not None else 0.025
        w = param2 if param2 is not None else 2.0
        gamma  = lambda s: s + a*sin(2*w*pi*s)
        dgamma = lambda s: 1 + 2*a*w*pi*cos(2*w*pi*s)

    elif name == 'flat':
        p = param1 if param1 is not None else 5.0
        gamma  = lambda s: 0.5 + 0.5*(2*s-1)**p
        dgamma = lambda s: (2*s-1)**(p-1)

    elif name == 'steep':
        a = param1 if param1 is not None else 0.25
        w = param2 if param2 is not None else 40.0
        f  = lambda s: -s + a * arctan( w*pi*(s-0.5) )
        df = lambda s: -1 + a * w*pi / (1 + (w*pi*(s-0.5))**2)
        [gamma,dgamma] = gamma_example('custom',f,df)

    elif name == 'bumpy':
        # f = lambda s: -s + (1 / 2.4) * (1.2 + 0.5 * (2 * s - 1). ^ 5 + 0.25 * atan(40 * pi * (s - 0.75))
        #                   + 0.2 * atan(40 * pi * (s - 0.4)))
        f = lambda s: -s + (1 / 2.4) * (1.2 + 0.5 * (2 * s - 1) ** 5 + 0.25 * arctan(40 * pi * (s - 0.75))
                          + 0.2 * arctan(40 * pi * (s - 0.4)))
        df = lambda s: -1 + (1 / 2.4) * (5.0 * (2 * s - 1)**4 + 0.25 * 40 * pi / (1 +
                                                                      (40 * pi * (s - 0.75))** 2) + 0.2 * 40 * pi / (
                                      1 + (40 * pi * (s - 0.4))**2))
        [gamma,dgamma] = gamma_example('custom',f,df)
    elif name == 'custom':
        if param1 is None or param2 is None:
            f = lambda s: 4*exp(s);  df = f
        else:
            f = param1;  df = param2

        f0=f(0); f1=f(1);  df0=df(0);  df1=df(1)
        gamma  = lambda s: f(s) + 0.5*(df0-df1)*s**2 + (1+f0-f1-0.5*(df0-df1))*s - f(0)
        dgamma = lambda s: df(s) + (df0-df1)*s + (1+f0-f1-0.5*(df0-df1))

    else:
        raise ValueError('Unknown gamma name!')

    return gamma, dgamma


def flat_steep(s):
    x = np.zeros(s.size)
    for i in range(s.size):
        if s[i] <= 3/10:
            x[i] = s[i] / 3
        elif s[i] <= 7/20:
            x[i] = (s[i]-3/10) * 7 + 1/10
        elif s[i] <= 13/20:
            x[i] = (s[i] - 7/20) / 3 + 9 / 20
        elif s[i] <= 7/10:
            x[i] = (s[i] - 13/20) * 7 + 11/20
        elif s[i] <= 1:
            x[i] = (s[i]-7/10) / 3 + 9/10
    return x