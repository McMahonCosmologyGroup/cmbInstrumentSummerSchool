import numpy as np

def getData1D(dataParams,team):
    """
    Get data from file.
    """
    cut = "H"
    FILE_NAME ="Data/"+team+"/"+str(dataParams.frequency)+"GHz_"+str(dataParams.sweep_ang)+"deg_1D_"+cut+".txt"

    data=np.loadtxt(FILE_NAME)
    ang=(len(data)-1) /2
    ang=np.linspace(-ang,ang,len(data))
    L_MEAN = 1
    N_INDIV = 7
    L=len(data[0,:])

    line_size = np.size(data[0])
    nsamp =  np.size(data,0)
    arr_f = np.zeros(nsamp)
    arr_x = np.zeros(nsamp)
    arr_y = np.zeros(nsamp)
    arr_phi = np.zeros(nsamp)
    amp_cross=np.zeros(nsamp)
    amp_phase=np.zeros(nsamp)
    amp_var=np.zeros(nsamp)
    phase=np.zeros(nsamp)

    i_AA_begin = int(N_INDIV + (1-1)*(line_size-N_INDIV)/4)
    i_AA_end= int(N_INDIV + (2-1)*(line_size-N_INDIV)/4) -1
    i_BB_begin = int(N_INDIV + (2-1)*(line_size-N_INDIV)/4)
    i_BB_end= int(N_INDIV + (3-1)*(line_size-N_INDIV)/4) -1
    i_AB_begin = int(N_INDIV + (3-1)*(line_size-N_INDIV)/4)
    i_AB_end= int(N_INDIV + (4-1)*(line_size-N_INDIV)/4) -1
    i_phase_begin = int(N_INDIV + (4-1)*(line_size-N_INDIV)/4)
    i_phase_end= int(N_INDIV + (5-1)*(line_size-N_INDIV)/4) -1

    arr_f = data[:,0]
    arr_x = data[:,1]
    arr_y = data[:,2]
    arr_phi = data[:,3]
    index_signal = data[:,4]

    for kk in range(nsamp):
        #take in raw DATA
        arr_AA = np.array(running_mean(data[kk][i_AA_begin : i_AA_end],L_MEAN))
        arr_BB = np.array(running_mean(data[kk][i_BB_begin : i_BB_end],L_MEAN))
        arr_AB = np.array(running_mean(data[kk][i_AB_begin : i_AB_end],L_MEAN))
        arr_phase = np.array( data[kk][i_phase_begin : i_phase_end] )
        n_channels = np.size(arr_AB)

        #make amplitude arrays, in case they need to be plotted.
        amp_cross[kk] = arr_AB[int(n_channels/2)]
        amp_var[kk] = np.power( np.divide(arr_AB[int(n_channels/2)],arr_AA[int(n_channels/2)]) , 2)
        amp_phase[kk] = np.remainder(arr_phase[int(n_channels/2)],360.)

    theta_arr = arr_x if arr_x[0] != 0 else arr_y
    A = amp_var/np.max(amp_var)
    beam_complex = A * np.exp(amp_phase * np.pi / 180. * complex(0, 1))

    return theta_arr,beam_complex

def getData2D(txt_file):
    """
    Get data from file.
    """
    DATA_1 = np.loadtxt(txt_file, skiprows=1)
    DATA = []
    RoachOpt.L_MEAN = 1
    N_INDIV = 7

    line_size = np.size(DATA_1[0])
    nsamp = np.size(DATA_1, 0)
    arr_x = np.zeros(nsamp)
    arr_y = np.zeros(nsamp)
    arr_phi = np.zeros(nsamp)
    amp_cross = np.zeros(nsamp)
    amp_AA = np.zeros(nsamp)
    amp_BB = np.zeros(nsamp)
    amp_var = np.zeros(nsamp)
    phase = np.zeros(nsamp)

    i_AA_begin = int(N_INDIV + (1-1)*(line_size-N_INDIV)/4)
    i_AA_end = int(N_INDIV + (2-1)*(line_size-N_INDIV)/4) - 1
    i_BB_begin = int(N_INDIV + (2-1)*(line_size-N_INDIV)/4)
    i_BB_end = int(N_INDIV + (3-1)*(line_size-N_INDIV)/4) - 1
    i_AB_begin = int(N_INDIV + (3-1)*(line_size-N_INDIV)/4)
    i_AB_end = int(N_INDIV + (4-1)*(line_size-N_INDIV)/4) - 1
    i_phase_begin = int(N_INDIV + (4-1)*(line_size-N_INDIV)/4)
    i_phase_end = int(N_INDIV + (5-1)*(line_size-N_INDIV)/4) - 1

    i = int(0)

    jj = 1
    while (jj <= 1):
        i = int(0)
        if jj == 1:
            str_data = 'Dataset 1'
            DATA = DATA_1
        else:
            DATA = DATA_2
            str_data = 'Dataset 2'
        while (i < (nsamp)):
            # take in raw DATA
            arr_x[i] = DATA[i][1]
            arr_y[i] = DATA[i][2]
            arr_phi[i] = DATA[i][3]
            # use same index singal for both datasets. keep it simple for now.
            index_signal = DATA[i][4]
            arr_AA = np.array(running_mean(
                DATA[i][i_AA_begin: i_AA_end], RoachOpt.L_MEAN))
            arr_BB = np.array(running_mean(
                DATA[i][i_BB_begin: i_BB_end], RoachOpt.L_MEAN))
            arr_AB = np.array(running_mean(
                DATA[i][i_AB_begin: i_AB_end], RoachOpt.L_MEAN))
            arr_phase = np.array(DATA[i][i_phase_begin: i_phase_end])
            n_channels = np.size(arr_AB)

            # make amplitude arrays, in case they need to be plotted.
            amp_cross[i] = np.power(arr_AB[int(n_channels/2)], 1)
            amp_var[i] = np.power(
                np.divide(arr_AB[int(n_channels/2)], arr_AA[int(n_channels/2)]), 2)
            amp_AA[i] = arr_AA[int(n_channels/2)]
            amp_BB[i] = arr_BB[int(n_channels/2)]
            phase[i] = np.remainder(arr_phase[int(n_channels/2)], 360.)
            #print('phase[i] = '+str(phase[i]))
            i = i+1


        amp = amp_var
        amp = np.divide(amp, np.max(amp))
        arr_x = np.unique(arr_x)
        arr_y = np.unique(arr_y)
        X, Y = np.meshgrid(arr_x, arr_y)
        P = amp_cross.reshape(len(arr_x), len(arr_y))
        Z = phase.reshape(len(arr_x), len(arr_y))

        jj = jj + 1

    beam_complex = P * np.exp(Z * np.pi / 180. * complex(0, 1))
    return X,Y,Z,abs(beam_complex)/np.max(abs(beam_complex))

def running_mean(x, N):
    """

    Calculates running mean of 1D array.

    """
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]

class RoachOpt:

    """

    ROACH2 configuration settings.

    """

    BITSTREAM = "t4_roach2_noquant_fftsat.fpg"

    f_clock_MHz = 500
    f_max_MHz = f_clock_MHz / 4
    N_CHANNELS = 21
    N_TO_AVG = 1
    L_MEAN = 1
    katcp_port = 7147

    ip = "192.168.4.20"

def coords_ang_to_spat(theta,freq):
    """
    Coordinate transformation from angular to spatial.
    """
    ff_ghz = freq * 1e9
    # Get spatial coordinates
    lam = (3 * 10 ** 8) / ff_ghz
    delta_th = abs(np.max(theta) - np.min(theta)) / (
        len(theta) - 1
    )  # increment in azimuthal angle

    x_len = len(theta)

    alpha = lam / delta_th  # increment in x
    delta_x = alpha / x_len  # spatial coordinates conversion

    x_spat = (
        np.linspace(
            -int((len(theta) / 2)), int((len(theta) / 2)), int((len(theta)))
        )
        * delta_x
        * 1e2
    )

    return x_spat