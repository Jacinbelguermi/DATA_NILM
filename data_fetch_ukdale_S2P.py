import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdf5plugin
import h5py


def get_data(
    filename="/content/drive/MyDrive/progressive disag/ukDale/ukdale.h5",
    sampling_period='60S',
    chosen_apps=[3,5,9,14,16,18,19,20,22,23,24],
    main_meter=26,
    segment_length=100,
    overlap=10,
    train_ratio=0.75,
    normalize=False,
    thresholds=None
):

    # =====================================================
    # APPLIANCE IDS
    # =====================================================
    house5_appp_id = {

        3: 'i7_desktop',
        4: 'hairdryer',
        5: 'primary_tv',
        6: '24_inch_lcd_bedroom',
        7: 'treadmill',
        8: 'network_attached_storage',
        9: 'core2_server',
        10: '24_inch_lcd',
        12: 'steam_iron',
        13: 'nespresso_pixie',
        14: 'atom_pc',
        15: 'toaster',
        16: 'home_theatre_amp',
        17: 'sky_hd_box',
        18: 'kettle',
        19: 'fridge_freezer',
        20: 'oven',
        21: 'electric_hob',
        22: 'dishwasher',
        23: 'microwave',
        24: 'washer_dryer',
    }

    # =====================================================
    # LOAD MAIN METER
    # =====================================================
    main_meter_df = pd.read_hdf(
        filename,
        key="building5/elec/meter" + str(main_meter)
    )

    data = main_meter_df["power"].astype(float)

    data = data.resample(sampling_period).mean()

    merged_df = data.copy()

    merged_df = merged_df.sort_index()

    # =====================================================
    # LOAD APPLIANCES
    # =====================================================
    for app in chosen_apps:

        appliance_df = pd.read_hdf(
            filename,
            key="building5/elec/meter" + str(app)
        )

        appliance_power = appliance_df["power"].astype(float)

        appliance_power = appliance_power.resample(
            sampling_period
        ).mean()

        appliance_power = appliance_power.sort_index()

        merged_df = pd.merge_asof(
            merged_df,
            appliance_power,
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(seconds=4),
            direction='forward',
            suffixes=('', '_' + str(app))
        )

        merged_df.dropna(inplace=True)

    # =====================================================
    # BUILD X AND Y
    # =====================================================
    X = merged_df.values[:, 0]

    Y = merged_df.values[:, 2:]

    # =====================================================
    # FIX NEGATIVE DIFFERENCE
    # =====================================================
    diff = X - np.sum(Y, axis=1)

    diff[np.where(diff >= 0)] = 0

    X = X + np.abs(diff)

    # =====================================================
    # OPTIONAL THRESHOLDING
    # =====================================================
    if thresholds is not None:

        for idx, thr in thresholds.items():

            Y[:, idx] = Y[:, idx] * (Y[:, idx] > thr)

    # =====================================================
    # BINARY ACTIVATIONS
    # =====================================================
    binaries = Y > 10

    print("\nActivation Ratios:\n")

    for ratio, app in zip(
        (np.sum(binaries, axis=0) / binaries.shape[0]),
        chosen_apps
    ):

        print(house5_appp_id[app], ":", ratio)

    # =====================================================
    # NORMALIZATION
    # =====================================================
    if normalize:

        X = (
            (X - np.min(X))
            / (np.max(X) - np.min(X) + 1e-12)
        )

        Y = (
            (Y - np.min(Y, axis=0))
            / (
                np.max(Y, axis=0)
                - np.min(Y, axis=0)
                + 1e-12
            )
        )

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================
    sep = int(train_ratio * X.shape[0])

    row_X_train = X[:sep].copy()
    row_Y_train = Y[:sep].copy()

    row_X_test = X[sep:].copy()
    row_Y_test = Y[sep:].copy()

    # =====================================================
    # SPLIT FUNCTIONS
    # =====================================================
    def split_data(data, leng=100, overlap_step=50):

        XTR = data[0].copy()
        YTR = data[1].copy()

        M = int(XTR.shape[0])

        lisXTR = []
        lisYTR = []

        for l in range(0, M - leng, overlap_step):

            lIdx = l
            hIdx = l + leng

            lisXTR.append(
                XTR[lIdx:hIdx].copy()
            )

            lisYTR.append(
                YTR[lIdx:hIdx].copy()
            )

        return (
            np.asarray(lisXTR),
            np.asarray(lisYTR)
        )

    def split_data_test(data, leng=100):

        XTR = data[0].copy()
        YTR = data[1].copy()

        M = int(XTR.shape[0])

        lisXTR = []
        lisYTR = []

        for l in range(0, M - leng, leng):

            lIdx = l
            hIdx = l + leng

            lisXTR.append(
                XTR[lIdx:hIdx].copy()
            )

            lisYTR.append(
                YTR[lIdx:hIdx].copy()
            )

        return (
            np.asarray(lisXTR),
            np.asarray(lisYTR)
        )

    # =====================================================
    # CREATE WINDOWS
    # =====================================================
    X_train, Y_train = split_data(
        [row_X_train, row_Y_train],
        segment_length,
        overlap
    )

    X_test, Y_test = split_data_test(
        [row_X_test, row_Y_test],
        segment_length
    )

    # =====================================================
    # PRINT SHAPES
    # =====================================================
    print("\nShapes\n")

    print("X_train :", X_train.shape)
    print("Y_train :", Y_train.shape)

    print("X_test  :", X_test.shape)
    print("Y_test  :", Y_test.shape)

    # =====================================================
    # RETURN
    # =====================================================
    return (
        X_train,
        X_test,
        Y_train,
        Y_test,
        X,
        Y,
        house5_appp_id[chosen_apps]
    )