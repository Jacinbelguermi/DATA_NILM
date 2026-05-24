import numpy as np
import pandas as pd


def get_data(
    x_train_path="/content/drive/MyDrive/progressive disag/eco_pq/X_train_raw_10s.csv",
    y_train_path="/content/drive/MyDrive/progressive disag/eco_pq/Y_train_raw_10s.csv",
    x_test_path="/content/drive/MyDrive/progressive disag/eco_pq/X_test_raw_10s.csv",
    y_test_path="/content/drive/MyDrive/progressive disag/eco_pq/Y_test_raw_10s.csv",
    resample_steps=6,
    segment_length=30,
    aggregate_input=False,
    normalization=False
):

    # =====================================================
    # APPLIANCE NAMES
    # =====================================================
    appliances_list = [
        'DWEn',
        'AIRn',
        'FGEn',
        'KETn',
        'FRZn',
        'MEDn',
        'TBLn',
        'LMPn'
    ]

    # =====================================================
    # RESAMPLE FUNCTION
    # =====================================================
    def myresample(signal, steps=60):

        temp = signal[::steps]

        return temp.copy()

    # =====================================================
    # SPLIT FUNCTION
    # =====================================================
    def split_data(data, length=100):

        XTR = data[0].copy()
        YTR = data[1].copy()

        M = int(XTR.shape[0])

        lisXTR = []
        lisYTR = []

        for i in range(M - length):

            lIdx = i
            hIdx = i + length

            lisXTR.append(
                XTR[lIdx:hIdx].copy()
            )

            lisYTR.append(
                YTR[hIdx - 1].copy()
            )

        return (
            np.asarray(lisXTR),
            np.asarray(lisYTR)
        )

    # =====================================================
    # LOAD DATA
    # =====================================================
    X_train_raw = np.loadtxt(
        x_train_path,
        delimiter=","
    )

    Y_train_raw = np.loadtxt(
        y_train_path,
        delimiter=","
    )

    X_test_raw = np.loadtxt(
        x_test_path,
        delimiter=","
    )

    Y_test_raw = np.loadtxt(
        y_test_path,
        delimiter=","
    )

    # =====================================================
    # RESAMPLE
    # =====================================================
    X_train_raw = myresample(
        X_train_raw,
        steps=resample_steps
    )

    Y_train_raw = myresample(
        Y_train_raw,
        steps=resample_steps
    )

    X_test_raw = myresample(
        X_test_raw,
        steps=resample_steps
    )

    Y_test_raw = myresample(
        Y_test_raw,
        steps=resample_steps
    )

    # =====================================================
    # MERGE CHANNELS
    # =====================================================
    Y_train_raw[:, 5] = (
        Y_train_raw[:, 5]
        + Y_train_raw[:, 7]
        + Y_train_raw[:, 9]
    )

    Y_test_raw[:, 5] = (
        Y_test_raw[:, 5]
        + Y_test_raw[:, 7]
        + Y_test_raw[:, 9]
    )

    # =====================================================
    # SELECT TARGETS
    # =====================================================
    indexes = [0,1,2,3,4,5,6,8]

    train_set_target = np.zeros(
        (Y_train_raw.shape[0], 8)
    )

    test_set_target = np.zeros(
        (Y_test_raw.shape[0], 8)
    )

    train_set_target[:] = Y_train_raw[:, indexes]

    test_set_target[:] = Y_test_raw[:, indexes]

    # =====================================================
    # TARGET STATISTICS
    # =====================================================
    total_targets = np.concatenate(
        (train_set_target, test_set_target)
    )

    binary_targets = total_targets > 1

    sum_targets = np.sum(
        binary_targets,
        axis=0
    )

    percentage_targets = (
        sum_targets / total_targets.shape[0]
    ) * 100

    print("\nActivation Percentages:\n")

    for i in range(8):

        print(
            appliances_list[i],
            ":",
            percentage_targets[i]
        )

    # =====================================================
    # AVERAGE ACTIVE POWER
    # =====================================================
    avg = []

    print("\nAverage Active Power:\n")

    for i in range(8):

        active_values = total_targets[
            np.where(total_targets[:, i] > 1),
            i
        ]

        mean_val = np.mean(active_values)

        avg.append(mean_val)

        print(
            appliances_list[i],
            ":",
            mean_val
        )

    avg[3] = 200

    # =====================================================
    # FIX NEGATIVE AGGREGATE
    # =====================================================
    diff_train = (
        X_train_raw[:, 0]
        - np.sum(train_set_target, axis=1)
    )

    diff_train[np.where(diff_train < 0)] = 0

    new_X_train = (
        diff_train
        + np.sum(train_set_target, axis=1)
    )

    diff_test = (
        X_test_raw[:, 0]
        - np.sum(test_set_target, axis=1)
    )

    diff_test[np.where(diff_test < 0)] = 0

    new_X_test = (
        diff_test
        + np.sum(test_set_target, axis=1)
    )

    # =====================================================
    # OPTIONAL NORMALIZATION
    # =====================================================
    if normalization:

        x_min = np.min(new_X_train)
        x_max = np.max(new_X_train)

        new_X_train = (
            (new_X_train - x_min)
            / (x_max - x_min + 1e-12)
        )

        new_X_test = (
            (new_X_test - x_min)
            / (x_max - x_min + 1e-12)
        )

        y_min = np.min(train_set_target, axis=0)
        y_max = np.max(train_set_target, axis=0)

        train_set_target = (
            (train_set_target - y_min)
            / (y_max - y_min + 1e-12)
        )

        test_set_target = (
            (test_set_target - y_min)
            / (y_max - y_min + 1e-12)
        )

    # =====================================================
    # INPUT SELECTION
    # =====================================================
    if aggregate_input:

        train_input = np.sum(
            train_set_target,
            axis=1
        )

        test_input = np.sum(
            test_set_target,
            axis=1
        )

    else:

        train_input = new_X_train

        test_input = new_X_test

    # =====================================================
    # BUILD SEQUENCES
    # =====================================================
    X_train, Y_train = split_data(
        [train_input, train_set_target],
        segment_length
    )

    X_test, Y_test = split_data(
        [test_input, test_set_target],
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
        new_X_train,
        new_X_test,
        train_set_target,
        test_set_target,
        appliances_list,
        avg,
        percentage_targets
    )