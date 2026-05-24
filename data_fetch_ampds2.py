import numpy as np

def get_data(segment_length=30,
             p_path="/content/drive/MyDrive/progressive disag/ampds2/Electricity_P.csv",
             q_path="/content/drive/MyDrive/progressive disag/ampds2/Electricity_Q.csv",
             i_path="/content/drive/MyDrive/progressive disag/ampds2/Electricity_I.csv",
             train_test_days=[365, 365+60]):

    # =========================
    # LOAD DATA
    # =========================
    data_csv_P = np.genfromtxt(p_path, delimiter=",")
    data_csv_Q = np.genfromtxt(q_path, delimiter=",")
    data_csv_I = np.genfromtxt(i_path, delimiter=",")

    cols = ["UNIX_TS","WHE","RSE","GRE","MHE","B1E","BME","CWE","DWE","EQE","FRE",
            "HPE","OFE","UTE","WOE","B2E","CDE","DNE","EBE","FGE","HTE","OUE","TVE","UNE"]

    # =========================
    # ACTIVE POWER (P)
    # =========================
    WHEp = data_csv_P[1:-1,1]
    RSEp = data_csv_P[1:-1,2]
    GREp = data_csv_P[1:-1,3]

    FGEp = data_csv_P[1:-1,cols.index("FGE")]
    HPEp = data_csv_P[1:-1,cols.index("HPE")]
    CWEp = data_csv_P[1:-1,cols.index("CWE")]
    WOEp = data_csv_P[1:-1,cols.index("WOE")]
    HTEp = data_csv_P[1:-1,cols.index("HTE")]
    FREp = data_csv_P[1:-1,cols.index("FRE")]
    DWEp = data_csv_P[1:-1,cols.index("DWE")]
    CDEp = data_csv_P[1:-1,cols.index("CDE")]
    OFEp = data_csv_P[1:-1,cols.index("OFE")]
    BMEp = data_csv_P[1:-1,cols.index("BME")]
    TVEp = data_csv_P[1:-1,cols.index("TVE")]

    MHEp = WHEp - (RSEp + GREp)

    # =========================
    # REACTIVE POWER (Q)
    # =========================
    WHEq = data_csv_Q[1:-1,1]
    RSEq = data_csv_Q[1:-1,2]
    GREq = data_csv_Q[1:-1,3]

    FGEq = data_csv_Q[1:-1,cols.index("FGE")]
    HPEq = data_csv_Q[1:-1,cols.index("HPE")]
    CWEq = data_csv_Q[1:-1,cols.index("CWE")]
    WOEq = data_csv_Q[1:-1,cols.index("WOE")]
    HTEq = data_csv_Q[1:-1,cols.index("HTE")]
    FREq = data_csv_Q[1:-1,cols.index("FRE")]
    DWEq = data_csv_Q[1:-1,cols.index("DWE")]
    CDEq = data_csv_Q[1:-1,cols.index("CDE")]
    OFEq = data_csv_Q[1:-1,cols.index("OFE")]
    BMEq = data_csv_Q[1:-1,cols.index("BME")]
    TVEq = data_csv_Q[1:-1,cols.index("TVE")]

    MHEq = WHEq - (RSEq + GREq)

    # =========================
    # CURRENT (I)
    # =========================
    WHEi = data_csv_I[1:-1,1]
    RSEi = data_csv_I[1:-1,2]
    GREi = data_csv_I[1:-1,3]

    FGEi = data_csv_I[1:-1,cols.index("FGE")]
    HPEi = data_csv_I[1:-1,cols.index("HPE")]
    CWEi = data_csv_I[1:-1,cols.index("CWE")]
    WOEi = data_csv_I[1:-1,cols.index("WOE")]
    HTEi = data_csv_I[1:-1,cols.index("HTE")]
    FREi = data_csv_I[1:-1,cols.index("FRE")]
    DWEi = data_csv_I[1:-1,cols.index("DWE")]
    CDEi = data_csv_I[1:-1,cols.index("CDE")]
    OFEi = data_csv_I[1:-1,cols.index("OFE")]
    BMEi = data_csv_I[1:-1,cols.index("BME")]
    TVEi = data_csv_I[1:-1,cols.index("TVE")]

    MHEi = WHEi - (RSEi + GREi)

    # =========================
    # STACK PER APPLIANCE (P, Q, I)
    # =========================
    def stack(p, q, i):
        return np.asarray([p, q, i]).T

    appliancesR = np.zeros((len(FGEp), 3, 11))

    appliancesR[:, :, 0] = stack(FGEp, FGEq, FGEi)
    appliancesR[:, :, 1] = stack(HPEp, HPEq, HPEi)
    appliancesR[:, :, 2] = stack(CWEp, CWEq, CWEi)
    appliancesR[:, :, 3] = stack(WOEp, WOEq, WOEi)
    appliancesR[:, :, 4] = stack(HTEp, HTEq, HTEi)
    appliancesR[:, :, 5] = stack(FREp, FREq, FREi)
    appliancesR[:, :, 6] = stack(DWEp, DWEq, DWEi)
    appliancesR[:, :, 7] = stack(CDEp, CDEq, CDEi)
    appliancesR[:, :, 8] = stack(OFEp, OFEq, OFEi)
    appliancesR[:, :, 9] = stack(BMEp, BMEq, BMEi)
    appliancesR[:, :, 10] = stack(TVEp, TVEq, TVEi)

    # =========================
    # SPLIT FUNCTION
    # =========================
    def split_test(X, Y, days):
        ilow = days[0] * 24 * 60
        ihigh = days[1] * 24 * 60

        Xtr, Xts, Ytr, Yts = [], [], [], []

        for i in range(len(X)):
            if ilow < i <= ihigh:
                Xts.append(X[i].copy())
                Yts.append(Y[i].copy())
            else:
                Xtr.append(X[i].copy())
                Ytr.append(Y[i].copy())

            if i > ihigh:
                break

        return np.asarray(Xtr), np.asarray(Xts), np.asarray(Ytr), np.asarray(Yts)

    rawXTR, rawXTS, rawYTR, rawYTS = split_test(
        np.asarray([MHEp, MHEq, MHEi]).T,
        appliancesR,
        train_test_days
    )

    # =========================
    # SEQUENCE BUILDER
    # =========================
    def split_data2(data, length):
        XTR, XTS, YTR, YTS = data

        lisXTR, lisXTS = [], []
        lisYTR, lisYTS = [], []

        M = len(XTR)
        N = len(XTS)

        # TRAIN
        for i in range(M - length):
            mid = i + length // 2
            lisXTR.append(XTR[i:i+length].copy())
            lisYTR.append(YTR[mid].copy())

        # TEST
        for i in range(N - length):
            mid = i + length // 2
            lisXTS.append(XTS[i:i+length].copy())
            lisYTS.append(YTS[mid].copy())

        return (np.asarray(lisXTR),
                np.asarray(lisXTS),
                np.asarray(lisYTR),
                np.asarray(lisYTS))

    X_train, X_test, y_train, y_test = split_data2(
        [rawXTR, rawXTS, rawYTR, rawYTS],
        segment_length
    )

    return X_train, X_test, y_train, y_test