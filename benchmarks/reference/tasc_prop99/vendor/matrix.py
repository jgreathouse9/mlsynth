import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from syslibutils import hsvt


class Matrix:
    def __init__(
        self, data: pd.DataFrame, T0: int, target_name, donor_names: list = None
    ):
        """
        Initialize the data matrix for RSC
        Args:
            data: pandas dataframe; each column is a time series (T by n matrix, T time points, n units)
            T0: int, the number of pre-intervention time points.
            num_sv : number of singular values to keep
            target_name: name of series that we intend to predict (target unit)
            donor_names: list of non-target ids (donor pool)

        Notes:
            In the paper, T0 starts with 1, so first intervention point is T0+1
            Here time index starts with 0, so first treatment/post-intervention unit's index is T0
        """
        self.data = data
        self.T0 = T0
        self.target_name = target_name

        self.T = data.shape[0]
        self.n = data.shape[1]

        self.data_is_transformed = False
        self.data_is_denoised = False

        # if donors are not specified, then they are all column names in data excluding target
        if donor_names is None:
            # self.donor_names = np.setdiff1d(self.data.columns.values, target_name)
            self.donor_names = self.data.columns.values[
                self.data.columns.values != target_name
            ]
        else:
            self.donor_names = donor_names

    def transform(self, method: str = "standard"):
        """
        Transform the data using the specified scaling method.
        Scaler should always be fit on donor pool only.
        Scaler should always be fit on time-feature wise. Hence, the data matrix should be when performing scaler transposed.

        Args:
            method: The scaling method to use ('standard' or 'minmax').
        """
        if self.data_is_transformed:
            raise ValueError("Data is already transformed.")

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid method. Choose 'standard' or 'minmax'.")

        self.scaler.fit(self.data.drop(columns=self.target_name).T)
        self.data = pd.DataFrame(
            self.scaler.transform(self.data.T),
            index=self.data.columns,
            columns=self.data.index,
        ).T
        self.data_is_transformed = True

    def inverse_transform(self):
        if not self.data_is_transformed:
            raise ValueError("Data is not transformed yet.")

        self.data = pd.DataFrame(
            self.scaler.inverse_transform(self.data.T),
            index=self.data.columns,
            columns=self.data.index,
        ).T
        self.data_is_transformed = False

    def denoise(
        self,
        num_sv,
        p: float = 1.0,
        filter_method: str = "HSVT",
        transform: bool = False,
    ):
        # Do not set default num_sv value. It should be specified by the user.
        if transform:
            self.transform()

        if filter_method == "HSVT":
            # do hsvt for donor pool only
            denoised_donors = hsvt(
                self.data.drop(columns=self.target_name), rank=num_sv, p=p
            )
        self.data = pd.concat([self.data[self.target_name], denoised_donors], axis=1)
        self.data_is_denoised = True

    @property
    def target(self):
        return pd.concat((self.pre_target, self.post_target), axis=0)

    @property
    def donor(self):
        return pd.concat((self.pre_donor, self.post_donor), axis=0)

    @property
    def pre_target(self):
        return self.data[: self.T0][[self.target_name]]

    @property
    def post_target(self):
        return self.data[self.T0 :][[self.target_name]]

    @property
    def pre_donor(self):
        return self.data[: self.T0].drop(
            columns=self.target_name, errors="ignore"
        )  # ignores KeyError

    @property
    def post_donor(self):
        return self.data[self.T0 :].drop(columns=self.target_name, errors="ignore")


# Test with random values
"""
X = np.arange(25).reshape((5,5))  
df = pd.DataFrame(X)
M = Matrix(df, T0 = 3)

print('data\n', df)
print('\npre_target\n', M.pre_target)
print('\npost_target\n', M.post_target)
print('\npre_donor\n', M.pre_donor)
print('\npost_donor\n', M.post_donor)
"""
