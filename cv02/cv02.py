import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint


def load_three_columns_from_csv(file_path, col1, col2, col3):
    # Read the CSV file without specifying columns to find out the exact column names
    df_preview = pd.read_csv(file_path, nrows=1)
    exact_col1 = df_preview.columns[df_preview.columns.str.strip() == col1.strip()][0]
    exact_col2 = df_preview.columns[df_preview.columns.str.strip() == col2.strip()][0]
    exact_col3 = df_preview.columns[df_preview.columns.str.strip() == col3.strip()][0]

    # Now read the CSV file with the exact column names
    df = pd.read_csv(file_path, usecols=[exact_col1, exact_col2, exact_col3])
    return df, exact_col1, exact_col2, exact_col3


def plot_columns(df, col1, col2, col3):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].plot(df[col1], df[col2])
    axes[0].set_title(f"{col2}")
    axes[0].set_xlabel(col1)
    axes[0].set_ylabel(col2)
    axes[0].grid()

    axes[1].plot(df[col1], df[col3])
    axes[1].set_title(f"{col3}")
    axes[1].set_xlabel(col1)
    axes[1].set_ylabel(col3)
    axes[1].grid()

    plt.tight_layout()
    plt.show()

    return axes


def bergman_ode(y, t, p, I_interpolated):
    G, X = y
    p2, Ib, SI, SG, Gb = p
    I = I_interpolated(t)
    dGdt = -(SG*X)*G + SG*Gb
    dXdt = -p2*X + p2*SI*(I - Ib)
    return dGdt, dXdt


# Example usage
if __name__ == "__main__":
    file_path = "Dat_IVGTT_AP.csv"
    col1, col2, col3 = 'time (minutes)', 'glucose level (mg/dl)', 'insulin level (μU/ml)'

    df, col1, col2, col3 = load_three_columns_from_csv(file_path, col1, col2, col3)

    time_data = df[col1].values
    insulin_data = df[col3].values
    I_interpolated = lambda t: np.interp(t, time_data, insulin_data)

    G0 = df[col2].iloc[0]
    X0 = 0
    initial_conditions = [G0, X0]
    Gb = 4.703760
    Ib = 18.02
    p2 = 1.0160/1e4
    SI = 1.3612/1e1
    SG = 3.6105/1e2
    params = [p2, Ib, SI, SG, Gb]

    # Time points for ODE solver
    t = np.linspace(0, np.max(df[col1]), 1000)

    # Solve ODE
    solution = odeint(bergman_ode, initial_conditions, t, args=(params, I_interpolated))
    G = solution[:, 0]
    X = solution[:, 1]
    print(G)

    plot_columns(df, col1, col2, col3)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot for glucose
    axes[0].plot(t, G, label='Simulated G')
    axes[0].scatter(df[col1], df[col2], c='r', marker='o', label='Measured G')
    axes[0].set_xlabel('Time (minutes)')
    axes[0].set_ylabel('Glucose (mg/dl)')
    axes[0].legend()

    # Plot for insulin action
    axes[1].plot(t, X, label='Simulated X')
    axes[1].set_xlabel('Time (minutes)')
    axes[1].set_ylabel('Insulin Action')
    axes[1].legend()

    # Plot for insulin
    axes[2].scatter(df[col1], df[col3], c='b', marker='x', label='Measured I')
    axes[2].set_xlabel('Time (minutes)')
    axes[2].set_ylabel('Insulin (uU/ml)')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
