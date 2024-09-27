import os
import pandas as pd
import numpy as np

yaw_path = "data/Case_01/measurements_turbines/30000_BL/nacYaw"
rotspeed_path = "data/Case_01/measurements_turbines/30000_BL/rotSpeed"
wind_direction_path = "data/Case_01/HKN_12_to_15_dir.csv"

def main():
    df_yaw = pd.read_csv(yaw_path, sep=" ")

    df_yaw = df_yaw.drop(df_yaw.columns[4:], axis=1)
    df_yaw.columns = ["turbine", "time", "dt(s)", "yaw"]
    df_yaw = df_yaw.drop("dt(s)", axis=1)

    df_yaw = df_yaw[df_yaw['time'].isin(range(30005, 42001, 5))]
    df_yaw['time'] = df_yaw['time'].astype(int)

    df_yaw['yaw_sin'] = df_yaw['yaw'].apply(lambda x: np.sin(np.deg2rad(x)))
    df_yaw['yaw_cos'] = df_yaw['yaw'].apply(lambda x: np.cos(np.deg2rad(x)))
    df_yaw = df_yaw.reset_index(drop=True)

    df_rot = pd.read_csv(rotspeed_path, sep=" ")

    df_rot = df_rot.drop(df_rot.columns[4:], axis=1)
    df_rot.columns = ["turbine", "time", "dt(s)", "speed"]
    df_rot = df_rot.drop("dt(s)", axis=1)

    df_rot = df_rot[df_rot['time'].isin(range(30005, 42001, 5))]
    df_rot['time'] = df_rot['time'].astype(int)    
    df_rot = df_rot.reset_index(drop=True)

    df_combined = pd.merge(df_rot, df_yaw)
    df_combined.to_csv("data/Case_01/measurements_turbines/30000_BL/rot_yaw_combined.csv")

    wind_direction = pd.read_csv(wind_direction_path, header=None)
    wind_direction.columns = ["time", "direction"]
    wind_direction[['time']] = wind_direction[['time']]+30000

    time_range = pd.DataFrame()
    time_range['time'] = list(range(30005, 42001, 5))
    wind_direction = time_range.set_index("time").join(wind_direction.set_index("time"), how="outer")
    wind_direction = wind_direction.reset_index()
    wind_direction = wind_direction.ffill()
    wind_direction = wind_direction.drop(0, axis=0)

    wind_direction['winddir_sin'] = wind_direction['direction'].apply(lambda x: np.sin(np.deg2rad(x)))
    wind_direction['winddir_cos'] = wind_direction['direction'].apply(lambda x: np.cos(np.deg2rad(x)))    

    wind_direction.to_csv("data/Case_01/winDir_processed.csv")


if __name__ == "__main__":
    print(os.getcwd())
    main()