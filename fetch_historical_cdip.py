"""
Fetch historical CDIP MOP (Monitoring and Prediction) nearshore data for calibration.

CDIP MOP = nearshore spectral model along transects. Distinct from CDIP buoy (offshore).
Data sources:
- nowcast.nc: Recent ~8-9 months of data (updated daily)
- hindcast.nc: Historical data from 2000 to where nowcast begins

Adds hs_mop_ft, per_pri_mop_s, dir_peak_deg_mop, per_weighted_s_mop to observations (MOP nearshore values).

CDIP MOP NetCDF ``waveTime`` is Unix epoch **seconds UTC**. Observation rows use ``datetime_pst`` converted to UTC
before nearest-index matching (same convention as offshore buoy enrichment).
"""

from datetime import datetime, timezone
import time
from typing import Optional

import netCDF4
import numpy as np
import pandas as pd
import pytz
import structlog

logger = structlog.get_logger()

# CDIP MOP alongshore nearshore model (transect-based)
CDIP_MOP_ALONGSHORE_BASE = "http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/"


def fetch_cdip_mop_data(
    transect_id: str,
    target_timestamps: list = None,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Fetch historical CDIP MOP nearshore data from nowcast/hindcast NetCDF.

    MOP = Monitoring and Prediction (nearshore spectral model along transects).
    Returns dict with waveTime, waveHs, waveTp, waveDp, waveDm, waveTa arrays.
    """
    variables = ["waveTime", "waveHs", "waveTp", "waveDp", "waveDm", "waveTa"]
    all_data = {var: [] for var in variables}

    nowcast_url = f"{CDIP_MOP_ALONGSHORE_BASE}{transect_id}_nowcast.nc"
    hindcast_url = f"{CDIP_MOP_ALONGSHORE_BASE}{transect_id}_hindcast.nc"

    for attempt in range(max_retries):
        try:
            ds = netCDF4.Dataset(nowcast_url)
            for var in variables:
                if var in ds.variables:
                    all_data[var].append(ds.variables[var][:])
            ds.close()
            break
        except Exception as e:
            logger.warning("cdip_mop_nowcast_retry", transect=transect_id, attempt=attempt + 1, error=str(e))
            if attempt < max_retries - 1:
                time.sleep(2)

    # Check if we need hindcast data
    need_hindcast = False
    wt = all_data.get("waveTime")
    if target_timestamps and wt and len(wt) > 0 and len(wt[0]) > 0:
        nowcast_min_time = float(np.min(wt[0]))
        for ts in target_timestamps:
            if ts < nowcast_min_time:
                need_hindcast = True
                break

    if need_hindcast:
        logger.info("cdip_mop_fetching_hindcast", transect=transect_id)
        for attempt in range(max_retries):
            try:
                ds = netCDF4.Dataset(hindcast_url)
                for var in variables:
                    if var in ds.variables:
                        all_data[var].insert(0, ds.variables[var][:])
                ds.close()
                break
            except Exception as e:
                logger.warning("cdip_mop_hindcast_retry", transect=transect_id, attempt=attempt + 1, error=str(e))
                if attempt < max_retries - 1:
                    time.sleep(2)

    result = {}
    for var in variables:
        if all_data[var]:
            result[var] = np.concatenate(all_data[var])

    if "waveTime" not in result or len(result["waveTime"]) == 0:
        logger.error("cdip_mop_fetch_failed", transect=transect_id)
        return None

    return result


def find_nearest_time_index(wave_times: np.ndarray, target_timestamp: float) -> int:
    """Find index of nearest time in wave_times (within 6 hours)."""
    if len(wave_times) == 0:
        return -1
    diffs = np.abs(wave_times - target_timestamp)
    min_idx = np.argmin(diffs)
    min_diff_hours = diffs[min_idx] / 3600
    if min_diff_hours > 6:
        return -1
    return min_idx


def meters_to_feet(meters: float) -> float:
    return meters * 3.28084


def enrich_observations_with_cdip_mop(
    obs_df: pd.DataFrame,
    breaks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich calibration observations with historical CDIP MOP nearshore data.

    Adds hs_mop_ft, per_pri_mop_s, dir_peak_deg_mop, per_weighted_s_mop (MOP nearshore model).
    Distinct from buoy enrichment (fetch_historical_buoy) which adds offshore buoy data.

    Args:
        obs_df: DataFrame with spot, datetime_pst, observed_hs_ft columns
        breaks_df: DataFrame with break_name, spot_name, cdip_transect, lat, lon columns
    """
    df = obs_df.copy()
    pst = pytz.timezone("US/Pacific")

    # Build spot -> transect mapping from breaks_df
    spot_to_transect = {}
    for _, row in breaks_df.iterrows():
        break_name = str(row.get('break_name', row.get('break', ''))).strip()
        spot_name = str(row.get('spot_name', row.get('spot', ''))).strip()
        transect = row.get('cdip_transect', '')
        lat = row.get('lat', None)
        lon = row.get('lon', row.get('lng', row.get('long', None)))

        if not transect:
            continue

        info = {'transect': transect, 'lat': lat, 'lon': lon}

        if spot_name and break_name:
            combined = f"{spot_name} {break_name}".lower()
            spot_to_transect[combined] = info
        if break_name:
            spot_to_transect[break_name.lower()] = info
        if spot_name:
            spot_to_transect[spot_name.lower()] = info

    def find_transect(spot_name):
        if pd.isna(spot_name):
            return None
        spot_lower = str(spot_name).lower().strip()
        if spot_lower in spot_to_transect:
            return spot_to_transect[spot_lower]
        for key, val in spot_to_transect.items():
            if spot_lower in key or key in spot_lower:
                return val
        return None

    lookup_col = df['break'] if 'break' in df.columns else df['spot']
    df['_transect_info'] = lookup_col.apply(find_transect)
    df['cdip_transect'] = df['_transect_info'].apply(lambda x: x['transect'] if x else None)
    df['lat'] = df['_transect_info'].apply(lambda x: x['lat'] if x else None)
    df['lon'] = df['_transect_info'].apply(lambda x: x['lon'] if x else None)
    df = df.drop(columns=['_transect_info'])

    # Ensure datetime_pst exists
    if 'datetime_pst' not in df.columns:
        if 'date' in df.columns and 'time' in df.columns:
            def fix_short_year(date_str):
                date_str = str(date_str).strip()
                parts = date_str.split('-')
                if len(parts) == 3 and len(parts[2]) == 2:
                    return f"{parts[0]}-{parts[1]}-20{parts[2]}"
                return date_str

            df['_date_fixed'] = df['date'].apply(fix_short_year)
            df['datetime_pst'] = pd.to_datetime(
                df['_date_fixed'] + ' ' + df['time'].astype(str), errors='coerce'
            )
            df = df.drop(columns=['_date_fixed'])
            df['datetime_pst'] = df['datetime_pst'].apply(
                lambda x: pst.localize(x) if pd.notna(x) and x.tzinfo is None else x
            )
        else:
            logger.error("cdip_mop_datetime_pst_missing")
            return df

    unique_transects = df['cdip_transect'].dropna().unique()
    transect_timestamps = {}
    for _, row in df.iterrows():
        transect = row['cdip_transect']
        dt = row['datetime_pst']
        if pd.notna(transect) and pd.notna(dt):
            try:
                if hasattr(dt, 'astimezone'):
                    obs_utc = dt.astimezone(pytz.UTC)
                else:
                    obs_utc = pst.localize(pd.to_datetime(dt)).astimezone(pytz.UTC)
                ts = obs_utc.timestamp()
                if transect not in transect_timestamps:
                    transect_timestamps[transect] = []
                transect_timestamps[transect].append(ts)
            except Exception as e:
                logger.warning("cdip_mop_datetime_convert_failed", dt=str(dt), error=str(e))

    logger.info("cdip_mop_fetching", transects=len(unique_transects))

    transect_data = {}
    for transect in unique_transects:
        timestamps = transect_timestamps.get(transect, [])
        data = fetch_cdip_mop_data(transect, target_timestamps=timestamps)
        if data is not None:
            transect_data[transect] = data
            logger.debug("cdip_mop_transect_fetched", transect=transect, records=len(data["waveTime"]))
        else:
            logger.warning("cdip_mop_transect_failed", transect=transect)

    logger.info("cdip_mop_fetch_complete", transects=len(transect_data))

    # Match each observation to CDIP MOP nearshore data
    mop_hs_values = []
    mop_period_values = []
    mop_direction_values = []
    mop_weighted_period_values = []

    for _, row in df.iterrows():
        transect = row['cdip_transect']
        obs_datetime = row['datetime_pst']

        mop_hs = None
        mop_period = None
        mop_dir = None
        mop_period_weighted = None

        if transect in transect_data and pd.notna(obs_datetime):
            data = transect_data[transect]
            try:
                if hasattr(obs_datetime, 'astimezone'):
                    obs_utc = obs_datetime.astimezone(pytz.UTC)
                else:
                    obs_utc = pst.localize(pd.to_datetime(obs_datetime)).astimezone(pytz.UTC)
                obs_timestamp = obs_utc.timestamp()

                wave_times = data["waveTime"]
                nearest_idx = find_nearest_time_index(wave_times, obs_timestamp)

                if nearest_idx >= 0:
                    mop_hs = meters_to_feet(float(data["waveHs"][nearest_idx]))
                    mop_period = float(data["waveTp"][nearest_idx])
                    mop_dir = float(data["waveDp"][nearest_idx])
                    if "waveTa" in data:
                        mop_period_weighted = float(data["waveTa"][nearest_idx])
            except Exception as e:
                logger.warning("cdip_mop_match_failed", error=str(e))

        mop_hs_values.append(mop_hs)
        mop_period_values.append(mop_period)
        mop_direction_values.append(mop_dir)
        mop_weighted_period_values.append(mop_period_weighted)

    df["hs_mop_ft"] = mop_hs_values
    df["per_pri_mop_s"] = mop_period_values
    df["dir_peak_deg_mop"] = mop_direction_values
    # Canonical weighted MOP period feature (CDIP MOP waveTa, seconds).
    df["per_weighted_s_mop"] = mop_weighted_period_values
    # Back-compat alias kept for older QC scripts.
    df["period_weighted_s_mop"] = df["per_weighted_s_mop"]

    # Scalar = observed / MOP (observed_hs_ft = human observation, not from swell model)
    def compute_scalar(row):
        obs = row.get("observed_hs_ft")
        mop = row.get("hs_mop_ft")
        if pd.notna(obs) and pd.notna(mop) and mop > 0:
            return obs / mop
        return None

    df["scalar"] = df.apply(compute_scalar, axis=1)

    valid_count = df['scalar'].notna().sum()
    logger.info("cdip_mop_matched", matched=valid_count, total=len(df))
    if valid_count > 0:
        scalars = df['scalar'].dropna()
        logger.info("cdip_mop_scalar_stats", median=round(scalars.median(), 2), mean=round(scalars.mean(), 2),
                   min=round(scalars.min(), 2), max=round(scalars.max(), 2))

    return df


# Alias for backward compatibility
enrich_observations_with_cdip = enrich_observations_with_cdip_mop
