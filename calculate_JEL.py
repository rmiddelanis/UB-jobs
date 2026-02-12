import sys
import os
import subprocess
import importlib.util
import pandas as pd

scriptdir = os.path.dirname(os.path.abspath(__file__))

# Check whether model data exists; if not, run the model first
model_data_dir = os.path.join(scriptdir, 'data', 'model_data')
if not os.path.isdir(model_data_dir):
    print("Model data not found. Running the model first...")
    run_model_script = os.path.join(scriptdir, 'lib', 'global-unbreakable-model', 'src', 'unbreakable', 'model', 'run_model.py')
    settings_file = os.path.join(scriptdir, 'model_settings.yml')
    subprocess.run(
        ['conda', 'run', '--no-capture-output', '-n', 'global-socioeconomic-resilience',
         'python', run_model_script, settings_file],
        check=True,
    )

# Load average_over_rp directly from helpers.py to avoid pulling in the full
# unbreakable package (which requires xarray, pycountry, etc. at import time).
_helpers_path = os.path.join(scriptdir, 'lib', 'global-unbreakable-model', 'src', 'unbreakable', 'misc', 'helpers.py')
_spec = importlib.util.spec_from_file_location('helpers', _helpers_path, submodule_search_locations=[])
_helpers = importlib.util.module_from_spec(_spec)
# Pre-populate heavy dependencies that helpers.py imports at module level
# so we don't need them installed just for average_over_rp.
sys.modules.setdefault('xarray', type(sys)('xarray'))
sys.modules.setdefault('requests', type(sys)('requests'))
sys.modules.setdefault('pycountry', type(sys)('pycountry'))
_spec.loader.exec_module(_helpers)
average_over_rp = _helpers.average_over_rp

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
iah = pd.read_csv(
    os.path.join(model_data_dir, 'simulation_outputs', 'iah.csv'),
    index_col=['iso3', 'hazard', 'rp', 'income_cat', 'affected_cat', 'helped_cat'],
)

cat_info = pd.read_csv(
    os.path.join(model_data_dir, 'model_inputs', 'scenario__cat_info.csv'),
    index_col=['iso3', 'income_cat'],
)

hazard_prot = pd.read_csv(
    os.path.join(model_data_dir, 'model_inputs', 'scenario__hazard_protection.csv'),
    index_col=['iso3', 'hazard'],
)

macro = pd.read_csv(
    os.path.join(model_data_dir, 'model_inputs', 'scenario__macro.csv'),
    index_col='iso3',
)

work_hours = pd.read_excel(
    os.path.join(scriptdir, 'data', 'work_hours.xlsx'), na_values='NA'
).dropna(subset='iso3').set_index('iso3')

epr = pd.read_excel(
    os.path.join(scriptdir, 'data', 'employment_pop_ratio.xlsx'), na_values='NA'
).dropna(subset='iso3').set_index('iso3').squeeze().rename('EPR')

# ---------------------------------------------------------------------------
# 2. Aggregate di_lab to quintile level and compute AAL
# ---------------------------------------------------------------------------
# Population-weighted average across affected_cat / helped_cat sub-groups
di_lab = iah['di_lab'].to_frame()
di_lab = (
    di_lab
    .mul(iah.n, axis=0)
    .groupby(['iso3', 'hazard', 'rp', 'income_cat']).sum()
    .div(iah.n.groupby(['iso3', 'hazard', 'rp', 'income_cat']).sum(), axis=0)
)

# Average over return periods → AAL per (iso3, hazard, income_cat)
aal = average_over_rp(di_lab, hazard_prot.protection, zero_rp=2)

# ---------------------------------------------------------------------------
# 3. Baseline labor income per capita: l_{q,i} = c_{q,i} * (1 - diversified_share_{q,i})
# ---------------------------------------------------------------------------
l_baseline = (cat_info['c'] * (1 - cat_info['diversified_share'])).rename('l')

# ---------------------------------------------------------------------------
# 4. Population per quintile: N_{q,i} = N_i * 0.2
# ---------------------------------------------------------------------------
N_q = (macro['pop'] * 0.2).rename('N_q')

# ---------------------------------------------------------------------------
# 5. Compute JEL per (hazard, quintile) (equation 7):
#    JEL_{q,i} = EPR_i * (Δl_{q,i} / l_{q,i}) * N_{q,i}
# ---------------------------------------------------------------------------
jel_params = pd.concat([epr, N_q], axis=1)
jel_params = pd.merge(jel_params, l_baseline, left_index=True, right_index=True).dropna()
jel_params = pd.merge(jel_params, aal.di_lab, left_index=True, right_index=True).dropna()
jel_params.sort_index(inplace=True)

jel = (jel_params['EPR'] * jel_params['di_lab'] / jel_params['l'] * jel_params['N_q']).rename('JEL')

# Add all_hazards rows by summing across hazards
all_haz_q = jel.groupby(['iso3', 'income_cat']).sum().to_frame()
all_haz_q['hazard'] = 'all_hazards'
all_haz_q = all_haz_q.set_index('hazard', append=True).reorder_levels(jel.index.names).squeeze()
jel = pd.concat([jel, all_haz_q], axis=0).sort_index()

tot_pop = jel.groupby(['iso3', 'hazard']).sum().to_frame()
tot_pop['income_cat'] = 'total'
tot_pop = tot_pop.set_index('income_cat', append=True).reorder_levels(jel.index.names).squeeze()
jel = pd.concat([jel, tot_pop], axis=0).sort_index()

# ---------------------------------------------------------------------------
# 6. Time-adjusted JEL (equation 10):
#    JEL_i^{adj} = JEL_i * h_i^{week} / 40
#    where h_i^{week} = annual_work_hrs / 47
# ---------------------------------------------------------------------------
h_week = (work_hours['annual_work_hrs'] / 47).rename('h_week')

jel_results = jel.to_frame()
jel_results['JEL_adj'] = jel_results['JEL'] * h_week / 40
jel_results['JEL_pop_rel'] = jel_results['JEL'] / macro['pop']
jel_results['JEL_adj_pop_rel'] = jel_results['JEL_adj'] / macro['pop']

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(scriptdir, 'output'), exist_ok=True)

jel_results.to_csv(os.path.join(scriptdir, 'output', 'JEL.csv'))
jel_params.to_csv(os.path.join(scriptdir, 'output', 'JEL_params.csv'))

print(f"JEL parameters saved to output/JEL_params.csv ({len(jel_params)} rows)")
print(f"JEL results saved to output/JEL.csv ({len(jel_results)} rows)")

# ---------------------------------------------------------------------------
# 8. Global choropleth map of adjusted JEL
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import geopandas as gpd

# One row per country: all hazards, total population
jel_totals = jel_results.loc[(slice(None), 'total', 'all_hazards'), :].droplevel(['income_cat', 'hazard'])

world = gpd.read_file('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip')
world = world.merge(jel_totals[['JEL_adj', 'JEL_adj_pop_rel']], left_on='ISO_A3_EH', right_index=True, how='left')
world['JEL_adj_pop_pct'] = world['JEL_adj_pop_rel'] * 100

maps = [
    ('JEL_adj', 'full-time JEL (log scale)', 'Adjusted Job Equivalent Loss (JEL) by Country — All Hazards', 'jel_adj_map.png'),
    ('JEL_adj_pop_pct', 'full-time JEL, % of population (log scale)', 'Population-Relative full-time JEL by Country — All Hazards', 'jel_adj_pop_rel_map.png'),
]

for col, cbar_label, title, fname in maps:
    series = world[col].dropna()
    vmin = series[series > 0].min()
    vmax = series.max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    world.plot(
        column=col,
        ax=ax,
        legend=True,
        legend_kwds={'label': cbar_label, 'shrink': 0.6},
        missing_kwds={'color': 'lightgrey', 'label': 'No data'},
        cmap='OrRd',
        norm=norm,
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=14)

    # Use plain decimal tick labels for percentage colorbar
    if 'pct' in col:
        cbar = fig.axes[-1]  # colorbar axis
        cbar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}%'))

    fig.savefig(os.path.join(scriptdir, 'output', fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Map saved to output/{fname}")
