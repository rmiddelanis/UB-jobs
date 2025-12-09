import pandas as pd
import os
from unbreakable.misc.helpers import average_over_rp
import shutil
from datetime import datetime

scriptdir = os.path.dirname(__file__)

scenario_cat_info = pd.read_csv(os.path.join(scriptdir, "lib/UB-global-socioeconomic-resilience/results/simulation_output/0_baseline/model_inputs/scenario__cat_info.csv"), index_col=[0, 1])
hazard_prot = pd.read_csv(os.path.join(scriptdir, "lib/UB-global-socioeconomic-resilience/results/simulation_output/0_baseline/model_inputs/scenario__hazard_protection.csv"), index_col=[0, 1])
iah = pd.read_csv(os.path.join(scriptdir, "lib/UB-global-socioeconomic-resilience/results/simulation_output/0_baseline/simulation_outputs/iah.csv"), index_col=[0, 1, 2, 3, 4, 5])

quintile_data = scenario_cat_info[['c', 'income_share', 'diversified_share']]

losses = (iah.dc_short_term - iah.dk_reco + iah.dS_reco_PDS).rename('di').to_frame()
# losses = pd.concat([iah[['dk', 'dc']], (iah.dc_short_term - iah.dk_reco + iah.dS_reco_PDS).rename('di')], axis=1)
losses = losses.mul(iah.n, axis=0).groupby(['iso3', 'hazard', 'rp', 'income_cat']).sum().div(iah.n.groupby(['iso3', 'hazard', 'rp', 'income_cat']).sum(), axis=0)

aal = average_over_rp(losses, hazard_prot.protection, zero_rp=2)
aal['rp'] = 'AAL'
aal = aal.reset_index().set_index(losses.index.names)
losses = pd.concat([losses, aal], axis=0)

all_hazards_loss = losses.groupby(['iso3', 'rp', 'income_cat']).sum()
all_hazards_loss['hazard'] = 'all_hazards'
all_hazards_loss = all_hazards_loss.reset_index().set_index(losses.index.names)
losses = pd.concat([losses, all_hazards_loss], axis=0)

national_losses = losses.groupby(['iso3', 'hazard', 'rp']).mean()
national_losses['income_cat'] = 'tot'
national_losses = national_losses.reset_index().set_index(losses.index.names)
losses = pd.concat([losses, national_losses], axis=0)

losses.sort_index(inplace=True)

os.makedirs(os.path.join(scriptdir, "results"), exist_ok=True)
losses.to_csv(os.path.join(scriptdir, "results", "output_losses.csv"))
scenario_cat_info.to_csv(os.path.join(scriptdir, "results", "quintile_info.csv"))
shutil.copy(os.path.join(scriptdir, "lib/UB-global-socioeconomic-resilience/results/simulation_output/0_baseline/simulation_outputs/results.csv"), os.path.join(scriptdir, "results/macro_results.csv"))

date_str = datetime.now().strftime("%Y-%m-%d")  # e.g., 2025-12-09
zip_name = f"{date_str}_data"
shutil.make_archive(zip_name, "zip", os.path.join(scriptdir, "results"))