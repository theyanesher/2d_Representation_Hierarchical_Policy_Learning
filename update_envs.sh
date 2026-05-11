
## This assumes you are using conda

#!/usr/bin/env bash

# Get the site-packages dir for the active env
SITE_PACKAGES=$(python - <<'PYCODE'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PYCODE
)

cd $SITE_PACKAGES/robomimic/envs/
mv env_robosuite.py env_robosuite_orig.py
ln -s /project_data/held/$USER/articubot-on-mimicgen/env_robosuite.py env_robosuite.py

cd $SITE_PACKAGES/robomimic/utils/
mv env_utils.py env_utils_orig.py
ln -s /project_data/held/$USER/articubot-on-mimicgen/env_utils.py env_utils.py
