folder=dp3_demo_weighted_full_dagger
folder=invert_push_reset_only
folder=articubot_all_reset_only_1203
folder=articubot_exps/2026-01-02full-new-pick-place-0101
gcloud storage managed-folders create "gs://cmu-gpucloud-yufeiw2/$folder"
gcloud storage managed-folders set-iam-policy "gs://cmu-gpucloud-yufeiw2/$folder" /project/flame/yufeiw2/RoboGen-sim2real/scripts_orchard/access.json