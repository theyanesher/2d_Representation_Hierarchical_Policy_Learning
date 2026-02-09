#!/usr/bin/env bash
set -euo pipefail

#################### EDIT THESE ####################
REMOTE_HOST="seuss"   # e.g. yufei@rchi-gpu-0
# REMOTE_BASE="/data/chenyuah/RoboGen-sim2real/data/pick_and_place_cgn/"
REMOTE_BASE="/data/chenyuah/RoboGen-sim2real/data/grasp/"
# LOCAL_BASE="/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/pick_and_place_cgn"
LOCAL_BASE="/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/grasp"
ID_LIST_FILE="ids.txt"      # file with 010015, 010016, ... one per line
####################################################

while read -r ID; do
    [[ -z "$ID" || "$ID" =~ ^# ]] && continue

    echo "=== Processing ID $ID ==="

    REMOTE_ID_DIR="$REMOTE_BASE/$ID"
    LOCAL_ID_DIR="$LOCAL_BASE/$ID"

    name=eval_grasp_0923
    mkdir -p "$LOCAL_ID_DIR/experiment/$name"

    ###########################
    # 1) Sync base_config.yaml
    ###########################
    # echo "  - Syncing base_config.yaml"
    # rsync -av \
    #     "$REMOTE_HOST:$REMOTE_ID_DIR/base_config.yaml" \
    #     "$LOCAL_ID_DIR/"

    ###########################
    # 2) List ALL folders
    ###########################
    REMOTE_EVAL0="$REMOTE_ID_DIR/experiment/$name"

    num=2
    RUN_FOLDERS=$(ssh -n "$REMOTE_HOST" \
        "cd '$REMOTE_EVAL0' && ls -1d */ 2>/dev/null | sort | head -n $num | tr -d '/'")
        # "cd '$REMOTE_EVAL0' && ls -1d */ 2>/dev/null | sort | tr -d '/'")

    if [[ -z "$RUN_FOLDERS" ]]; then
        echo "  ! No folders found in $REMOTE_EVAL0, skipping."
        continue
    fi

    ###########################
    # 3) Loop over all folders
    ###########################
    echo "  - Found runs:" 
    echo "$RUN_FOLDERS" | sed 's/^/    - /'
    for RUN in $RUN_FOLDERS; do
        echo "  - Processing run: $RUN"

        REMOTE_RUN_DIR="$REMOTE_EVAL0/$RUN"
        LOCAL_RUN_DIR="$LOCAL_ID_DIR/experiment/$name/$RUN"

        STATE_REL_PATH="states/state_0.pkl"
        STATE_REL_PATH="stage_length.json"
        STATE_REL_PATH="task_config.yaml"
        STATE_REL_PATH="object_name.yaml"

        # ✅ Don’t let a single ssh failure kill the whole script
        if ! HAS_STATE=$(ssh -n "$REMOTE_HOST" \
            "cd '$REMOTE_RUN_DIR' 2>/dev/null && [ -f '$STATE_REL_PATH' ] && echo yes" ); then
            echo "    ! ssh check failed for $REMOTE_RUN_DIR, skipping."
            continue
        fi

        if [[ "$HAS_STATE" != "yes" ]]; then
            echo "    ! $STATE_REL_PATH not found, skipping."
            continue
        fi

        mkdir -p "$LOCAL_RUN_DIR/states"

        echo "    - Syncing $STATE_REL_PATH"
        # ✅ Don’t let a single rsync failure kill the whole script
        if ! rsync -av \
            "$REMOTE_HOST:$REMOTE_RUN_DIR/$STATE_REL_PATH" \
            "$LOCAL_RUN_DIR/$STATE_REL_PATH"; then
            echo "    ! rsync failed for $RUN, skipping."
            continue
        fi
    done

done < "$ID_LIST_FILE"