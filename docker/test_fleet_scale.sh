#!/bin/bash
# docker/test_fleet_scale.sh
# Run a ROS fleet API scale check with the same launcher/checker split as
# test_bridge_api.sh.
set -e

ROBOTS=100
PUBLISH_RATE=5.0
TIMEOUT=60.0
GUI=false
TARGET_RTF=0.0
INTERFACE_MODE=fleet
COMMAND_INTERFACE=fleet
CONFIG_OUT=
GENERATE_ONLY=false
TEMPLATE=
PER_ROBOT_GROUPS=default
VERIFY_MOTION=true
PER_ROBOT_PUBLISH_REPEATS=3
PER_ROBOT_PUBLISH_BATCH_SIZE=0
MEASURE_RTF=false
STATE_PLANE=
MEASURE_TRANSPORT=false
NO_STATE_SUBSCRIPTION=false
PROFILE=false
PROFILING_INTERVAL=1000
FLEET_SERVICE_REPEATS=1
FLEET_TOPIC_REPEATS=1
STATE_QOS_PRESET=fleet_state_reliable
STATE_QOS_RELIABILITY=
STATE_QOS_HISTORY=
STATE_QOS_DEPTH=
STATE_QOS_DURABILITY=
MANAGER_COUNT=0
MANAGER_SUBSCRIPTION_MODE=complete
RTF_WARMUP=1.0
RTF_DURATION=10.0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --robots)
            ROBOTS="$2"
            shift 2
            ;;
        --publish-rate)
            PUBLISH_RATE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --gui)
            GUI=true
            shift
            ;;
        --target-rtf)
            TARGET_RTF="$2"
            shift 2
            ;;
        --interface-mode)
            INTERFACE_MODE="$2"
            shift 2
            ;;
        --command-interface)
            COMMAND_INTERFACE="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --per-robot-groups)
            PER_ROBOT_GROUPS="$2"
            shift 2
            ;;
        --no-verify-motion)
            VERIFY_MOTION=false
            shift
            ;;
        --per-robot-publish-repeats)
            PER_ROBOT_PUBLISH_REPEATS="$2"
            shift 2
            ;;
        --per-robot-publish-batch-size)
            PER_ROBOT_PUBLISH_BATCH_SIZE="$2"
            shift 2
            ;;
        --measure-rtf)
            MEASURE_RTF=true
            shift
            ;;
        --state-plane)
            STATE_PLANE="$2"
            shift 2
            ;;
        --measure-transport)
            MEASURE_TRANSPORT=true
            shift
            ;;
        --no-state-subscription)
            NO_STATE_SUBSCRIPTION=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --profiling-interval)
            PROFILING_INTERVAL="$2"
            shift 2
            ;;
        --fleet-service-repeats)
            FLEET_SERVICE_REPEATS="$2"
            shift 2
            ;;
        --fleet-topic-repeats)
            FLEET_TOPIC_REPEATS="$2"
            shift 2
            ;;
        --state-qos-reliability)
            STATE_QOS_RELIABILITY="$2"
            shift 2
            ;;
        --state-qos-preset)
            STATE_QOS_PRESET="$2"
            shift 2
            ;;
        --state-qos-history)
            STATE_QOS_HISTORY="$2"
            shift 2
            ;;
        --state-qos-depth)
            STATE_QOS_DEPTH="$2"
            shift 2
            ;;
        --state-qos-durability)
            STATE_QOS_DURABILITY="$2"
            shift 2
            ;;
        --manager-count)
            MANAGER_COUNT="$2"
            shift 2
            ;;
        --manager-subscription-mode)
            MANAGER_SUBSCRIPTION_MODE="$2"
            shift 2
            ;;
        --rtf-warmup)
            RTF_WARMUP="$2"
            shift 2
            ;;
        --rtf-duration)
            RTF_DURATION="$2"
            shift 2
            ;;
        --config-out)
            CONFIG_OUT="$2"
            shift 2
            ;;
        --generate-only)
            GENERATE_ONLY=true
            shift
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

echo "=== Fleet API scale check: ${ROBOTS} robots (${INTERFACE_MODE}/${COMMAND_INTERFACE}) ==="

LIB=${BRIDGE_TEST_LIB:-/bridge_test_lib.sh}
if [ ! -f "$LIB" ]; then
    LIB=/docker/bridge_test_lib.sh
fi
if [ ! -f "$LIB" ]; then
    LIB=/docker/docker/bridge_test_lib.sh
fi
if [ ! -f "$LIB" ]; then
    LIB=/opt/pybullet_fleet/docker/bridge_test_lib.sh
fi
source "$LIB"

CHECKER=${FLEET_SCALE_CHECK:-/fleet_scale_check.py}
if [ ! -f "$CHECKER" ]; then
    CHECKER=/docker/fleet_scale_check.py
fi
if [ ! -f "$CHECKER" ]; then
    CHECKER=/docker/docker/fleet_scale_check.py
fi
if [ ! -f "$CHECKER" ]; then
    CHECKER=/opt/pybullet_fleet/docker/fleet_scale_check.py
fi
CONFIG_GENERATOR=${FLEET_SCALE_CONFIG:-/fleet_scale_config.py}
if [ ! -f "$CONFIG_GENERATOR" ]; then
    CONFIG_GENERATOR=/docker/fleet_scale_config.py
fi
if [ ! -f "$CONFIG_GENERATOR" ]; then
    CONFIG_GENERATOR=/docker/docker/fleet_scale_config.py
fi
if [ ! -f "$CONFIG_GENERATOR" ]; then
    CONFIG_GENERATOR=/opt/pybullet_fleet/docker/fleet_scale_config.py
fi

source_ros_env
bridge_repo_root

TMPDIR=$(mktemp -d -t pbf_fleet_scale_XXXXXX)
CONFIG_PATH="$TMPDIR/bridge_fleet_scale.yaml"

cleanup() {
    stop_bridge_node "${STATE_PUBLISHER_PID:-}"
    stop_bridge_node "${BRIDGE_PID:-}"
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

CONFIG_ARGS=(
    --robots "$ROBOTS"
    --target-rtf "$TARGET_RTF"
    --interface-mode "$INTERFACE_MODE"
    --per-robot-groups "$PER_ROBOT_GROUPS"
    --state-qos-preset "$STATE_QOS_PRESET"
    --manager-count "$MANAGER_COUNT"
    --config-out "$CONFIG_PATH"
)
if [ "$GUI" = true ]; then
    CONFIG_ARGS+=(--gui)
fi
if [ -n "$TEMPLATE" ]; then
    CONFIG_ARGS+=(--template "$TEMPLATE")
fi
if [ "$MEASURE_TRANSPORT" = true ]; then
    CONFIG_ARGS+=(--transport-probe)
fi
if [ -n "$STATE_PLANE" ]; then
    CONFIG_ARGS+=(--state-plane "$STATE_PLANE")
fi
if [ "$PROFILE" = true ]; then
    CONFIG_ARGS+=(--enable-time-profiling --profiling-interval "$PROFILING_INTERVAL")
fi
if [ -n "$STATE_QOS_RELIABILITY" ]; then
    CONFIG_ARGS+=(--state-qos-reliability "$STATE_QOS_RELIABILITY")
fi
if [ -n "$STATE_QOS_HISTORY" ]; then
    CONFIG_ARGS+=(--state-qos-history "$STATE_QOS_HISTORY")
fi
if [ -n "$STATE_QOS_DEPTH" ]; then
    CONFIG_ARGS+=(--state-qos-depth "$STATE_QOS_DEPTH")
fi
if [ -n "$STATE_QOS_DURABILITY" ]; then
    CONFIG_ARGS+=(--state-qos-durability "$STATE_QOS_DURABILITY")
fi
echo "--- Phase 1/2: generate bridge config ---"
python3 "$CONFIG_GENERATOR" "${CONFIG_ARGS[@]}"

if [ -n "$CONFIG_OUT" ]; then
    cp "$CONFIG_PATH" "$CONFIG_OUT"
    echo "Wrote generated bridge config: $CONFIG_OUT"
fi

if [ "$GENERATE_ONLY" = true ]; then
    trap - EXIT
    rm -rf "$TMPDIR"
    exit 0
fi

if [ -n "$STATE_PLANE" ]; then
    # A bridge killed by a previous run leaves its segment behind (host IPC namespace).
    rm -f /dev/shm/pbf_state
fi
start_bridge_node "$CONFIG_PATH" "$GUI" "$PUBLISH_RATE"
if [ -n "$STATE_PLANE" ]; then
    # /clock and /fleet/states come from this process, read from shared memory.
    # Run the module rather than the console script so a source tree mounted
    # over an installed package works without rebuilding the image.
    python3 -m pybullet_fleet_ros.state_publisher_node \
        --ros-args -p plane_name:=pbf_state -p mode:="$STATE_PLANE" -p publish_rate:="$PUBLISH_RATE" &
    STATE_PUBLISHER_PID=$!
fi

echo "--- Phase 2/2: run ROS scale checks ---"
if [ "$MANAGER_COUNT" -gt 0 ]; then
    if [ "$COMMAND_INTERFACE" != "fleet" ]; then
        echo "--manager-count requires the default --command-interface fleet; manager scale checks are state-only" >&2
        exit 2
    fi
    COMMAND_INTERFACE=none
fi
CHECK_ARGS=(
    --robots "$ROBOTS"
    --timeout "$TIMEOUT"
    --interface-mode "$INTERFACE_MODE"
    --command-interface "$COMMAND_INTERFACE"
    --fleet-service-repeats "$FLEET_SERVICE_REPEATS"
    --fleet-topic-repeats "$FLEET_TOPIC_REPEATS"
    --state-qos-preset "$STATE_QOS_PRESET"
    --manager-count "$MANAGER_COUNT"
    --manager-subscription-mode "$MANAGER_SUBSCRIPTION_MODE"
    --per-robot-publish-repeats "$PER_ROBOT_PUBLISH_REPEATS"
    --per-robot-publish-batch-size "$PER_ROBOT_PUBLISH_BATCH_SIZE"
)
if [ -n "$STATE_QOS_RELIABILITY" ]; then
    CHECK_ARGS+=(--state-qos-reliability "$STATE_QOS_RELIABILITY")
fi
if [ -n "$STATE_QOS_HISTORY" ]; then
    CHECK_ARGS+=(--state-qos-history "$STATE_QOS_HISTORY")
fi
if [ -n "$STATE_QOS_DEPTH" ]; then
    CHECK_ARGS+=(--state-qos-depth "$STATE_QOS_DEPTH")
fi
if [ -n "$STATE_QOS_DURABILITY" ]; then
    CHECK_ARGS+=(--state-qos-durability "$STATE_QOS_DURABILITY")
fi
if [ "$VERIFY_MOTION" = false ]; then
    CHECK_ARGS+=(--no-verify-motion)
fi
if [ "$MEASURE_RTF" = true ]; then
    CHECK_ARGS+=(--measure-rtf --rtf-warmup "$RTF_WARMUP" --rtf-duration "$RTF_DURATION")
fi
if [ "$MEASURE_TRANSPORT" = true ]; then
    CHECK_ARGS+=(--measure-transport --transport-warmup "$RTF_WARMUP" --transport-duration "$RTF_DURATION")
fi
if [ "$NO_STATE_SUBSCRIPTION" = true ]; then
    CHECK_ARGS+=(--no-state-subscription)
fi

if [ "$MANAGER_SUBSCRIPTION_MODE" = distributed ]; then
    if [ "$MANAGER_COUNT" -le 0 ]; then
        echo "--manager-subscription-mode distributed requires --manager-count" >&2
        exit 2
    fi
    base_manager_robots=$(( ROBOTS / MANAGER_COUNT ))
    manager_remainder=$(( ROBOTS % MANAGER_COUNT ))
    DISTRIBUTED_ARGS=(--state-qos-preset "$STATE_QOS_PRESET")
    if [ -n "$STATE_QOS_RELIABILITY" ]; then
        DISTRIBUTED_ARGS+=(--state-qos-reliability "$STATE_QOS_RELIABILITY")
    fi
    if [ -n "$STATE_QOS_HISTORY" ]; then
        DISTRIBUTED_ARGS+=(--state-qos-history "$STATE_QOS_HISTORY")
    fi
    if [ -n "$STATE_QOS_DEPTH" ]; then
        DISTRIBUTED_ARGS+=(--state-qos-depth "$STATE_QOS_DEPTH")
    fi
    if [ -n "$STATE_QOS_DURABILITY" ]; then
        DISTRIBUTED_ARGS+=(--state-qos-durability "$STATE_QOS_DURABILITY")
    fi
    if [ "$MEASURE_RTF" = true ]; then
        DISTRIBUTED_ARGS+=(--measure-rtf --rtf-warmup "$RTF_WARMUP" --rtf-duration "$RTF_DURATION")
    fi
    if [ "$MEASURE_TRANSPORT" = true ]; then
        DISTRIBUTED_ARGS+=(--measure-transport --transport-warmup "$RTF_WARMUP" --transport-duration "$RTF_DURATION")
    fi
    pids=()
    logs=()
    for ((index = 0; index < MANAGER_COUNT; index++)); do
        endpoint=$(printf '/fleet/manager_%02d/states' "$index")
        manager_robots=$base_manager_robots
        if [ "$index" -lt "$manager_remainder" ]; then
            manager_robots=$((manager_robots + 1))
        fi
        log="$TMPDIR/check_manager_${index}.log"
        logs+=("$log")
        python3 "$CHECKER" \
            --robots "$ROBOTS" --timeout "$TIMEOUT" --interface-mode "$INTERFACE_MODE" \
            --command-interface none --state-endpoints "$endpoint" \
            --expected-state-robots "$manager_robots" "${DISTRIBUTED_ARGS[@]}" \
            >"$log" 2>&1 &
        pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    cat "${logs[@]}"
    if [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
else
    python3 "$CHECKER" "${CHECK_ARGS[@]}"
fi

trap - EXIT
cleanup

echo ""
echo "=== Fleet API scale check PASSED ==="
