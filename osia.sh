#!/bin/bash

# OSIA Global Management Script
# Usage: ./osia.sh {start|stop|restart|status|logs}

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICES=(
    "osia-orchestrator.service"
    "osia-signal-ingress.service"
    "osia-rss-ingress.service"
    "osia-mcp-arxiv-bridge.service"
    "osia-mcp-phone-bridge.service"
    "osia-mcp-semantic-scholar-bridge.service"
    "osia-mcp-tavily-bridge.service"
    "osia-mcp-time-bridge.service"
    "osia-mcp-wikipedia-bridge.service"
    "osia-cyber-bridge.service"
    "osia-status-api.service"
    "osia-queue-api.service"
)

TIMERS=(
    "osia-daily-sitrep.timer"
    "osia-rss-ingress.timer"
    "osia-research-worker.timer"
    "osia-instagram-pool-health.timer"
)

CONTAINERS=(
    "osia-anythingllm"
    "osia-qdrant"
    "osia-redis"
    "osia-signal"
    "mailserver"
    "osia-kali"
)

# --- Helper Functions ---

section_header() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

check_docker_container() {
    local container=$1
    local state
    state=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)
    local uptime
    uptime=$(docker inspect -f '{{.State.StartedAt}}' "$container" 2>/dev/null)

    if [ "$state" == "running" ]; then
        local since=""
        if [ -n "$uptime" ] && [ "$uptime" != "<no value>" ]; then
            since=" ${DIM}(since $(date -d "$uptime" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "$uptime"))${NC}"
        fi
        echo -e "[${GREEN}OK${NC}]   Docker: $container${since}"
    else
        echo -e "[${RED}FAIL${NC}] Docker: $container is ${state:-not found}"
    fi
}

check_systemd_service() {
    local service=$1
    local active_state
    active_state=$(systemctl is-active "$service" 2>/dev/null)

    if [ "$active_state" == "active" ]; then
        local mem=""
        local pid=""
        pid=$(systemctl show -p MainPID --value "$service" 2>/dev/null)
        if [ -n "$pid" ] && [ "$pid" != "0" ]; then
            local rss
            rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ -n "$rss" ]; then
                mem=" ${DIM}(PID $pid, $(( rss / 1024 ))MB RSS)${NC}"
            fi
        fi
        echo -e "[${GREEN}OK${NC}]   $service${mem}"
    else
        echo -e "[${RED}FAIL${NC}] $service ($active_state)"
    fi
}

check_timer() {
    local timer=$1
    local active_state
    active_state=$(systemctl is-active "$timer" 2>/dev/null)

    if [ "$active_state" == "active" ]; then
        local next
        next=$(systemctl show -p NextElapseUSecRealtime --value "$timer" 2>/dev/null)
        local last
        last=$(systemctl show -p LastTriggerUSec --value "$timer" 2>/dev/null)
        local info=""
        if [ -n "$next" ] && [ "$next" != "n/a" ]; then
            info=" ${DIM}(next: $next)${NC}"
        fi
        echo -e "[${GREEN}OK${NC}]   $timer${info}"
    else
        echo -e "[${RED}FAIL${NC}] $timer ($active_state)"
    fi
}

check_api_health() {
    local name=$1
    local url=$2
    local timeout=${3:-3}
    local http_code
    http_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout "$timeout" "$url" 2>/dev/null)

    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 400 ] 2>/dev/null; then
        echo -e "[${GREEN}OK${NC}]   $name ${DIM}($url → HTTP $http_code)${NC}"
    else
        echo -e "[${RED}FAIL${NC}] $name ${DIM}($url → ${http_code:-timeout})${NC}"
    fi
}

check_adb_device() {
    if ! command -v adb &>/dev/null; then
        echo -e "[${YELLOW}SKIP${NC}] adb not found in PATH"
        return
    fi

    local devices
    devices=$(adb devices 2>/dev/null | tail -n +2 | grep -v '^$')

    if [ -z "$devices" ]; then
        echo -e "[${RED}FAIL${NC}] No ADB devices connected"
        return
    fi

    while IFS= read -r line; do
        local serial state
        serial=$(echo "$line" | awk '{print $1}')
        state=$(echo "$line" | awk '{print $2}')

        if [ "$state" == "device" ]; then
            local model brand
            model=$(adb -s "$serial" shell getprop ro.product.model 2>/dev/null | tr -d '\r')
            brand=$(adb -s "$serial" shell getprop ro.product.brand 2>/dev/null | tr -d '\r')
            local battery
            battery=$(adb -s "$serial" shell dumpsys battery 2>/dev/null | grep 'level:' | awk '{print $2}' | tr -d '\r')
            local bat_info=""
            if [ -n "$battery" ]; then
                if [ "$battery" -le 15 ] 2>/dev/null; then
                    bat_info=" ${RED}🔋 ${battery}%${NC}"
                elif [ "$battery" -le 40 ] 2>/dev/null; then
                    bat_info=" ${YELLOW}🔋 ${battery}%${NC}"
                else
                    bat_info=" 🔋 ${battery}%"
                fi
            fi
            echo -e "[${GREEN}OK${NC}]   $serial — ${brand} ${model}${bat_info}"
        elif [ "$state" == "unauthorized" ]; then
            echo -e "[${YELLOW}WARN${NC}] $serial — unauthorized (accept USB debugging prompt on device)"
        else
            echo -e "[${RED}FAIL${NC}] $serial — $state"
        fi
    done <<< "$devices"
}

# --- Main ---

command=$1

case $command in
    start)
        echo -e "${YELLOW}Starting OSIA Docker containers...${NC}"
        docker start "${CONTAINERS[@]}"

        echo -e "${YELLOW}Starting OSIA Systemd services...${NC}"
        sudo systemctl start "${SERVICES[@]}" "${TIMERS[@]}"

        echo -e "${GREEN}OSIA successfully started.${NC}"
        ;;
    stop)
        echo -e "${YELLOW}Stopping OSIA Systemd services...${NC}"
        sudo systemctl stop "${SERVICES[@]}" "${TIMERS[@]}"

        echo -e "${YELLOW}Stopping OSIA Docker containers...${NC}"
        docker stop "${CONTAINERS[@]}"

        echo -e "${GREEN}OSIA successfully stopped.${NC}"
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    status)
        echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${CYAN}║         OSIA FRAMEWORK STATUS            ║${NC}"
        echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}"
        echo -e "${DIM}$(date '+%Y-%m-%d %H:%M:%S %Z')${NC}"

        # --- System Overview ---
        section_header "SYSTEM"
        echo -e "  Hostname:  $(hostname)"
        echo -e "  Kernel:    $(uname -r)"
        echo -e "  Uptime:    $(uptime -p 2>/dev/null || uptime | sed 's/.*up /up /' | sed 's/,.*load.*//')"
        echo -e "  Load:      $(cat /proc/loadavg 2>/dev/null | awk '{print $1, $2, $3}' || uptime | sed 's/.*load average: //')"

        # CPU temp (common on ARM SBCs like Orange Pi)
        if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
            cpu_temp=$(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 ))
            temp_color="$GREEN"
            [ "$cpu_temp" -ge 70 ] && temp_color="$YELLOW"
            [ "$cpu_temp" -ge 85 ] && temp_color="$RED"
            echo -e "  CPU Temp:  ${temp_color}${cpu_temp}°C${NC}"
        fi

        # Memory
        mem_total=$(free -m | awk '/^Mem:/{print $2}')
        mem_used=$(free -m | awk '/^Mem:/{print $3}')
        if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null; then
            mem_pct=$(( mem_used * 100 / mem_total ))
            mem_color="$GREEN"
            [ "$mem_pct" -ge 75 ] && mem_color="$YELLOW"
            [ "$mem_pct" -ge 90 ] && mem_color="$RED"
            echo -e "  Memory:    ${mem_color}${mem_used}MB / ${mem_total}MB (${mem_pct}%)${NC}"
        fi

        # Disk usage for project root
        disk_info=$(df -h "$SCRIPT_DIR" 2>/dev/null | tail -1)
        if [ -n "$disk_info" ]; then
            disk_use=$(echo "$disk_info" | awk '{print $3}')
            disk_avail=$(echo "$disk_info" | awk '{print $4}')
            disk_pct=$(echo "$disk_info" | awk '{print $5}' | tr -d '%')
            disk_color="$GREEN"
            [ "$disk_pct" -ge 80 ] && disk_color="$YELLOW"
            [ "$disk_pct" -ge 95 ] && disk_color="$RED"
            echo -e "  Disk:      ${disk_color}${disk_use} used, ${disk_avail} free (${disk_pct}%)${NC}"
        fi

        # GPU (nvidia-smi if available)
        if command -v nvidia-smi &>/dev/null; then
            gpu_info=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
            if [ -n "$gpu_info" ]; then
                IFS=',' read -r gpu_name gpu_temp gpu_util gpu_mem_used gpu_mem_total <<< "$gpu_info"
                gpu_name=$(echo "$gpu_name" | xargs)
                gpu_temp=$(echo "$gpu_temp" | xargs)
                gpu_util=$(echo "$gpu_util" | xargs)
                gpu_mem_used=$(echo "$gpu_mem_used" | xargs)
                gpu_mem_total=$(echo "$gpu_mem_total" | xargs)
                echo -e "  GPU:       ${gpu_name} — ${gpu_temp}°C, ${gpu_util}% util, ${gpu_mem_used}/${gpu_mem_total} MiB VRAM"
            fi
        fi

        # --- Configuration ---
        section_header "CONFIGURATION"
        if [ -f "$SCRIPT_DIR/.env" ]; then
            echo -e "[${GREEN}OK${NC}]   .env file exists"

            # Check critical env vars are set (non-empty)
            missing_vars=()
            required_vars=("GEMINI_API_KEY" "SIGNAL_SENDER_NUMBER" "ANYTHINGLLM_API_KEY")
            for var in "${required_vars[@]}"; do
                val=$(grep -E "^${var}=" "$SCRIPT_DIR/.env" 2>/dev/null | head -1 | cut -d'=' -f2-)
                if [ -z "$val" ] || [[ "$val" == *"your_"*"_here"* ]]; then
                    missing_vars+=("$var")
                fi
            done

            # Optional but recommended (HuggingFace)
            hf_vars=("HF_TOKEN" "HF_NAMESPACE")
            missing_hf=()
            for var in "${hf_vars[@]}"; do
                val=$(grep -E "^${var}=" "$SCRIPT_DIR/.env" 2>/dev/null | head -1 | cut -d'=' -f2-)
                if [ -z "$val" ] || [[ "$val" == *"your_"*"_here"* ]]; then
                    missing_hf+=("$var")
                fi
            done

            if [ ${#missing_vars[@]} -eq 0 ]; then
                echo -e "[${GREEN}OK${NC}]   Required env vars set (${required_vars[*]})"
            else
                echo -e "[${RED}FAIL${NC}] Missing or placeholder env vars: ${missing_vars[*]}"
            fi

            if [ ${#missing_hf[@]} -eq 0 ]; then
                echo -e "[${GREEN}OK${NC}]   HuggingFace config set"
            else
                echo -e "[${YELLOW}WARN${NC}] HF Endpoints not configured (missing: ${missing_hf[*]})"
            fi
        else
            echo -e "[${RED}FAIL${NC}] .env file not found — copy .env.example and configure it"
        fi

        if [ -f "$SCRIPT_DIR/config/feeds.txt" ]; then
            feed_count=$(grep -cve '^\s*$' "$SCRIPT_DIR/config/feeds.txt" 2>/dev/null || echo 0)
            echo -e "[${GREEN}OK${NC}]   RSS feeds config: ${feed_count} feeds"
        else
            echo -e "[${YELLOW}WARN${NC}] config/feeds.txt not found (RSS ingress won't have feeds)"
        fi

        # --- Docker Containers ---
        section_header "DOCKER CONTAINERS"
        if ! command -v docker &>/dev/null; then
            echo -e "[${RED}FAIL${NC}] docker not found in PATH"
        elif ! docker info &>/dev/null 2>&1; then
            echo -e "[${RED}FAIL${NC}] Docker daemon not running or no permission"
        else
            for c in "${CONTAINERS[@]}"; do
                check_docker_container "$c"
            done
        fi

        # --- Systemd Services ---
        section_header "SYSTEMD SERVICES"
        for s in "${SERVICES[@]}"; do
            check_systemd_service "$s"
        done

        # --- Systemd Timers ---
        section_header "SYSTEMD TIMERS"
        for t in "${TIMERS[@]}"; do
            check_timer "$t"
        done

        # --- HuggingFace Endpoints ---
        section_header "HUGGINGFACE ENDPOINTS"
        if [ -f "$SCRIPT_DIR/scripts/provision_hf_endpoints.py" ]; then
            if [ -n "$(grep "HF_TOKEN=" "$SCRIPT_DIR/.env" | cut -d'=' -f2)" ]; then
                # Run status check via python script
                $SCRIPT_DIR/.venv/bin/python "$SCRIPT_DIR/scripts/provision_hf_endpoints.py" --status | sed 's/^/  /'
            else
                echo -e "[${YELLOW}SKIP${NC}] HF_TOKEN not set — cannot check status"
            fi
        else
            echo -e "[${YELLOW}SKIP${NC}] provision_hf_endpoints.py not found"
        fi

        # --- API & Component Health ---
        section_header "API HEALTH"
        check_api_health "AnythingLLM" "http://localhost:3001"
        QDRANT_API_KEY_HEALTH=$(grep -E '^QDRANT_API_KEY=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d'=' -f2-)
        if [ -n "$QDRANT_API_KEY_HEALTH" ]; then
            http_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 -H "api-key: $QDRANT_API_KEY_HEALTH" http://localhost:6333/collections 2>/dev/null)
            if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 400 ] 2>/dev/null; then
                echo -e "[${GREEN}OK${NC}]   Qdrant ${DIM}(http://localhost:6333 → HTTP $http_code)${NC}"
            else
                echo -e "[${RED}FAIL${NC}] Qdrant ${DIM}(http://localhost:6333 → ${http_code:-timeout})${NC}"
            fi
        else
            check_api_health "Qdrant" "http://localhost:6333"
        fi
        check_api_health "Signal API" "http://localhost:8081/v1/about"

        # Redis ping
        if docker exec osia-redis redis-cli ping 2>/dev/null | grep -q PONG; then
            # Also grab queue depth
            queue_len=$(docker exec osia-redis redis-cli LLEN osia:task_queue 2>/dev/null | tr -d '\r')
            echo -e "[${GREEN}OK${NC}]   Redis PONG ${DIM}(task queue depth: ${queue_len:-0})${NC}"
        else
            echo -e "[${RED}FAIL${NC}] Redis is unresponsive"
        fi

        # --- Qdrant Vector DB ---
        section_header "QDRANT KNOWLEDGE BASE"
        QDRANT_API_KEY=$(grep -E '^QDRANT_API_KEY=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d'=' -f2-)
        qdrant_auth=()
        [ -n "$QDRANT_API_KEY" ] && qdrant_auth=(-H "api-key: $QDRANT_API_KEY")

        # Expected desk collections (vectorTag → display name)
        declare -A QDRANT_DESKS=(
            ["collection-directorate"]="Collection Directorate"
            ["geopolitical-and-security-desk"]="Geopolitical & Security"
            ["cultural-and-theological-intelligence-desk"]="Cultural & Theological"
            ["science-technology-and-commercial-desk"]="Science & Tech"
            ["human-intelligence-and-profiling-desk"]="Human Intelligence"
            ["finance-and-economics-directorate"]="Finance & Economics"
            ["cyber-intelligence-and-warfare-desk"]="Cyber Intelligence"
            ["the-watch-floor"]="The Watch Floor"
        )

        qdrant_collections=$(curl -s --connect-timeout 3 "${qdrant_auth[@]}" http://localhost:6333/collections 2>/dev/null)
        if [ -n "$qdrant_collections" ] && echo "$qdrant_collections" | grep -q '"status":"ok"'; then
            # Build a set of existing collection names
            existing_names=$(echo "$qdrant_collections" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)

            total_points=0
            active_count=0
            empty_count=0

            for tag in $(echo "${!QDRANT_DESKS[@]}" | tr ' ' '\n' | sort); do
                desk_label="${QDRANT_DESKS[$tag]}"
                if echo "$existing_names" | grep -qx "$tag"; then
                    cinfo=$(curl -s --connect-timeout 3 "${qdrant_auth[@]}" "http://localhost:6333/collections/${tag}" 2>/dev/null)
                    points=$(echo "$cinfo" | grep -o '"points_count":[0-9]*' | head -1 | cut -d: -f2)
                    vectors=$(echo "$cinfo" | grep -o '"vectors_count":[0-9]*' | head -1 | cut -d: -f2)
                    segments=$(echo "$cinfo" | grep -o '"segments_count":[0-9]*' | head -1 | cut -d: -f2)

                    points=${points:-0}
                    total_points=$((total_points + points))
                    active_count=$((active_count + 1))

                    echo -e "  ${GREEN}●${NC} ${desk_label}: ${BOLD}${points}${NC} points ${DIM}(vectors: ${vectors:-0}, segments: ${segments:-0})${NC}"
                else
                    empty_count=$((empty_count + 1))
                    echo -e "  ${DIM}○ ${desk_label}: no collection yet${NC}"
                fi
            done

            # Show any extra collections not in our expected list
            while IFS= read -r cname; do
                [ -z "$cname" ] && continue
                if [ -z "${QDRANT_DESKS[$cname]+x}" ]; then
                    cinfo=$(curl -s --connect-timeout 3 "${qdrant_auth[@]}" "http://localhost:6333/collections/${cname}" 2>/dev/null)
                    points=$(echo "$cinfo" | grep -o '"points_count":[0-9]*' | head -1 | cut -d: -f2)
                    vectors=$(echo "$cinfo" | grep -o '"vectors_count":[0-9]*' | head -1 | cut -d: -f2)
                    segments=$(echo "$cinfo" | grep -o '"segments_count":[0-9]*' | head -1 | cut -d: -f2)

                    points=${points:-0}
                    total_points=$((total_points + points))
                    active_count=$((active_count + 1))

                    echo -e "  ${CYAN}●${NC} ${cname}: ${BOLD}${points}${NC} points ${DIM}(vectors: ${vectors:-0}, segments: ${segments:-0})${NC}"
                fi
            done <<< "$existing_names"

            echo -e "  ${DIM}─────────────────────────────────────────────${NC}"
            echo -e "  Total: ${BOLD}${active_count}${NC}/${#QDRANT_DESKS[@]} desks active, ${BOLD}${total_points}${NC} points indexed"
        else
            echo -e "  ${DIM}Qdrant unavailable — cannot query collections${NC}"
        fi

        # Phone bridge
        phone_health=$(curl -s --connect-timeout 3 http://localhost:8006/health 2>/dev/null)
        if [ -n "$phone_health" ]; then
            phone_connected=$(echo "$phone_health" | grep -o '"phone_connected":[^,}]*' | cut -d: -f2 | tr -d ' ')
            phone_device=$(echo "$phone_health" | grep -o '"device_id":"[^"]*"' | cut -d'"' -f4)
            phone_configured=$(echo "$phone_health" | grep -o '"bridge_configured":[^,}]*' | cut -d: -f2 | tr -d ' ')
            phone_info="${DIM}(device: ${phone_device:-none}, connected: ${phone_connected}, configured: ${phone_configured})${NC}"
            if [ "$phone_connected" = "true" ]; then
                echo -e "[${GREEN}OK${NC}]   Phone Bridge ${phone_info}"
            else
                echo -e "[${YELLOW}WARN${NC}] Phone Bridge — phone not connected ${phone_info}"
            fi
        else
            echo -e "[${RED}FAIL${NC}] Phone Bridge ${DIM}(http://localhost:8006/health → timeout)${NC}"
        fi

        # --- ADB / Phone Gateway ---
        section_header "ADB DEVICES"
        check_adb_device

        # --- Redis Queue Snapshot ---
        section_header "TASK QUEUE"
        if docker exec osia-redis redis-cli ping 2>/dev/null | grep -q PONG; then
            queue_depth=$(docker exec osia-redis redis-cli LLEN osia:task_queue 2>/dev/null | tr -d '\r')
            research_depth=$(docker exec osia-redis redis-cli LLEN osia:research_queue 2>/dev/null | tr -d '\r')
            echo -e "  Task queue:     ${BOLD}${queue_depth:-0}${NC} pending"
            echo -e "  Research queue: ${BOLD}${research_depth:-0}${NC} pending"

            if [ "${queue_depth:-0}" -gt 0 ]; then
                echo -e "  ${DIM}Next task preview:${NC}"
                next_task=$(docker exec osia-redis redis-cli LINDEX osia:task_queue 0 2>/dev/null | head -c 200)
                echo -e "  ${DIM}${next_task}${NC}"
            fi

            # Queue API health
            QUEUE_API_UA=$(grep -E '^QUEUE_API_UA_SENTINEL=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d'=' -f2-)
            QUEUE_API_UA="${QUEUE_API_UA:-osia-worker/1}"
            queue_api_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 -A "$QUEUE_API_UA" http://localhost:8098/health 2>/dev/null)
            if [ "$queue_api_code" = "200" ]; then
                echo -e "[${GREEN}OK${NC}]   Queue API ${DIM}(http://localhost:8098 → HTTP $queue_api_code)${NC}"
            else
                echo -e "[${RED}FAIL${NC}] Queue API ${DIM}(http://localhost:8098 → ${queue_api_code:-timeout})${NC}"
            fi
        else
            echo -e "  ${DIM}Redis unavailable — cannot inspect queue${NC}"
        fi

        # --- Research Worker ---
        section_header "RESEARCH WORKER"

        check_timer "osia-research-worker.timer"

        # Last run result
        last_run=$(journalctl -u osia-research-worker.service -n 1 --no-pager -q 2>/dev/null | grep -oE "(Batch complete.*|Research Worker starting|error.*)" | head -1)
        [ -n "$last_run" ] && echo -e "  Last run:        ${DIM}${last_run}${NC}"

        # Queue depths
        if docker exec osia-redis redis-cli ping 2>/dev/null | grep -q PONG; then
            research_depth=$(docker exec osia-redis redis-cli LLEN osia:research_queue 2>/dev/null | tr -d '\r')
            seen_count=$(docker exec osia-redis redis-cli SCARD osia:research:seen_topics 2>/dev/null | tr -d '\r')
            echo -e "  Pending topics:  ${BOLD}${research_depth:-0}${NC} queued, ${seen_count:-0} already researched"
        else
            echo -e "  ${DIM}Redis unavailable — cannot inspect research queue${NC}"
        fi

        # OpenRouter key check
        OR_KEY=$(grep -E '^OPENROUTER_API_KEY=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d'=' -f2- | tr -d '[:space:]')
        if [ -n "$OR_KEY" ]; then
            echo -e "  OpenRouter:      ${GREEN}key configured${NC}"
        else
            echo -e "  OpenRouter:      ${RED}OPENROUTER_API_KEY not set — worker will not run${NC}"
        fi

        echo -e "  ${DIM}Logs: journalctl -u osia-research-worker.service -f${NC}"
        echo -e "  ${DIM}Run now: sudo systemctl start osia-research-worker.service${NC}"

        echo ""
        ;;
    logs)
        # Quick access to recent logs from key services
        target="${2:-osia-orchestrator.service}"
        echo -e "${YELLOW}Showing last 50 lines for ${target}...${NC}"
        journalctl -u "$target" -n 50 --no-pager
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [service]}"
        exit 1
        ;;
esac
