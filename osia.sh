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
    "osia-persona-daemon.service"
    "osia-rss-ingress.service"
    "osia-mcp-arxiv-bridge.service"
    "osia-mcp-phone-bridge.service"
    "osia-mcp-semantic-scholar-bridge.service"
    "osia-mcp-tavily-bridge.service"
    "osia-mcp-time-bridge.service"
    "osia-mcp-wikipedia-bridge.service"
    "osia-cyber-bridge.service"
)

TIMERS=(
    "osia-daily-sitrep.timer"
    "osia-rss-ingress.timer"
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
            local cpu_temp=$(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 ))
            local temp_color="$GREEN"
            [ "$cpu_temp" -ge 70 ] && temp_color="$YELLOW"
            [ "$cpu_temp" -ge 85 ] && temp_color="$RED"
            echo -e "  CPU Temp:  ${temp_color}${cpu_temp}°C${NC}"
        fi

        # Memory
        local mem_total mem_used mem_pct
        mem_total=$(free -m | awk '/^Mem:/{print $2}')
        mem_used=$(free -m | awk '/^Mem:/{print $3}')
        if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null; then
            mem_pct=$(( mem_used * 100 / mem_total ))
            local mem_color="$GREEN"
            [ "$mem_pct" -ge 75 ] && mem_color="$YELLOW"
            [ "$mem_pct" -ge 90 ] && mem_color="$RED"
            echo -e "  Memory:    ${mem_color}${mem_used}MB / ${mem_total}MB (${mem_pct}%)${NC}"
        fi

        # Disk usage for project root
        local disk_info
        disk_info=$(df -h "$SCRIPT_DIR" 2>/dev/null | tail -1)
        if [ -n "$disk_info" ]; then
            local disk_use disk_avail disk_pct
            disk_use=$(echo "$disk_info" | awk '{print $3}')
            disk_avail=$(echo "$disk_info" | awk '{print $4}')
            disk_pct=$(echo "$disk_info" | awk '{print $5}' | tr -d '%')
            local disk_color="$GREEN"
            [ "$disk_pct" -ge 80 ] && disk_color="$YELLOW"
            [ "$disk_pct" -ge 95 ] && disk_color="$RED"
            echo -e "  Disk:      ${disk_color}${disk_use} used, ${disk_avail} free (${disk_pct}%)${NC}"
        fi

        # GPU (nvidia-smi if available)
        if command -v nvidia-smi &>/dev/null; then
            local gpu_info
            gpu_info=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
            if [ -n "$gpu_info" ]; then
                local gpu_name gpu_temp gpu_util gpu_mem_used gpu_mem_total
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
            local missing_vars=()
            local required_vars=("GEMINI_API_KEY" "SIGNAL_SENDER_NUMBER" "ANYTHINGLLM_API_KEY")
            for var in "${required_vars[@]}"; do
                local val
                val=$(grep -E "^${var}=" "$SCRIPT_DIR/.env" 2>/dev/null | head -1 | cut -d'=' -f2-)
                if [ -z "$val" ] || [[ "$val" == *"your_"*"_here"* ]]; then
                    missing_vars+=("$var")
                fi
            done

            if [ ${#missing_vars[@]} -eq 0 ]; then
                echo -e "[${GREEN}OK${NC}]   Required env vars set (${required_vars[*]})"
            else
                echo -e "[${RED}FAIL${NC}] Missing or placeholder env vars: ${missing_vars[*]}"
            fi
        else
            echo -e "[${RED}FAIL${NC}] .env file not found — copy .env.example and configure it"
        fi

        if [ -f "$SCRIPT_DIR/config/feeds.txt" ]; then
            local feed_count
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

        # --- API & Component Health ---
        section_header "API HEALTH"
        check_api_health "AnythingLLM" "http://localhost:3001"
        check_api_health "Qdrant" "http://localhost:6333"
        check_api_health "Signal API" "http://localhost:8081/v1/about"

        # Redis ping
        if docker exec osia-redis redis-cli ping 2>/dev/null | grep -q PONG; then
            # Also grab queue depth
            local queue_len
            queue_len=$(docker exec osia-redis redis-cli LLEN osia:task_queue 2>/dev/null | tr -d '\r')
            echo -e "[${GREEN}OK${NC}]   Redis PONG ${DIM}(task queue depth: ${queue_len:-0})${NC}"
        else
            echo -e "[${RED}FAIL${NC}] Redis is unresponsive"
        fi

        # MCP bridges (check if any are listening)
        for port_name in "8090:ArXiv MCP" "8091:Wikipedia MCP" "8092:Tavily MCP" "8093:Semantic Scholar MCP" "8094:Time MCP"; do
            local port="${port_name%%:*}"
            local name="${port_name##*:}"
            check_api_health "$name" "http://localhost:${port}/sse" 2
        done

        # Phone bridge
        check_api_health "Phone Bridge" "http://localhost:8095/health" 2

        # --- ADB / Phone Gateway ---
        section_header "ADB DEVICES"
        check_adb_device

        # --- Redis Queue Snapshot ---
        section_header "TASK QUEUE"
        if docker exec osia-redis redis-cli ping 2>/dev/null | grep -q PONG; then
            local queue_depth
            queue_depth=$(docker exec osia-redis redis-cli LLEN osia:task_queue 2>/dev/null | tr -d '\r')
            echo -e "  Pending tasks: ${BOLD}${queue_depth:-0}${NC}"

            if [ "${queue_depth:-0}" -gt 0 ]; then
                echo -e "  ${DIM}Next task preview:${NC}"
                local next_task
                next_task=$(docker exec osia-redis redis-cli LINDEX osia:task_queue 0 2>/dev/null | head -c 200)
                echo -e "  ${DIM}${next_task}${NC}"
            fi
        else
            echo -e "  ${DIM}Redis unavailable — cannot inspect queue${NC}"
        fi

        echo ""
        ;;
    logs)
        # Quick access to recent logs from key services
        local target="${2:-osia-orchestrator.service}"
        echo -e "${YELLOW}Showing last 50 lines for ${target}...${NC}"
        journalctl -u "$target" -n 50 --no-pager
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [service]}"
        exit 1
        ;;
esac
