#!/bin/bash

# OSIA Global Management Script
# Usage: ./osia.sh {start|stop|restart|status}

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
)

check_docker_container() {
    local container=$1
    local state=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)
    if [ "$state" == "running" ]; then
        echo -e "[${GREEN}OK${NC}] Docker: $container is running"
    else
        echo -e "[${RED}FAIL${NC}] Docker: $container is ${state:-missing}"
    fi
}

check_systemd_service() {
    local service=$1
    if systemctl is-active --quiet "$service"; then
        echo -e "[${GREEN}OK${NC}] Systemd: $service is active"
    else
        echo -e "[${RED}FAIL${NC}] Systemd: $service is inactive/failed"
    fi
}

check_api_health() {
    local name=$1
    local url=$2
    if curl -s -f -o /dev/null "$url"; then
        echo -e "[${GREEN}OK${NC}] API: $name is reachable ($url)"
    else
        echo -e "[${RED}FAIL${NC}] API: $name is unreachable ($url)"
    fi
}

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
        echo -e "\n${YELLOW}=== DOCKER CONTAINERS ===${NC}"
        for c in "${CONTAINERS[@]}"; do
            check_docker_container "$c"
        done

        echo -e "\n${YELLOW}=== SYSTEMD SERVICES & TIMERS ===${NC}"
        for s in "${SERVICES[@]}" "${TIMERS[@]}"; do
            check_systemd_service "$s"
        done

        echo -e "\n${YELLOW}=== API & COMPONENT HEALTH CHECKS ===${NC}"
        
        # AnythingLLM Frontend/API
        check_api_health "AnythingLLM" "http://localhost:3001"
        
        # Qdrant Vector DB
        check_api_health "Qdrant" "http://localhost:6333"
        
        # Signal API
        check_api_health "Signal API" "http://localhost:8081/v1/about"
        
        # Redis
        if docker exec osia-redis redis-cli ping | grep -q PONG 2>/dev/null; then
            echo -e "[${GREEN}OK${NC}] API: Redis is responding to pings"
        else
            echo -e "[${RED}FAIL${NC}] API: Redis is unresponsive"
        fi
        
        echo ""
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
esac