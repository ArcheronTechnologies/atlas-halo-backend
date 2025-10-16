#!/bin/bash
# Halo Backend - Monitoring Setup Script
# This script helps you set up Sentry and UptimeRobot monitoring

set -e

echo "🔧 Halo Backend Monitoring Setup"
echo "================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Container ID
CONTAINER_ID="35a73370-0199-42de-862c-88b67af1890d"
BACKEND_URL="https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud"

echo "📋 Step 1: Sentry Setup"
echo "----------------------"
echo ""
echo "Please follow these steps:"
echo "1. Go to https://sentry.io/signup/"
echo "2. Create an account (free)"
echo "3. Create a new project named 'halo-backend'"
echo "4. Copy the DSN that looks like:"
echo "   https://abc123...@o123456.ingest.sentry.io/123456"
echo ""
read -p "Do you have your Sentry DSN ready? (y/n): " has_dsn

if [ "$has_dsn" = "y" ] || [ "$has_dsn" = "Y" ]; then
    echo ""
    read -p "Enter your Sentry DSN: " sentry_dsn

    if [ -z "$sentry_dsn" ]; then
        echo -e "${RED}❌ Error: DSN cannot be empty${NC}"
        exit 1
    fi

    echo ""
    echo "Adding SENTRY_DSN to Scaleway container..."

    scw container container update $CONTAINER_ID \
        environment-variables.SENTRY_DSN="$sentry_dsn"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Sentry DSN added successfully${NC}"

        echo ""
        echo "Deploying updated container..."
        scw container container deploy $CONTAINER_ID

        echo ""
        echo "⏳ Waiting 40 seconds for deployment..."
        sleep 40

        echo ""
        echo "Testing backend health..."
        response=$(curl -s "$BACKEND_URL/health")

        if echo "$response" | grep -q "healthy"; then
            echo -e "${GREEN}✅ Backend is healthy!${NC}"
            echo "Response: $response"

            echo ""
            echo "🎉 Sentry is now configured!"
            echo "Check your Sentry dashboard at https://sentry.io/"
        else
            echo -e "${YELLOW}⚠️  Backend responded but may not be healthy${NC}"
            echo "Response: $response"
        fi
    else
        echo -e "${RED}❌ Failed to update container${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⏭️  Skipping Sentry setup${NC}"
    echo "You can set it up later using SENTRY_SETUP_GUIDE.md"
fi

echo ""
echo "================================="
echo ""
echo "📊 Step 2: UptimeRobot Setup"
echo "----------------------------"
echo ""
echo "UptimeRobot monitors your API and alerts you when it goes down."
echo ""
echo "Please follow these steps:"
echo "1. Go to https://uptimerobot.com/"
echo "2. Sign up for a free account"
echo "3. Click '+ Add New Monitor'"
echo "4. Configure the monitor:"
echo ""
echo -e "   ${GREEN}Monitor Type:${NC} HTTP(s)"
echo -e "   ${GREEN}Name:${NC} Halo Backend Health"
echo -e "   ${GREEN}URL:${NC} $BACKEND_URL/health"
echo -e "   ${GREEN}Interval:${NC} 5 minutes"
echo -e "   ${GREEN}Keyword:${NC} healthy (should exist)"
echo ""
echo "5. Set up email alerts in 'Alert Contacts'"
echo "6. Save the monitor"
echo ""

read -p "Have you completed UptimeRobot setup? (y/n): " uptimerobot_done

if [ "$uptimerobot_done" = "y" ] || [ "$uptimerobot_done" = "Y" ]; then
    echo -e "${GREEN}✅ UptimeRobot setup complete!${NC}"
else
    echo -e "${YELLOW}⏭️  You can set up UptimeRobot later${NC}"
    echo "Refer to UPTIMEROBOT_SETUP_GUIDE.md for detailed instructions"
fi

echo ""
echo "================================="
echo ""
echo "🎉 Monitoring Setup Summary"
echo "============================"
echo ""

if [ "$has_dsn" = "y" ] || [ "$has_dsn" = "Y" ]; then
    echo -e "✅ Sentry error tracking: ${GREEN}CONFIGURED${NC}"
else
    echo -e "⏭️  Sentry error tracking: ${YELLOW}PENDING${NC}"
fi

if [ "$uptimerobot_done" = "y" ] || [ "$uptimerobot_done" = "Y" ]; then
    echo -e "✅ UptimeRobot monitoring: ${GREEN}CONFIGURED${NC}"
else
    echo -e "⏭️  UptimeRobot monitoring: ${YELLOW}PENDING${NC}"
fi

echo ""
echo "📚 Documentation:"
echo "   - Sentry Setup: SENTRY_SETUP_GUIDE.md"
echo "   - UptimeRobot Setup: UPTIMEROBOT_SETUP_GUIDE.md"
echo "   - Production Status: PRODUCTION_READY_STATUS.md"
echo ""
echo "🔗 Quick Links:"
echo "   - Backend Health: $BACKEND_URL/health"
echo "   - API Docs: $BACKEND_URL/docs"
echo "   - Sentry: https://sentry.io/"
echo "   - UptimeRobot: https://uptimerobot.com/"
echo ""
echo "✨ Done! Your backend monitoring is ready."
