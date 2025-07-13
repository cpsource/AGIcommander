#!/bin/bash
# AGIcommander Startup Script
# Simple wrapper for the Python startup script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo -e "${BLUE}"
cat << 'EOF'
    ___   _____ ___________                                         __           
   /   | / ___//  _/ ____/___  ____ ___  ____ ___  ____ _____  ____/ /__  _____
  / /| | \__ \ / // /   / __ \/ __ `__ \/ __ `__ \/ __ `/ __ \/ __  / _ \/ ___/
 / ___ |___/ // // /___/ /_/ / / / / / / / / / / / /_/ / / / / /_/ /  __/ /    
/_/  |_/____/___/\____/\____/_/ /_/ /_/_/ /_/ /_/\__,_/_/ /_/\__,_/\___/_/     
                                                                               
EOF
echo -e "${NC}"

echo -e "${GREEN}🚀 AGIcommander - Autonomous Development Assistant${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Found Python $PYTHON_VERSION"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    print_error "Python 3.8+ is required, found $PYTHON_VERSION"
    exit 1
fi

# Check virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Not running in a virtual environment"
    echo "Consider running: python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Check if running from correct directory
if [[ ! -f "pyproject.toml" ]]; then
    print_error "Please run this script from the AGIcommander root directory"
    exit 1
fi

# Parse command line arguments
MODE="development"
CONFIG="config/startup.yaml"
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --mode MODE       Set startup mode (development|production|autonomous)"
            echo "  -c, --config FILE     Use custom configuration file"
            echo "  -v, --verbose         Enable verbose logging"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Start in development mode"
            echo "  $0 -m production            # Start in production mode"
            echo "  $0 -m autonomous -v         # Start in autonomous mode with verbose logging"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

print_status "Starting AGIcommander in $MODE mode..."

# Create necessary directories
print_status "Creating required directories..."
mkdir -p config logs memory servers/memory/vector_db servers/memory/S3

# Check for required configuration files
if [[ ! -f ".env" ]]; then
    print_warning ".env file not found"
    if [[ -f ".env.example" ]]; then
        print_status "Copying .env.example to .env"
        cp .env.example .env
        print_warning "Please edit .env file with your API keys and configuration"
    fi
fi

# Install/check dependencies
if [[ ! -f "requirements-lock.txt" ]] && [[ -f "pyproject.toml" ]]; then
    print_status "Installing dependencies from pyproject.toml..."
    pip install -e . || {
        print_error "Failed to install dependencies"
        exit 1
    }
fi

# Start the Python application
print_status "Launching AGIcommander..."
echo ""

# Handle interruption gracefully
trap 'echo -e "\n${YELLOW}Shutting down AGIcommander...${NC}"; exit 0' SIGINT SIGTERM

# Run the Python startup script
python3 scripts/startup.py --mode "$MODE" --config "$CONFIG" $VERBOSE || {
    print_error "AGIcommander startup failed"
    exit 1
}

print_success "AGIcommander shutdown complete"

