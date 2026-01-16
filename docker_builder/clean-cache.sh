#!/bin/bash
# Clean Docker BuildKit cache

echo "=========================================="
echo "Docker BuildKit Cache Cleanup"
echo "=========================================="
echo ""

# Show current cache usage
echo "Current BuildKit cache usage:"
docker buildx du 2>/dev/null || docker system df

echo ""
echo "Options:"
echo "  1) Clean only PyTorch build cache (pytorch-cache)"
echo "  2) Clean all BuildKit caches"
echo "  3) Cancel"
echo ""

read -p "Select option (1-3): " -n 1 -r
echo

case $REPLY in
    1)
        echo "Cleaning PyTorch build cache..."
        docker builder prune --filter id=pytorch-cache -f
        echo ""
        echo "✓ PyTorch cache cleaned"
        echo "Next build will re-download PyTorch source"
        ;;
    2)
        echo "Cleaning all BuildKit caches..."
        docker builder prune -af
        echo ""
        echo "✓ All caches cleaned"
        ;;
    *)
        echo "Cancelled"
        ;;
esac
