#!/bin/bash
"""
Submit Frodobots DPVO Jobs
This script submits DPVO processing jobs for both front and rear cameras.
"""

set -e  # Exit on any error

SCRIPTS_DIR="/scratch/hs5580/citywalker/CityWalker/dataset/frodobots_2k/scripts/dpvo"

echo "🚀 Frodobots DPVO Job Submission"
echo "================================"

# Check if we're in the right directory
if [ ! -f "run_frodobots_dpvo.slurm" ]; then
    echo "❌ Error: run_frodobots_dpvo.slurm not found in current directory"
    echo "Please run this script from: $SCRIPTS_DIR"
    exit 1
fi

# Function to submit job for a specific camera type
submit_camera_job() {
    local camera_type=$1
    echo ""
    echo "📹 Submitting DPVO job for $camera_type camera..."
    
    # Submit the job with camera type environment variable
    job_id=$(sbatch --export=CAMERA_TYPE=$camera_type run_frodobots_dpvo.slurm | awk '{print $4}')
    
    if [ $? -eq 0 ]; then
        echo "✅ Job submitted successfully!"
        echo "   Job ID: $job_id"
        echo "   Camera: $camera_type"
        echo "   Array: 0-199 (200 tasks)"
    else
        echo "❌ Failed to submit job for $camera_type camera"
        return 1
    fi
}

# Check available video counts
echo "📊 Checking available videos..."

# Count front camera videos
front_count=$(find /vast/hs5580/data/frodobots_2k/extracted -name "*_front_camera_seg_*.mp4" -path "*/recordings_converted_2min/*" | wc -l)
echo "   Front camera videos: $front_count"

# Count rear camera videos  
rear_count=$(find /vast/hs5580/data/frodobots_2k/extracted -name "*_rear_camera_seg_*.mp4" -path "*/recordings_converted_2min/*" | wc -l)
echo "   Rear camera videos: $rear_count"

if [ $front_count -eq 0 ] && [ $rear_count -eq 0 ]; then
    echo "❌ Error: No video segments found!"
    echo "Please ensure video splitting has completed successfully."
    exit 1
fi

# Ask user which cameras to process
echo ""
echo "🎛️  Which cameras would you like to process?"
echo "1) Front camera only ($front_count videos)"
echo "2) Rear camera only ($rear_count videos)"  
echo "3) Both cameras ($((front_count + rear_count)) total videos)"
echo "4) Cancel"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        if [ $front_count -gt 0 ]; then
            submit_camera_job "front"
        else
            echo "❌ No front camera videos found"
            exit 1
        fi
        ;;
    2)
        if [ $rear_count -gt 0 ]; then
            submit_camera_job "rear"
        else
            echo "❌ No rear camera videos found"
            exit 1
        fi
        ;;
    3)
        if [ $front_count -gt 0 ]; then
            submit_camera_job "front"
        fi
        
        if [ $rear_count -gt 0 ]; then
            submit_camera_job "rear"  
        fi
        
        if [ $front_count -eq 0 ] && [ $rear_count -eq 0 ]; then
            echo "❌ No videos found for either camera"
            exit 1
        fi
        ;;
    4)
        echo "Cancelled by user"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Job submission completed!"
echo ""
echo "📋 Monitoring commands:"
echo "   Check job status: squeue -u \$USER"
echo "   View logs: tail -f /scratch/hs5580/citywalker/logs/frodobots_dpvo_*_0.out"
echo "   Check progress: find /vast/hs5580/data/frodobots_2k/extracted -name '*.txt' -path '*/dpvo_poses/*' | wc -l"
echo ""
echo "💡 Tips:"
echo "   - Each job processes ~$((front_count / 200)) videos per array task"
echo "   - Estimated completion time: 2-4 hours depending on system load"
echo "   - Output trajectories will be saved in: */dpvo_poses/*.txt"
