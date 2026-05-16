"""
Compatibility wrapper for the video job processor.

New code should import from api.jobs.video_processor.
"""
from api.jobs.video_processor import cancel_video_job, process_video_background, video_jobs

__all__ = ["cancel_video_job", "process_video_background", "video_jobs"]
