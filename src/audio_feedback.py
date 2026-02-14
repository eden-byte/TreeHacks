"""
Audio Feedback System (disabled)
Kept for API compatibility — audio is disabled for the Raspberry Pi build.
"""

from typing import List, Dict
import time


class AudioFeedback:
    def __init__(self, config: dict):
        """Audio disabled stub initialization."""
        # keep API compatible but do not initialize audio engines/threads
        self.config = config
        self.enabled = False
        self.rate = config.get('audio', {}).get('rate', 180)
        self.volume = config.get('audio', {}).get('volume', 1.0)
        self.last_announcement = {}
        self.announcement_cooldown = 2.0

    
    def _audio_worker(self):
        """No-op worker (audio disabled)."""
        return
    
    def announce_detections(self, detections: List[Dict]):
        """
        Announce detected objects with priority and distance
        
        Args:
            detections: List of detection dicts with keys:
                - class_name: str
                - confidence: float
                - distance_zone: str (danger/warning/info)
                - position: str (left/center/right)
        """
        if not self.enabled or not detections:
            return
        
        current_time = time.time()
        priority_detections = []
        
        # Filter priority objects in danger zone
        for det in detections:
            class_name = det['class_name']
            zone = det['distance_zone']
            
            # Check cooldown for this object type
            last_time = self.last_announcement.get(class_name, 0)
            if current_time - last_time < self.announcement_cooldown:
                continue
            
            if zone == 'danger' and class_name in self.config['detection']['priority_classes']:
                priority_detections.append(det)
                self.last_announcement[class_name] = current_time
        
        if priority_detections:
            # Announce most critical detection (danger zone)
            det = priority_detections[0]
            message = f"Warning! {det['class_name']} ahead on your {det['position']}"
            self.message_queue.put(message)
        elif detections:
            # Announce general detections periodically (warning zone)
            det = detections[0]
            if det['distance_zone'] == 'warning':
                last_time = self.last_announcement.get(det['class_name'], 0)
                if current_time - last_time >= self.announcement_cooldown:
                    message = f"{det['class_name']} detected {det['position']}"
                    self.message_queue.put(message)
                    self.last_announcement[det['class_name']] = current_time
    
    def announce_custom(self, message: str):
        """Announce a custom message"""
        if self.enabled:
            self.message_queue.put(message)
    
    def stop(self):
        """Stop the audio engine"""
        if self.enabled:
            self.engine.stop()