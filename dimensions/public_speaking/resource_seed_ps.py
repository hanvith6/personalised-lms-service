"""Seed learning-resource library for the Public Speaking dimension.

Module 9 §3.6, scoped to Public Speaking. Real, public resources tagged by
skill / resource_type / difficulty. URLs are stable, well-known public pages.
"""
from __future__ import annotations

from utils.taxonomy import Resource

PS_RESOURCES: list[Resource] = [
    Resource("ps-eye-01", "Julian Treasure: How to speak so that people want to listen",
             "eye_contact", "video", "beginner",
             "https://www.ted.com/talks/julian_treasure_how_to_speak_so_that_people_want_to_listen", 20),
    Resource("ps-eye-02", "Maintaining eye contact while presenting (practice drill)",
             "eye_contact", "practice", "beginner",
             "https://www.toastmasters.org/resources/public-speaking-tips/eye-contact", 15),
    Resource("ps-post-01", "Amy Cuddy: Your body language may shape who you are",
             "posture", "video", "beginner",
             "https://www.ted.com/talks/amy_cuddy_your_body_language_may_shape_who_you_are", 21),
    Resource("ps-post-02", "Posture and stance for presenters (exercise)",
             "posture", "exercise", "intermediate",
             "https://www.toastmasters.org/resources/public-speaking-tips/body-language", 25),
    Resource("ps-pace-01", "Paced reading drill: control your speaking speed",
             "speech_pace", "exercise", "beginner",
             "https://www.write-out-loud.com/speech-rate.html", 15),
    Resource("ps-pace-02", "Pacing and pausing for emphasis",
             "speech_pace", "article", "intermediate",
             "https://www.toastmasters.org/resources/public-speaking-tips/vocal-variety", 12),
    Resource("ps-voice-01", "Roger Love: Vocal Exercises and Voice Training",
             "voice_stability", "video", "beginner",
             "https://www.youtube.com/watch?v=I2H-BO4OZZY", 20),
    Resource("ps-voice-02", "Reduce vocal jitter: steady-tone practice",
             "voice_stability", "practice", "intermediate",
             "https://www.write-out-loud.com/how-to-improve-your-speaking-voice.html", 18),
    Resource("ps-open-01", "Nancy Duarte: The secret structure of great talks",
             "opening_closing_impact", "video", "intermediate",
             "https://www.ted.com/talks/nancy_duarte_the_secret_structure_of_great_talks", 18),
    Resource("ps-open-02", "How to open and close a speech with impact",
             "opening_closing_impact", "article", "beginner",
             "https://www.write-out-loud.com/how-to-start-a-speech.html", 14),
    Resource("ps-open-03", "Opening/closing impact self-check quiz",
             "opening_closing_impact", "quiz", "beginner",
             "https://www.toastmasters.org/resources/public-speaking-tips", 10),
    Resource("ps-slide-01", "David JP Phillips: How to avoid death by PowerPoint",
             "slide_structure", "video", "beginner",
             "https://www.youtube.com/watch?v=Iwpi1Lm6dFo", 20),
    Resource("ps-slide-02", "Presentation structure template (signposting)",
             "slide_structure", "article", "intermediate",
             "https://www.write-out-loud.com/speech-outline.html", 15),
    Resource("ps-aud-01", "Audience engagement techniques for speakers",
             "audience_engagement", "article", "intermediate",
             "https://www.toastmasters.org/resources/public-speaking-tips/audience-engagement", 12),
    Resource("ps-aud-02", "Group discussion & Q&A practice prompts",
             "audience_engagement", "practice", "advanced",
             "https://www.write-out-loud.com/impromptu-speaking-topics.html", 20),
]
