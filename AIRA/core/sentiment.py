"""
core/sentiment.py

Sentiment analysis engine using VADER (Valence Aware Dictionary and sEntiment Reasoner).
Analyses user input and returns an emotional profile used to adapt AIRA's tone.

Emotional states mapped:
    joy         — very positive, high compound score
    content     — mildly positive
    neutral     — balanced or factual input
    anxious     — negative with question patterns
    sad         — negative, low energy
    angry       — strongly negative, high intensity
    frustrated  — mildly negative, repeated or terse input
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Thresholds for compound score bucketing
VERY_POSITIVE  =  0.55
MILD_POSITIVE  =  0.15
MILD_NEGATIVE  = -0.15
VERY_NEGATIVE  = -0.55


class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyse(self, text: str) -> dict:
        """
        Analyse the emotional content of a text string.

        Returns a dict with:
            compound    float   overall sentiment score (-1 to +1)
            positive    float   proportion of positive sentiment
            negative    float   proportion of negative sentiment
            neutral     float   proportion of neutral sentiment
            emotion     str     mapped emotional label
            intensity   str     'high' | 'medium' | 'low'
            tone_hint   str     guidance string passed to the LLM prompt
        """
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        emotion = self._map_emotion(text, scores)
        intensity = self._intensity(compound)
        tone_hint = self._tone_hint(emotion)

        return {
            "compound":  round(compound, 4),
            "positive":  round(scores["pos"], 4),
            "negative":  round(scores["neg"], 4),
            "neutral":   round(scores["neu"], 4),
            "emotion":   emotion,
            "intensity": intensity,
            "tone_hint": tone_hint,
        }

    def _map_emotion(self, text: str, scores: dict) -> str:
        compound = scores["compound"]
        neg      = scores["neg"]
        text_lower = text.lower().strip()

        # Check for question patterns — can signal anxiety even with neutral score
        has_question = bool(re.search(r"\?|how do i|what if|am i|will i|should i|can i", text_lower))

        if compound >= VERY_POSITIVE:
            return "joy"
        elif compound >= MILD_POSITIVE:
            return "content"
        elif compound <= VERY_NEGATIVE:
            if neg > 0.4:
                return "angry"
            return "sad"
        elif compound <= MILD_NEGATIVE:
            if has_question:
                return "anxious"
            if len(text.split()) <= 4:
                return "frustrated"
            return "sad"
        else:
            if has_question:
                return "anxious"
            return "neutral"

    def _intensity(self, compound: float) -> str:
        abs_score = abs(compound)
        if abs_score >= 0.6:
            return "high"
        elif abs_score >= 0.25:
            return "medium"
        return "low"

    def _tone_hint(self, emotion: str) -> str:
        hints = {
            "joy":        "The user seems happy and energetic. Match their enthusiasm, be warm and upbeat.",
            "content":    "The user is in a positive mood. Keep the tone friendly and encouraging.",
            "neutral":    "The user is calm and factual. Be clear, helpful, and conversational.",
            "anxious":    "The user seems worried or uncertain. Be calm, reassuring, and patient. Avoid overwhelming them.",
            "sad":        "The user seems low or unhappy. Be gentle, empathetic, and supportive. Don't rush them.",
            "angry":      "The user seems frustrated or angry. Stay calm, acknowledge their feelings, and be solution-focused.",
            "frustrated": "The user seems mildly frustrated. Be concise, direct, and practical. Don't add fluff.",
        }
        return hints.get(emotion, "Be helpful and conversational.")

    def display_label(self, emotion: str) -> str:
        """Returns a short visual label for the CLI."""
        labels = {
            "joy":        "😊 joyful",
            "content":    "🙂 content",
            "neutral":    "😐 neutral",
            "anxious":    "😟 anxious",
            "sad":        "😢 sad",
            "angry":      "😠 angry",
            "frustrated": "😤 frustrated",
        }
        return labels.get(emotion, "😐 neutral")
