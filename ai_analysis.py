"""
AI Chart Analysis Module
Uses GPT-4 Vision to analyze stock chart screenshots and provide trading recommendations
"""

import logging
import base64
from typing import Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


@dataclass
class AIRecommendation:
    """AI recommendation from chart analysis"""
    action: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: int  # 0-100
    risk_reward_ratio: float
    pattern_detected: str
    analysis_text: str


class AIChartAnalyzer:
    """AI-powered chart analysis using GPT-4 Vision"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-4-vision-preview"
    
    def analyze_chart(self, chart_path: str, symbol: str, 
                     current_price: float) -> Optional[AIRecommendation]:
        """Analyze chart screenshot using GPT-4 Vision"""
        try:
            image_data = self._encode_image(chart_path)
            if not image_data:
                return self._fallback_analysis(symbol, current_price)
            
            response = self._call_openai_api(image_data, symbol, current_price)
            if response:
                return self._parse_ai_response(response, symbol, current_price)
            
            return self._fallback_analysis(symbol, current_price)
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return self._fallback_analysis(symbol, current_price)
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64"""
        try:
            if not Path(image_path).exists():
                logger.error(f"Image not found: {image_path}")
                return None
            with open(image_path, "rb") as f:
                return base64.standard_b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def _call_openai_api(self, image_data: str, symbol: str, price: float) -> Optional[Dict]:
        """Call GPT-4 Vision API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            prompt = f"""Analyze this {symbol} chart (current price: ₹{price}).
            Provide JSON: {{"action":"BUY/SELL/HOLD", "entry_price":X, "stop_loss":Y, 
            "take_profit":Z, "confidence":0-100, "pattern":"...", "analysis":"..."}}"""
            
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }],
                "max_tokens": 1024
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                   headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return self._extract_json_from_response(response.json()['choices'][0]['message']['content'])
            return None
        except Exception as e:
            logger.error(f"API error: {e}")
            return None
    
    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from response"""
        try:
            import json, re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group(0)) if match else None
        except:
            return None
    
    def _parse_ai_response(self, data: Dict, symbol: str, price: float) -> AIRecommendation:
        """Parse AI response"""
        entry = float(data.get('entry_price', price))
        sl = float(data.get('stop_loss', price * 0.98))
        tp = float(data.get('take_profit', price * 1.05))
        risk = abs(entry - sl)
        rr = abs(tp - entry) / risk if risk > 0 else 0
        return AIRecommendation(
            action=data.get('action', 'HOLD').upper(),
            entry_price=entry, stop_loss=sl, take_profit=tp,
            confidence=int(data.get('confidence', 50)),
            risk_reward_ratio=rr,
            pattern_detected=data.get('pattern', 'Unknown'),
            analysis_text=data.get('analysis', '')
        )
    
    def _fallback_analysis(self, symbol: str, price: float) -> AIRecommendation:
        """Fallback analysis"""
        return AIRecommendation(
            action='HOLD', entry_price=price,
            stop_loss=price * 0.97, take_profit=price * 1.05,
            confidence=30, risk_reward_ratio=1.67,
            pattern_detected='Unknown',
            analysis_text='Fallback analysis'
        )
