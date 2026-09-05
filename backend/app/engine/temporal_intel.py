import re
from datetime import datetime, timedelta
from typing import Tuple

class TemporalIntelligenceEngine:
    """
    Temporal Intelligence Engine for notice2action-ai (TrustExtract-N)
    Resolves relative and absolute deadlines from notice text into calendar dates.
    """

    @staticmethod
    def parse_deadline(raw_deadline_str: str, notice_date_str: str = None) -> Tuple[str, int]:
        """
        Parses raw deadline string (e.g. 'within 15 days of receipt') and notice date into:
        Returns (calculated_deadline_iso_date, days_remaining)
        """
        base_date = datetime.now()
        if notice_date_str:
            try:
                # Try common formats
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %B %Y"):
                    try:
                        base_date = datetime.strptime(notice_date_str.strip(), fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                base_date = datetime.now()

        text = raw_deadline_str.lower().strip()

        # Check for relative days: "within X days" or "in X days"
        days_match = re.search(r'(?:within|in|by|before)\s+(\d+)\s+days', text)
        if days_match:
            num_days = int(days_match.group(1))
            target_date = base_date + timedelta(days=num_days)
            today = datetime.now()
            days_remaining = (target_date - today).days
            return target_date.strftime("%Y-%m-%d"), max(0, days_remaining)

        # Check for relative months: "within X months"
        months_match = re.search(r'(?:within|in|by|before)\s+(\d+)\s+months', text)
        if months_match:
            num_months = int(months_match.group(1))
            target_date = base_date + timedelta(days=num_months * 30)
            today = datetime.now()
            days_remaining = (target_date - today).days
            return target_date.strftime("%Y-%m-%d"), max(0, days_remaining)

        # Check for explicit date strings e.g. "30th September 2026" or "2026-10-15"
        date_explicit = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})|(\d{4})[/-](\d{1,2})[/-](\d{1,2})', text)
        if date_explicit:
            try:
                g = date_explicit.groups()
                if g[0]:
                    dt = datetime(int(g[2]), int(g[1]), int(g[0]))
                else:
                    dt = datetime(int(g[3]), int(g[4]), int(g[5]))
                days_rem = (dt - datetime.now()).days
                return dt.strftime("%Y-%m-%d"), max(0, days_rem)
            except Exception:
                pass

        # Month name explicit e.g. "30 September 2026" or "September 30, 2026"
        month_names = r'(january|february|march|april|may|june|july|august|september|october|november|december)'
        month_explicit = re.search(rf'(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_names}\s+(\d{{4}})', text)
        if month_explicit:
            try:
                day = int(month_explicit.group(1))
                m_str = month_explicit.group(2).capitalize()
                year = int(month_explicit.group(3))
                dt = datetime.strptime(f"{day} {m_str} {year}", "%d %B %Y")
                days_rem = (dt - datetime.now()).days
                return dt.strftime("%Y-%m-%d"), max(0, days_rem)
            except Exception:
                pass

        # Fallback default: 14 days from notice date
        target_date = base_date + timedelta(days=14)
        days_rem = (target_date - datetime.now()).days
        return target_date.strftime("%Y-%m-%d"), max(0, days_rem)
